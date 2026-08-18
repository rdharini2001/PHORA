from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from scipy import ndimage, stats
import nibabel as nib
from nibabel.processing import resample_from_to

RAD_SPACING_MM=1.0
RAD_BIN_WIDTH_HU=25.0
RAD_CT_MIN_HU=-1024.0
RAD_CT_MAX_HU=3071.0
RAD_MIN_VOXELS=10
RAD_CROP_PAD_MM=5.0
PHASES=("ART","VEN","DEL","DRY")
DIRS13=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,-1,0),(1,0,1),(1,0,-1),(0,1,1),(0,1,-1),(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1)]
NEIGH26=[(z,y,x) for z in (-1,0,1) for y in (-1,0,1) for x in (-1,0,1) if (z,y,x)!=(0,0,0)]

def _safe_div(a, b):
    return float(a / b) if np.isfinite(b) and abs(float(b)) > 1e-12 else np.nan


def _finite(x):
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def _discover_cases(root):
    out = {}
    for p in Path(root).rglob("lesion.nii.gz"):
        case_dir = p.parent.parent
        if (case_dir / "ct").is_dir():
            out[case_dir.name] = case_dir
    if not out:
        for p in Path(root).rglob("lesion.nii"):
            case_dir = p.parent.parent
            if (case_dir / "ct").is_dir():
                out[case_dir.name] = case_dir
    return out


def _find_phase(case_dir, case_id, phase):
    ct = case_dir / "ct"
    for suffix in (".nii.gz", ".nii"):
        p = ct / f"{case_id}_{phase}{suffix}"
        if p.exists():
            return p
    matches = sorted(ct.glob(f"*_{phase}.nii*"))
    return matches[0] if matches else None


def _read_binary_mask(path):
    m = sitk.ReadImage(str(path))
    return sitk.Cast(m > 0, sitk.sitkUInt8)


def _align_mask(mask, image):
    return sitk.Resample(
        mask, image, sitk.Transform(), sitk.sitkNearestNeighbor,
        0, sitk.sitkUInt8
    )


def _crop_to_mask(image, mask, pad_mm=5.0):
    a = sitk.GetArrayViewFromImage(mask) > 0
    if not np.any(a):
        raise ValueError("empty mask after alignment")
    zz, yy, xx = np.where(a)
    spacing = np.asarray(image.GetSpacing(), float)  # x,y,z
    pad_xyz = np.maximum(1, np.ceil(float(pad_mm) / spacing).astype(int))

    x0 = max(0, int(xx.min()) - int(pad_xyz[0]))
    y0 = max(0, int(yy.min()) - int(pad_xyz[1]))
    z0 = max(0, int(zz.min()) - int(pad_xyz[2]))
    x1 = min(image.GetSize()[0], int(xx.max()) + 1 + int(pad_xyz[0]))
    y1 = min(image.GetSize()[1], int(yy.max()) + 1 + int(pad_xyz[1]))
    z1 = min(image.GetSize()[2], int(zz.max()) + 1 + int(pad_xyz[2]))

    index = [x0, y0, z0]
    size = [x1-x0, y1-y0, z1-z0]
    return sitk.RegionOfInterest(image, size=size, index=index), sitk.RegionOfInterest(mask, size=size, index=index)


def _resample_isotropic(image, mask, spacing_mm=1.0):
    old_sp = np.asarray(image.GetSpacing(), float)
    old_sz = np.asarray(image.GetSize(), int)
    new_sp = np.array([spacing_mm]*3, float)
    new_sz = np.maximum(1, np.rint(old_sz * old_sp / new_sp).astype(int))

    def _resample(im, interp, default):
        f = sitk.ResampleImageFilter()
        f.SetOutputSpacing(tuple(new_sp.tolist()))
        f.SetSize([int(v) for v in new_sz])
        f.SetOutputOrigin(image.GetOrigin())
        f.SetOutputDirection(image.GetDirection())
        f.SetTransform(sitk.Transform())
        f.SetInterpolator(interp)
        f.SetDefaultPixelValue(default)
        return f.Execute(im)

    image_r = _resample(image, sitk.sitkBSpline, -1024.0)
    mask_r = _resample(mask, sitk.sitkNearestNeighbor, 0)
    return sitk.Cast(image_r, sitk.sitkFloat32), sitk.Cast(mask_r > 0, sitk.sitkUInt8)


def _prepare_phase(image_path, native_mask):
    img = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    m = _align_mask(native_mask, img)
    img, m = _crop_to_mask(img, m, RAD_CROP_PAD_MM)
    img, m = _resample_isotropic(img, m, RAD_SPACING_MM)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)   # z,y,x
    roi = sitk.GetArrayFromImage(m) > 0
    roi &= np.isfinite(arr)
    if int(roi.sum()) < RAD_MIN_VOXELS:
        raise ValueError(f"ROI too small after resampling: {int(roi.sum())} voxels")
    return arr, roi


def _discretize_ct(arr, mask):
    """Fixed-bin-width discretization with a fixed CT origin for reproducibility."""
    q = np.zeros(arr.shape, dtype=np.int16)
    x = np.clip(arr[mask], RAD_CT_MIN_HU, RAD_CT_MAX_HU)
    bins = np.floor((x - RAD_CT_MIN_HU) / RAD_BIN_WIDTH_HU).astype(np.int16) + 1
    q[mask] = bins
    return q, int(bins.max())


def _shape_features(mask, spacing_zyx):
    n = int(mask.sum())
    if n == 0:
        return {}
    sp = np.asarray(spacing_zyx, float)
    voxel_volume = float(np.prod(sp))
    volume = n * voxel_volume

    # Surface area from exposed voxel faces; spacing-aware and deterministic.
    surface = 0.0
    for ax in range(3):
        pad = [(0,0)] * 3
        pad[ax] = (1,1)
        transitions = np.abs(np.diff(np.pad(mask.astype(np.int8), pad, mode="constant"), axis=ax)).sum()
        face_area = float(np.prod(np.delete(sp, ax)))
        surface += float(transitions) * face_area

    coords = np.argwhere(mask).astype(float) * sp[None, :]
    ext = coords.max(0) - coords.min(0) + sp
    eq_d = (6.0 * volume / np.pi) ** (1.0/3.0)
    sphericity = (np.pi ** (1.0/3.0) * (6.0*volume) ** (2.0/3.0) / surface) if surface > 0 else np.nan

    if n >= 4:
        eig = np.sort(np.linalg.eigvalsh(np.cov(coords, rowvar=False)))[::-1]
        elongation = np.sqrt(eig[1] / eig[0]) if eig[0] > 0 and eig[1] >= 0 else np.nan
        flatness = np.sqrt(eig[2] / eig[0]) if eig[0] > 0 and eig[2] >= 0 else np.nan
    else:
        elongation = flatness = np.nan

    return {
        "shape_VoxelCount": n,
        "shape_Volume_mm3": volume,
        "shape_SurfaceArea_mm2": surface,
        "shape_SurfaceVolumeRatio": _safe_div(surface, volume),
        "shape_Sphericity": float(sphericity),
        "shape_EquivalentDiameter_mm": float(eq_d),
        "shape_MaxBBoxDiameter_mm": float(np.max(ext)),
        "shape_BBoxDiagonal_mm": float(np.linalg.norm(ext)),
        "shape_Elongation": float(elongation),
        "shape_Flatness": float(flatness),
    }


def _firstorder(arr, mask):
    x = _finite(arr[mask])
    if x.size == 0:
        return {}
    p10, p25, p75, p90 = np.percentile(x, [10,25,75,90])
    med = float(np.median(x))
    mad = float(np.median(np.abs(x-med)))
    trimmed = x[(x >= p10) & (x <= p90)]
    rmed = float(np.median(trimmed)) if trimmed.size else np.nan
    rmad = float(np.median(np.abs(trimmed-rmed))) if trimmed.size else np.nan
    hist, _ = np.histogram(x, bins=min(128, max(16, int(np.sqrt(x.size)))))
    p = hist[hist > 0].astype(float)
    p /= p.sum()
    mu = float(np.mean(x)); sd = float(np.std(x))
    return {
        "original_firstorder_Mean": mu,
        "original_firstorder_Median": med,
        "original_firstorder_Minimum": float(np.min(x)),
        "original_firstorder_Maximum": float(np.max(x)),
        "original_firstorder_Range": float(np.ptp(x)),
        "original_firstorder_Variance": float(np.var(x)),
        "original_firstorder_StandardDeviation": sd,
        "original_firstorder_Skewness": float(stats.skew(x, bias=False)) if x.size > 2 else np.nan,
        "original_firstorder_Kurtosis": float(stats.kurtosis(x, bias=False, fisher=False)) if x.size > 3 else np.nan,
        "original_firstorder_10Percentile": float(p10),
        "original_firstorder_25Percentile": float(p25),
        "original_firstorder_75Percentile": float(p75),
        "original_firstorder_90Percentile": float(p90),
        "original_firstorder_InterquartileRange": float(p75-p25),
        "original_firstorder_MeanAbsoluteDeviation": float(np.mean(np.abs(x-mu))),
        "original_firstorder_MedianAbsoluteDeviation": mad,
        "original_firstorder_RobustMedianAbsoluteDeviation": rmad,
        "original_firstorder_Energy": float(np.sum(x*x)),
        "original_firstorder_RootMeanSquared": float(np.sqrt(np.mean(x*x))),
        "original_firstorder_Entropy": float(-np.sum(p*np.log2(p))),
        "original_firstorder_CoefficientOfVariation": _safe_div(sd, abs(mu)),
    }


def _neighbor(arr, offset, fill=0):
    """out[v] = arr[v + offset], with constant fill outside."""
    out = np.full_like(arr, fill)
    src, dst = [], []
    for n, o in zip(arr.shape, offset):
        if o >= 0:
            src.append(slice(o, n)); dst.append(slice(0, n-o))
        else:
            src.append(slice(0, n+o)); dst.append(slice(-o, n))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def _glcm(q, mask, G):
    P = np.zeros((G,G), dtype=np.float64)
    Z,Y,X = q.shape
    for dz,dy,dx in DIRS13:
        z0,z1=max(0,-dz),min(Z,Z-dz)
        y0,y1=max(0,-dy),min(Y,Y-dy)
        x0,x1=max(0,-dx),min(X,X-dx)
        a=q[z0:z1,y0:y1,x0:x1]
        b=q[z0+dz:z1+dz,y0+dy:y1+dy,x0+dx:x1+dx]
        m=mask[z0:z1,y0:y1,x0:x1] & mask[z0+dz:z1+dz,y0+dy:y1+dy,x0+dx:x1+dx]
        if not np.any(m):
            continue
        ai=a[m]-1; bi=b[m]-1
        np.add.at(P,(ai,bi),1)
        np.add.at(P,(bi,ai),1)  # symmetric aggregation
    if P.sum() == 0:
        return {}
    P /= P.sum()
    i=np.arange(1,G+1,dtype=float)[:,None]
    j=np.arange(1,G+1,dtype=float)[None,:]
    d=i-j
    mui=float((P*i).sum()); muj=float((P*j).sum())
    si=float(np.sqrt((P*(i-mui)**2).sum())); sj=float(np.sqrt((P*(j-muj)**2).sum()))
    nz=P[P>0]
    asm=float(np.sum(P*P))
    return {
        "original_glcm_Contrast": float(np.sum(P*d*d)),
        "original_glcm_Dissimilarity": float(np.sum(P*np.abs(d))),
        "original_glcm_Idm": float(np.sum(P/(1.0+d*d))),
        "original_glcm_JointEnergy": asm,
        "original_glcm_Energy": float(np.sqrt(asm)),
        "original_glcm_Correlation": _safe_div(np.sum(P*(i-mui)*(j-muj)), si*sj),
        "original_glcm_JointEntropy": float(-np.sum(nz*np.log2(nz))),
        "original_glcm_JointAverage": mui,
        "original_glcm_ClusterShade": float(np.sum(P*(i+j-mui-muj)**3)),
        "original_glcm_ClusterProminence": float(np.sum(P*(i+j-mui-muj)**4)),
    }


def _matrix_features(P, nvox, kind):
    total=float(P.sum())
    if total <= 0:
        return {}
    pg=P.sum(axis=1); ps=P.sum(axis=0)
    g=np.arange(1,P.shape[0]+1,dtype=float)[:,None]
    s=np.arange(1,P.shape[1]+1,dtype=float)[None,:]

    if kind == "rl":
        return {
            "original_glrlm_ShortRunEmphasis": float(np.sum(P/s**2)/total),
            "original_glrlm_LongRunEmphasis": float(np.sum(P*s**2)/total),
            "original_glrlm_GrayLevelNonUniformity": float(np.sum(pg**2)/total),
            "original_glrlm_GrayLevelNonUniformityNormalized": float(np.sum(pg**2)/total**2),
            "original_glrlm_RunLengthNonUniformity": float(np.sum(ps**2)/total),
            "original_glrlm_RunLengthNonUniformityNormalized": float(np.sum(ps**2)/total**2),
            "original_glrlm_RunPercentage": _safe_div(total,nvox),
            "original_glrlm_LowGrayLevelRunEmphasis": float(np.sum(P/g**2)/total),
            "original_glrlm_HighGrayLevelRunEmphasis": float(np.sum(P*g**2)/total),
            "original_glrlm_ShortRunLowGrayLevelEmphasis": float(np.sum(P/(s**2*g**2))/total),
            "original_glrlm_ShortRunHighGrayLevelEmphasis": float(np.sum(P*g**2/s**2)/total),
            "original_glrlm_LongRunLowGrayLevelEmphasis": float(np.sum(P*s**2/g**2)/total),
            "original_glrlm_LongRunHighGrayLevelEmphasis": float(np.sum(P*s**2*g**2)/total),
        }
    if kind == "sz":
        return {
            "original_glszm_SmallAreaEmphasis": float(np.sum(P/s**2)/total),
            "original_glszm_LargeAreaEmphasis": float(np.sum(P*s**2)/total),
            "original_glszm_GrayLevelNonUniformity": float(np.sum(pg**2)/total),
            "original_glszm_GrayLevelNonUniformityNormalized": float(np.sum(pg**2)/total**2),
            "original_glszm_SizeZoneNonUniformity": float(np.sum(ps**2)/total),
            "original_glszm_SizeZoneNonUniformityNormalized": float(np.sum(ps**2)/total**2),
            "original_glszm_ZonePercentage": _safe_div(total,nvox),
            "original_glszm_LowGrayLevelZoneEmphasis": float(np.sum(P/g**2)/total),
            "original_glszm_HighGrayLevelZoneEmphasis": float(np.sum(P*g**2)/total),
            "original_glszm_SmallAreaLowGrayLevelEmphasis": float(np.sum(P/(s**2*g**2))/total),
            "original_glszm_SmallAreaHighGrayLevelEmphasis": float(np.sum(P*g**2/s**2)/total),
            "original_glszm_LargeAreaLowGrayLevelEmphasis": float(np.sum(P*s**2/g**2)/total),
            "original_glszm_LargeAreaHighGrayLevelEmphasis": float(np.sum(P*s**2*g**2)/total),
        }
    if kind == "dm":
        return {
            "original_gldm_SmallDependenceEmphasis": float(np.sum(P/s**2)/total),
            "original_gldm_LargeDependenceEmphasis": float(np.sum(P*s**2)/total),
            "original_gldm_GrayLevelNonUniformity": float(np.sum(pg**2)/total),
            "original_gldm_GrayLevelNonUniformityNormalized": float(np.sum(pg**2)/total**2),
            "original_gldm_DependenceNonUniformity": float(np.sum(ps**2)/total),
            "original_gldm_DependenceNonUniformityNormalized": float(np.sum(ps**2)/total**2),
            "original_gldm_LowGrayLevelEmphasis": float(np.sum(P/g**2)/total),
            "original_gldm_HighGrayLevelEmphasis": float(np.sum(P*g**2)/total),
            "original_gldm_SmallDependenceLowGrayLevelEmphasis": float(np.sum(P/(s**2*g**2))/total),
            "original_gldm_SmallDependenceHighGrayLevelEmphasis": float(np.sum(P*g**2/s**2)/total),
            "original_gldm_LargeDependenceLowGrayLevelEmphasis": float(np.sum(P*s**2/g**2)/total),
            "original_gldm_LargeDependenceHighGrayLevelEmphasis": float(np.sum(P*s**2*g**2)/total),
        }
    return {}


def _glrlm(q, mask, G):
    counts={}; Z,Y,X=q.shape; nvox=int(mask.sum())
    for dz,dy,dx in DIRS13:
        prev=_neighbor(mask,(-dz,-dy,-dx),False)
        starts=np.argwhere(mask & ~prev)
        for z,y,x in starts:
            z=int(z); y=int(y); x=int(x)
            cz,cy,cx=z,y,x; cur=int(q[z,y,x]); run=0
            while 0<=cz<Z and 0<=cy<Y and 0<=cx<X and mask[cz,cy,cx]:
                g=int(q[cz,cy,cx])
                if g == cur:
                    run += 1
                else:
                    counts[(cur,run)] = counts.get((cur,run),0) + 1
                    cur=g; run=1
                cz+=dz; cy+=dy; cx+=dx
            counts[(cur,run)] = counts.get((cur,run),0) + 1
    if not counts:
        return {}
    S=max(r for _,r in counts)
    P=np.zeros((G,S),float)
    for (g,r),v in counts.items():
        P[g-1,r-1]+=v
    return _matrix_features(P,nvox,"rl")


def _glszm(q, mask, G):
    counts={}; structure=np.ones((3,3,3),np.uint8)
    for g in np.unique(q[mask]):
        lab,n=ndimage.label(mask & (q==g), structure=structure)
        if n == 0:
            continue
        sizes=np.bincount(lab.ravel())[1:]
        for s in sizes:
            counts[(int(g),int(s))]=counts.get((int(g),int(s)),0)+1
    if not counts:
        return {}
    S=max(s for _,s in counts)
    P=np.zeros((G,S),float)
    for (g,s),v in counts.items():
        P[g-1,s-1]+=v
    return _matrix_features(P,int(mask.sum()),"sz")


def _gldm(q, mask, G, alpha=0):
    dep=np.ones(q.shape,np.int16)
    Z,Y,X=q.shape
    for dz,dy,dx in NEIGH26:
        z0,z1=max(0,-dz),min(Z,Z-dz)
        y0,y1=max(0,-dy),min(Y,Y-dy)
        x0,x1=max(0,-dx),min(X,X-dx)
        a=q[z0:z1,y0:y1,x0:x1]
        b=q[z0+dz:z1+dz,y0+dy:y1+dy,x0+dx:x1+dx]
        m=mask[z0:z1,y0:y1,x0:x1] & mask[z0+dz:z1+dz,y0+dy:y1+dy,x0+dx:x1+dx]
        dep[z0:z1,y0:y1,x0:x1] += (m & (np.abs(a-b)<=alpha)).astype(np.int16)
    gs=q[mask].astype(int); ds=dep[mask].astype(int)
    P=np.zeros((G,int(ds.max())),float)
    np.add.at(P,(gs-1,ds-1),1)
    return _matrix_features(P,int(mask.sum()),"dm")


def _ngtdm(q, mask, G):
    kernel=np.ones((3,3,3),float); kernel[1,1,1]=0
    cnt=ndimage.convolve(mask.astype(float),kernel,mode="constant",cval=0.0)
    sm=ndimage.convolve((q*mask).astype(float),kernel,mode="constant",cval=0.0)
    valid=mask & (cnt>0)
    if not np.any(valid):
        return {}
    avg=np.zeros(q.shape,float); avg[valid]=sm[valid]/cnt[valid]
    n=np.zeros(G,float); s=np.zeros(G,float)
    for g in range(1,G+1):
        m=valid & (q==g)
        n[g-1]=m.sum()
        if n[g-1] > 0:
            s[g-1]=np.abs(g-avg[m]).sum()
    Nv=float(n.sum())
    if Nv <= 0:
        return {}
    p=n/Nv; idx=np.where(n>0)[0]; gi=np.arange(1,G+1,dtype=float)
    coarseness=1.0/(np.sum(p*s)+1e-12)
    denom=max(len(idx)*(len(idx)-1),1)
    contrast=float(np.sum((p[:,None]*p[None,:])*(gi[:,None]-gi[None,:])**2)/denom * (s.sum()/Nv))
    busy_den=sum(abs(gi[i]*p[i]-gi[j]*p[j]) for i in idx for j in idx)
    busyness=float(np.sum(p*s)/(busy_den+1e-12))
    complexity=0.0; strength_num=0.0
    for i in idx:
        for j in idx:
            if i == j: continue
            complexity += abs(gi[i]-gi[j])*(p[i]*s[i]+p[j]*s[j])/(p[i]+p[j]+1e-12)
            strength_num += (p[i]+p[j])*(gi[i]-gi[j])**2
    return {
        "original_ngtdm_Coarseness": float(coarseness),
        "original_ngtdm_Contrast": contrast,
        "original_ngtdm_Busyness": busyness,
        "original_ngtdm_Complexity": float(complexity/Nv),
        "original_ngtdm_Strength": float(strength_num/(s.sum()+1e-12)),
    }


def _texture_features(arr, mask):
    q,G=_discretize_ct(arr,mask)
    out={}
    out.update(_glcm(q,mask,G))
    out.update(_glrlm(q,mask,G))
    out.update(_glszm(q,mask,G))
    out.update(_gldm(q,mask,G,alpha=0))
    out.update(_ngtdm(q,mask,G))
    return out


def _native_shape(native_mask):
    a=sitk.GetArrayFromImage(native_mask)>0
    spacing_zyx=tuple(reversed(native_mask.GetSpacing()))
    return _shape_features(a,spacing_zyx)


def _extract_case(case_id, case_dir):
    row={"case_id":str(case_id)}; failures=[]
    lesion_path=case_dir/"annotations"/"lesion.nii.gz"
    if not lesion_path.exists():
        lesion_path=case_dir/"annotations"/"lesion.nii"
    try:
        native_mask=_read_binary_mask(lesion_path)
        row.update(_native_shape(native_mask))
    except Exception as e:
        return row,[{"case_id":case_id,"component":"shape","error":repr(e)}]

    for phase in PHASES:
        p=_find_phase(case_dir,case_id,phase)
        row[f"phase_present_{phase}"]=int(p is not None)
        if p is None:
            continue
        try:
            arr,mask=_prepare_phase(p,native_mask)
            row[f"{phase}_roi_voxels"]=int(mask.sum())
            feats={}
            feats.update(_firstorder(arr,mask))
            feats.update(_texture_features(arr,mask))
            for k,v in feats.items():
                row[f"{phase}_{k}"]=float(v) if np.isscalar(v) else np.nan
        except Exception as e:
            failures.append({"case_id":case_id,"component":phase,"error":repr(e)})
    return row,failures


def _add_phase_deltas(df):
    pairs=(("ART","DRY"),("VEN","ART"),("DEL","ART"),("DEL","VEN"))
    suffixes=set()
    for c in df.columns:
        for ph in PHASES:
            prefix=f"{ph}_original_firstorder_"
            if c.startswith(prefix):
                suffixes.add(c[len(ph)+1:])
    add={}
    for left,right in pairs:
        for suf in suffixes:
            a=f"{left}_{suf}"; b=f"{right}_{suf}"
            if a in df.columns and b in df.columns:
                add[f"DELTA_{left}_MINUS_{right}_{suf}"] = pd.to_numeric(df[a],errors="coerce") - pd.to_numeric(df[b],errors="coerce")
    if add:
        df=pd.concat([df,pd.DataFrame(add,index=df.index)],axis=1)
    return df



# ---- portable NIfTI/SciPy replacements for the development SimpleITK preparation ----
def _voxel_sizes(affine):
    return np.sqrt((np.asarray(affine,float)[:3,:3]**2).sum(axis=0))

def _read_binary_mask(path):
    m=nib.load(str(path))
    a=np.asarray(m.dataobj)>0
    return nib.Nifti1Image(a.astype(np.uint8),m.affine)

def _align_mask(mask,image):
    if tuple(mask.shape[:3])==tuple(image.shape[:3]) and np.allclose(mask.affine,image.affine,atol=1e-5):
        return nib.Nifti1Image((np.asarray(mask.dataobj)>0).astype(np.uint8), image.affine)
    return resample_from_to(mask,(image.shape[:3],image.affine),order=0,mode="constant",cval=0.0)

def _crop_to_mask_np(image,mask,pad_mm=5.0):
    a=np.asarray(mask.dataobj)>0
    if not np.any(a): raise ValueError("empty mask after alignment")
    xx,yy,zz=np.where(a)
    spacing=_voxel_sizes(image.affine)
    pad=np.maximum(1,np.ceil(float(pad_mm)/spacing).astype(int))
    lo=np.maximum([xx.min(),yy.min(),zz.min()]-pad,0).astype(int)
    hi=np.minimum([xx.max()+1,yy.max()+1,zz.max()+1]+pad,np.asarray(image.shape[:3])).astype(int)
    sl=tuple(slice(int(lo[i]),int(hi[i])) for i in range(3))
    T=np.eye(4); T[:3,3]=lo
    A=np.asarray(image.affine,float)@T
    return nib.Nifti1Image(np.asarray(image.dataobj)[sl],A), nib.Nifti1Image(np.asarray(mask.dataobj)[sl],A)

def _resample_isotropic_np(image,mask,spacing_mm=1.0):
    old_sp=_voxel_sizes(image.affine); old_sz=np.asarray(image.shape[:3],int)
    new_sz=np.maximum(1,np.rint(old_sz*old_sp/float(spacing_mm)).astype(int))
    A=np.asarray(image.affine,float).copy()
    dirs=A[:3,:3]/np.where(old_sp>0,old_sp,1.0)[None,:]
    A[:3,:3]=dirs*float(spacing_mm)
    ir=resample_from_to(image,(tuple(new_sz.tolist()),A),order=3,mode="constant",cval=-1024.0)
    mr=resample_from_to(mask,(tuple(new_sz.tolist()),A),order=0,mode="constant",cval=0.0)
    return ir,mr

def _prepare_phase(image_path,native_mask):
    img=nib.load(str(image_path)); m=_align_mask(native_mask,img)
    img,m=_crop_to_mask_np(img,m,RAD_CROP_PAD_MM)
    img,m=_resample_isotropic_np(img,m,RAD_SPACING_MM)
    # feature implementation below follows the SimpleITK z,y,x array convention
    arr=np.asarray(img.get_fdata(dtype=np.float32),dtype=np.float32).transpose(2,1,0)
    roi=(np.asarray(m.dataobj)>0).transpose(2,1,0)
    roi &= np.isfinite(arr)
    if int(roi.sum())<RAD_MIN_VOXELS: raise ValueError(f"ROI too small after resampling: {int(roi.sum())} voxels")
    return arr,roi

def _native_shape(native_mask):
    a=(np.asarray(native_mask.dataobj)>0).transpose(2,1,0)
    spacing_xyz=_voxel_sizes(native_mask.affine)
    return _shape_features(a,tuple(spacing_xyz[::-1]))

def _extract_case(case_id,case_dir):
    case_dir=Path(case_dir); row={"case_id":str(case_id)}; failures=[]
    lesion_path=case_dir/"annotations"/"lesion.nii.gz"
    if not lesion_path.exists(): lesion_path=case_dir/"annotations"/"lesion.nii"
    try:
        native_mask=_read_binary_mask(lesion_path); row.update(_native_shape(native_mask))
    except Exception as e:
        return row,[{"case_id":case_id,"component":"shape","error":repr(e)}]
    phase_first={}
    for phase in PHASES:
        p=_find_phase(case_dir,case_id,phase); row[f"phase_present_{phase}"]=int(p is not None)
        if p is None: continue
        try:
            arr,mask=_prepare_phase(p,native_mask); row[f"{phase}_roi_voxels"]=int(mask.sum())
            feats={}; feats.update(_firstorder(arr,mask)); feats.update(_texture_features(arr,mask)); phase_first[phase]=feats
            for k,v in feats.items(): row[f"{phase}_{k}"]=float(v) if np.isscalar(v) else np.nan
        except Exception as e:
            failures.append({"case_id":case_id,"component":phase,"error":repr(e)})
    # phase-difference first-order descriptors
    pairs=(("ART","DRY"),("VEN","ART"),("DEL","ART"),("DEL","VEN"))
    suffixes=set()
    for ph,feats in phase_first.items():
        suffixes.update(k for k in feats if k.startswith("original_firstorder_"))
    for left,right in pairs:
        if left not in phase_first or right not in phase_first: continue
        for suf in suffixes:
            if suf in phase_first[left] and suf in phase_first[right]:
                row[f"DELTA_{left}_MINUS_{right}_{suf}"]=float(phase_first[left][suf]-phase_first[right][suf])
    return row,failures

def extract_case_radiomics(case_dir,case_id):
    return _extract_case(str(case_id),Path(case_dir))[0]
