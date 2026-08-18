from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
EPS=1e-8

def robust_stats(v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    if len(v)<8: return (np.nan,)*4
    lo,hi=np.percentile(v,[2.5,97.5]); v=np.clip(v,lo,hi)
    q10,q25,q50,q75,q90=np.percentile(v,[10,25,50,75,90])
    return float(q50),float(q75-q25),float(q10),float(q90)


def moran_geary(arr,mask):
    x=np.asarray(arr,float); m=mask & np.isfinite(x); vals=x[m]
    if len(vals)<20 or np.var(vals)<1e-8: return np.nan,np.nan
    mu=vals.mean(); denom=np.sum((vals-mu)**2); cross=0.; sq=0.; W=0
    for ax in range(3):
        a=[slice(None)]*3; b=[slice(None)]*3; a[ax]=slice(1,None); b[ax]=slice(None,-1)
        mm=m[tuple(a)]&m[tuple(b)]; xa=x[tuple(a)][mm]; xb=x[tuple(b)][mm]
        if len(xa):
            cross+=2*np.sum((xa-mu)*(xb-mu)); sq+=2*np.sum((xa-xb)**2); W+=2*len(xa)
    N=len(vals)
    return float(N*cross/(W*denom+EPS)), float((N-1)*sq/(2*W*denom+EPS))


def hot_topology(arr,mask):
    vals=arr[mask & np.isfinite(arr)]
    if len(vals)<30: return (np.nan,)*4
    comps=[]; lfrac=[]
    for q in [60,70,80,90]:
        th=np.percentile(vals,q); hot=mask & np.isfinite(arr) & (arr>=th)
        lab,n=ndimage.label(hot,structure=ndimage.generate_binary_structure(3,1)); comps.append(n)
        if n:
            cnt=np.bincount(lab.ravel())[1:]; lfrac.append(float(cnt.max()/(hot.sum()+EPS)))
        else: lfrac.append(0.)
    return float(np.mean(comps)),float(np.std(comps)),float(np.mean(lfrac)),float(lfrac[-1])


def perivascular_contact(arr,mask,spacing,dout):
    ring=(~mask)&(dout>0)&(dout<=12)&np.isfinite(arr)
    vals=arr[ring]
    if len(vals)<50: return (np.nan,)*3
    th=np.percentile(vals,90); hot=ring&(arr>=th); contact=hot&(dout<=3.0)
    contact_fraction=float(contact.sum()/(((~mask)&(dout<=3.0)).sum()+EPS))
    lab,n=ndimage.label(hot,structure=ndimage.generate_binary_structure(3,1))
    ids=np.unique(lab[contact]); ids=ids[ids>0]
    if len(ids)==0: return contact_fraction,0.0,np.nan
    counts=[np.sum(lab==i) for i in ids]; best=int(ids[int(np.argmax(counts))]); pts=np.argwhere(lab==best)*spacing[None,:]
    frac=float(len(pts)/(ring.sum()+EPS))
    if len(pts)>=5:
        ev=np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
        elong=float(np.sqrt((ev[0]+EPS)/(ev[1]+EPS)))
    else: elong=np.nan
    return contact_fraction,frac,elong


def extract_context_ot_case(case_dir,case_id):
    from phora_features import _load_case_on_art_grid
    d=_load_case_on_art_grid(Path(case_dir),case_id,margin_vox=14)
    if d is None: return {'case_id':case_id}
    mask=d['mask']; sp=np.asarray(d['spacing'],float); imgs=d['images']
    din=ndimage.distance_transform_edt(mask,sampling=sp); dout=ndimage.distance_transform_edt(~mask,sampling=sp)
    outer03=(~mask)&(dout>0)&(dout<=3); bg=(~mask)&(dout>=3)&(dout<=12)
    inner03=mask&(din<=3); maxd=float(np.max(din[mask])) if mask.any() else 0
    core=mask&(din>=max(3.0,.55*maxd))
    out={'case_id':case_id}; rel={}; lesvals={}
    for ph,arr in imgs.items():
        if arr is None: continue
        lv=arr[mask & np.isfinite(arr)]; bv=arr[bg & np.isfinite(arr)]
        lm,li,lp10,lp90=robust_stats(lv); bm,bi,bp10,bp90=robust_stats(bv)
        im,*_=robust_stats(arr[inner03 & np.isfinite(arr)]); om,*_=robust_stats(arr[outer03 & np.isfinite(arr)]); cm,*_=robust_stats(arr[core & np.isfinite(arr)])
        out[f'ctx_{ph}_lesion_median']=lm; out[f'ctx_{ph}_background_median']=bm
        out[f'ctx_{ph}_lesion_minus_bg']=lm-bm; out[f'ctx_{ph}_inner_minus_outer']=im-om; out[f'ctx_{ph}_rim_minus_core']=im-cm
        out[f'ctx_{ph}_lesion_iqr']=li; out[f'ctx_{ph}_bg_iqr']=bi
        mor,gea=moran_geary(arr,mask); out[f'ctx_{ph}_moranI']=mor; out[f'ctx_{ph}_gearyC']=gea
        tc,ts,lf,l90=hot_topology(arr,mask); out[f'ctx_{ph}_topo_components_mean']=tc; out[f'ctx_{ph}_topo_components_sd']=ts; out[f'ctx_{ph}_hot_largest_frac_mean']=lf; out[f'ctx_{ph}_hot_largest_frac_q90']=l90
        cf,vf,ve=perivascular_contact(arr,mask,sp,dout); out[f'ctx_{ph}_vascular_contact']=cf; out[f'ctx_{ph}_vascular_component_frac']=vf; out[f'ctx_{ph}_vascular_elongation']=ve
        rel[ph]=lm-bm; lesvals[ph]=lv
    # Background-corrected kinetic features.
    pairs=[('ART','DRY'),('ART','VEN'),('ART','DEL'),('VEN','DEL')]
    for a,b in pairs:
        if a in rel and b in rel:
            out[f'ctx_rel_{a}_minus_{b}']=rel[a]-rel[b]
        if a in lesvals and b in lesvals:
            va,vb=lesvals[a],lesvals[b]
            if len(va)>20 and len(vb)>20:
                # subsample deterministic quantiles if huge
                if len(va)>5000: va=np.quantile(va,np.linspace(0,1,5000))
                if len(vb)>5000: vb=np.quantile(vb,np.linspace(0,1,5000))
                out[f'ot_W1_{a}_{b}']=float(wasserstein_distance(va,vb))
                lo=min(np.percentile(va,1),np.percentile(vb,1)); hi=max(np.percentile(va,99),np.percentile(vb,99))
                if hi>lo:
                    bins=np.linspace(lo,hi,49); ha,_=np.histogram(va,bins=bins,density=True); hb,_=np.histogram(vb,bins=bins,density=True)
                    ha=ha+EPS; hb=hb+EPS; ha/=ha.sum(); hb/=hb.sum()
                    out[f'ot_JSD_{a}_{b}']=float(jensenshannon(ha,hb,base=2.0)**2)
    if 'ART' in rel and 'VEN' in rel: out['phys_APHE_washout_axis']=rel['ART']-rel['VEN']
    if 'ART' in rel and 'DEL' in rel: out['phys_APHE_delayed_axis']=rel['ART']-rel['DEL']
    return out

