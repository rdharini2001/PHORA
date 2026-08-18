from __future__ import annotations

"""PHORA-LI inference-safe spatial-temporal feature extraction.

Designed for AMPLIFAI multi-phase CT with only CT volumes + lesion mask available
at inference. Dependencies: numpy, scipy, nibabel.

The extractor intentionally does NOT use LI-RADS labels or feature annotation masks
as predictors. Training-only feature masks can be summarized separately as auxiliary
supervision targets.
"""

import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.stats import skew, kurtosis

try:
    import nibabel as nib
    from nibabel.processing import resample_from_to
except Exception as e:  # pragma: no cover
    nib = None
    resample_from_to = None
    _NIB_IMPORT_ERROR = e
else:
    _NIB_IMPORT_ERROR = None

PHASES = ("DRY", "ART", "VEN", "DEL")
EPS = 1e-8


def _require_nibabel() -> None:
    if nib is None:
        raise ImportError(
            "nibabel is required for PHORA feature extraction. Install nibabel==5.2.1."
        ) from _NIB_IMPORT_ERROR


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.is_file():
            return p
    return None


def find_phase(case_dir: Path, case_id: str, phase: str) -> Optional[Path]:
    ct_dir = case_dir / "ct"
    exact = _first_existing([
        ct_dir / f"{case_id}_{phase}.nii.gz",
        ct_dir / f"{case_id}_{phase}.nii",
    ])
    if exact is not None:
        return exact
    matches = sorted(list(ct_dir.glob(f"*_{phase}.nii.gz")) + list(ct_dir.glob(f"*_{phase}.nii")))
    return matches[0] if matches else None


def find_lesion_mask(case_dir: Path) -> Optional[Path]:
    ann = case_dir / "annotations"
    exact = _first_existing([ann / "lesion.nii.gz", ann / "lesion.nii"])
    if exact is not None:
        return exact
    matches = sorted(list(ann.glob("*lesion*.nii.gz")) + list(ann.glob("*lesion*.nii")))
    return matches[0] if matches else None


def discover_cases(root: str | Path) -> Dict[str, Path]:
    """Recursively discover AMPLIFAI cases across arbitrary batch nesting."""
    root = Path(root)
    cases: Dict[str, Path] = {}
    patterns = ("annotations/lesion.nii.gz", "annotations/lesion.nii")
    for pat in patterns:
        for lesion in root.rglob(pat):
            case_dir = lesion.parent.parent
            if not (case_dir / "ct").is_dir():
                continue
            cid = case_dir.name
            # Prefer the first occurrence; duplicate IDs are almost certainly a data-layout error.
            if cid in cases and cases[cid].resolve() != case_dir.resolve():
                raise RuntimeError(f"Duplicate case_id {cid}: {cases[cid]} and {case_dir}")
            cases[cid] = case_dir
    return dict(sorted(cases.items()))


def _voxel_sizes(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).astype(float)


def _safe_array(img) -> np.ndarray:
    arr = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _resample_mask_to(mask_img, target) -> np.ndarray:
    if mask_img.shape[:3] == tuple(target[0]) and np.allclose(mask_img.affine, target[1], atol=1e-4):
        return np.asanyarray(mask_img.dataobj) > 0
    out = resample_from_to(mask_img, target, order=0, mode="constant", cval=0.0)
    return np.asanyarray(out.dataobj) > 0


def _roi_target_from_bbox(ref_img, mask_ref: np.ndarray, margin_vox: int = 3):
    idx = np.argwhere(mask_ref)
    if idx.size == 0:
        raise ValueError("empty lesion mask after alignment to arterial phase")
    lo = np.maximum(idx.min(axis=0) - int(margin_vox), 0)
    hi = np.minimum(idx.max(axis=0) + int(margin_vox) + 1, np.array(ref_img.shape[:3]))
    shape = tuple((hi - lo).astype(int).tolist())
    T = np.eye(4, dtype=float)
    T[:3, 3] = lo.astype(float)
    roi_affine = ref_img.affine @ T
    crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return shape, roi_affine, crop


def _load_case_on_art_grid(case_dir: Path, case_id: str, margin_vox: int = 3):
    """Resample all available phases directly into a tight ART-grid ROI."""
    _require_nibabel()
    art_path = find_phase(case_dir, case_id, "ART")
    lesion_path = find_lesion_mask(case_dir)
    if art_path is None:
        raise FileNotFoundError(f"Missing ART phase for {case_id}")
    if lesion_path is None:
        raise FileNotFoundError(f"Missing lesion mask for {case_id}")

    art_img = nib.load(str(art_path))
    lesion_img = nib.load(str(lesion_path))
    full_target = (art_img.shape[:3], art_img.affine)
    lesion_art = _resample_mask_to(lesion_img, full_target)
    if not np.any(lesion_art):
        return None

    roi_shape, roi_affine, crop = _roi_target_from_bbox(art_img, lesion_art, margin_vox=margin_vox)
    mask = lesion_art[crop].astype(bool)
    spacing = _voxel_sizes(roi_affine)

    images: Dict[str, Optional[np.ndarray]] = {}
    present: Dict[str, int] = {}
    for phase in PHASES:
        p = find_phase(case_dir, case_id, phase)
        present[phase] = int(p is not None)
        if p is None:
            images[phase] = None
            continue
        img = nib.load(str(p))
        # Resample only the lesion ROI rather than the full CT volume.
        if img.shape[:3] == roi_shape and np.allclose(img.affine, roi_affine, atol=1e-4):
            arr = _safe_array(img)
        else:
            r = resample_from_to(img, (roi_shape, roi_affine), order=1, mode="constant", cval=np.nan)
            arr = _safe_array(r)
        images[phase] = arr

    return {
        "mask": mask,
        "spacing": spacing,
        "affine": roi_affine,
        "images": images,
        "present": present,
    }


def _finite_values(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = np.asarray(arr[mask], dtype=float)
    return v[np.isfinite(v)]


def _entropy(values: np.ndarray, bins: int = 32) -> float:
    if values.size < 2 or np.nanmax(values) <= np.nanmin(values):
        return 0.0
    lo, hi = np.nanpercentile(values, [1, 99])
    if not np.isfinite(lo + hi) or hi <= lo:
        return 0.0
    hist, _ = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
    p = hist.astype(float)
    p /= p.sum() + EPS
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _intensity_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if values.size == 0:
        for k in ("mean", "std", "median", "mad", "iqr", "p10", "p90", "skew", "kurtosis", "entropy", "energy"):
            out[f"{prefix}{k}"] = np.nan
        return out
    q10, q25, q50, q75, q90 = np.percentile(values, [10, 25, 50, 75, 90])
    mad = np.median(np.abs(values - q50))
    out[f"{prefix}mean"] = float(np.mean(values))
    out[f"{prefix}std"] = float(np.std(values))
    out[f"{prefix}median"] = float(q50)
    out[f"{prefix}mad"] = float(mad)
    out[f"{prefix}iqr"] = float(q75 - q25)
    out[f"{prefix}p10"] = float(q10)
    out[f"{prefix}p90"] = float(q90)
    out[f"{prefix}skew"] = float(skew(values, bias=False)) if values.size >= 8 and np.std(values) > EPS else 0.0
    out[f"{prefix}kurtosis"] = float(kurtosis(values, fisher=True, bias=False)) if values.size >= 8 and np.std(values) > EPS else 0.0
    out[f"{prefix}entropy"] = _entropy(values)
    out[f"{prefix}energy"] = float(np.mean(values * values))
    return out


def _surface_area(mask: np.ndarray, spacing: np.ndarray) -> float:
    """Voxel-face surface area approximation in mm^2."""
    m = mask.astype(np.int8)
    padded = np.pad(m, 1, mode="constant", constant_values=0)
    area = 0.0
    for axis in range(3):
        transitions = np.diff(padded, axis=axis) != 0
        face_area = float(np.prod(np.delete(spacing, axis)))
        area += float(np.count_nonzero(transitions)) * face_area
    return area


def _box_count_fractal_dimension(mask: np.ndarray) -> float:
    """3-D box-counting dimension of the binary lesion mask."""
    shape = np.array(mask.shape)
    max_pow = int(np.floor(np.log2(max(2, int(shape.min())))))
    sizes = [2 ** k for k in range(0, max_pow + 1) if 2 ** k <= max(2, int(shape.min()))]
    counts, eps = [], []
    for s in sizes:
        pad = (s - (shape % s)) % s
        pm = np.pad(mask, [(0, int(p)) for p in pad], mode="constant", constant_values=False)
        new_shape = (pm.shape[0] // s, s, pm.shape[1] // s, s, pm.shape[2] // s, s)
        blocks = pm.reshape(new_shape).any(axis=(1, 3, 5))
        n = int(np.count_nonzero(blocks))
        if n > 0:
            counts.append(n)
            eps.append(1.0 / s)
    if len(counts) < 2:
        return np.nan
    slope = np.polyfit(np.log(eps), np.log(counts), 1)[0]
    return float(slope)


def _geometry_features(mask: np.ndarray, spacing: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    nvox = int(mask.sum())
    voxel_vol = float(np.prod(spacing))
    vol = nvox * voxel_vol
    area = _surface_area(mask, spacing) if nvox else np.nan
    idx = np.argwhere(mask)

    out["geom_voxel_count"] = float(nvox)
    out["geom_volume_mm3"] = float(vol)
    out["geom_surface_area_mm2"] = float(area)
    out["geom_surface_to_volume"] = float(area / (vol + EPS)) if nvox else np.nan
    out["geom_equiv_diameter_mm"] = float((6.0 * vol / math.pi) ** (1.0 / 3.0)) if vol > 0 else np.nan
    out["geom_sphericity"] = float((math.pi ** (1/3) * (6.0 * vol) ** (2/3)) / (area + EPS)) if vol > 0 else np.nan

    if idx.size:
        extents = (idx.max(axis=0) - idx.min(axis=0) + 1) * spacing
        out["geom_bbox_x_mm"], out["geom_bbox_y_mm"], out["geom_bbox_z_mm"] = map(float, extents)
        out["geom_max_bbox_diameter_mm"] = float(extents.max())
        out["geom_bbox_volume_mm3"] = float(np.prod(extents))
        out["geom_extent"] = float(vol / (np.prod(extents) + EPS))

        xyz = idx.astype(float) * spacing[None, :]
        centered = xyz - xyz.mean(axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False) if len(xyz) > 3 else np.zeros((3, 3))
        eig = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))[::-1]
        l1, l2, l3 = (list(eig) + [0.0, 0.0, 0.0])[:3]
        out["spectral_lambda1"] = float(l1)
        out["spectral_lambda2"] = float(l2)
        out["spectral_lambda3"] = float(l3)
        out["spectral_elongation"] = float(math.sqrt((l2 + EPS) / (l1 + EPS)))
        out["spectral_flatness"] = float(math.sqrt((l3 + EPS) / (l1 + EPS)))
        out["spectral_anisotropy"] = float((l1 - l3) / (l1 + EPS))

        r = np.linalg.norm(centered, axis=1)
        if r.size:
            out["geom_radial_mean_mm"] = float(np.mean(r))
            out["geom_radial_std_mm"] = float(np.std(r))
            out["geom_radial_p90_mm"] = float(np.percentile(r, 90))
            out["geom_radial_cv"] = float(np.std(r) / (np.mean(r) + EPS))

        # Convexity / solidity on a deterministic boundary subsample.
        boundary = mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0)
        bidx = np.argwhere(boundary)
        if len(bidx) >= 4:
            if len(bidx) > 5000:
                sel = np.linspace(0, len(bidx) - 1, 5000).astype(int)
                bidx = bidx[sel]
            pts = bidx.astype(float) * spacing[None, :]
            try:
                hull = ConvexHull(pts)
                out["geom_convex_hull_volume_mm3"] = float(hull.volume)
                out["geom_solidity"] = float(vol / (hull.volume + EPS))
            except Exception:
                out["geom_convex_hull_volume_mm3"] = np.nan
                out["geom_solidity"] = np.nan
        else:
            out["geom_convex_hull_volume_mm3"] = np.nan
            out["geom_solidity"] = np.nan

        # Topology-lite: components and internal cavity fraction.
        _, ncc = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
        filled = ndimage.binary_fill_holes(mask)
        cavity_vox = int(np.count_nonzero(filled & ~mask))
        out["topology_components"] = float(ncc)
        out["topology_cavity_fraction"] = float(cavity_vox / (nvox + EPS))
        out["fractal_box_dimension"] = _box_count_fractal_dimension(mask)
    return out


def _shell_masks(mask: np.ndarray, spacing: np.ndarray):
    d = ndimage.distance_transform_edt(mask, sampling=spacing)
    if not np.any(mask):
        return {"rim": mask, "mid": mask, "core": mask}, d
    md = float(np.nanmax(d[mask]))
    if md <= EPS:
        return {"rim": mask, "mid": np.zeros_like(mask), "core": np.zeros_like(mask)}, d
    rho = d / md
    shells = {
        "rim": mask & (rho <= 1/3),
        "mid": mask & (rho > 1/3) & (rho <= 2/3),
        "core": mask & (rho > 2/3),
    }
    return shells, d


def _neighbor_spatial_features(arr: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    vals = _finite_values(arr, mask)
    var = float(np.var(vals)) if vals.size else np.nan
    abs_diffs, sq_diffs = [], []
    for axis in range(3):
        s1 = [slice(None)] * 3
        s2 = [slice(None)] * 3
        s1[axis] = slice(1, None)
        s2[axis] = slice(None, -1)
        m = mask[tuple(s1)] & mask[tuple(s2)]
        a = arr[tuple(s1)][m]
        b = arr[tuple(s2)][m]
        f = np.isfinite(a) & np.isfinite(b)
        if np.any(f):
            d = a[f].astype(float) - b[f].astype(float)
            abs_diffs.append(np.abs(d))
            sq_diffs.append(d * d)
    if abs_diffs:
        ad = np.concatenate(abs_diffs)
        sd = np.concatenate(sq_diffs)
        out[f"{prefix}graph_total_variation"] = float(np.mean(ad))
        out[f"{prefix}dirichlet_energy"] = float(np.mean(sd))
        out[f"{prefix}dirichlet_energy_norm"] = float(np.mean(sd) / (var + EPS)) if np.isfinite(var) else np.nan
    else:
        out[f"{prefix}graph_total_variation"] = np.nan
        out[f"{prefix}dirichlet_energy"] = np.nan
        out[f"{prefix}dirichlet_energy_norm"] = np.nan

    # Semivariogram at voxel lags 1,2,4 aggregated over axes.
    for lag in (1, 2, 4):
        gammas = []
        for axis in range(3):
            if arr.shape[axis] <= lag:
                continue
            s1 = [slice(None)] * 3
            s2 = [slice(None)] * 3
            s1[axis] = slice(lag, None)
            s2[axis] = slice(None, -lag)
            m = mask[tuple(s1)] & mask[tuple(s2)]
            a = arr[tuple(s1)][m]
            b = arr[tuple(s2)][m]
            f = np.isfinite(a) & np.isfinite(b)
            if np.any(f):
                gammas.append(0.5 * float(np.mean((a[f].astype(float) - b[f].astype(float)) ** 2)))
        g = float(np.mean(gammas)) if gammas else np.nan
        out[f"{prefix}semivariogram_lag{lag}"] = g
        out[f"{prefix}semivariogram_lag{lag}_norm"] = float(g / (var + EPS)) if np.isfinite(g) and np.isfinite(var) else np.nan
    return out


def _center_of_mass_offset(field: np.ndarray, mask: np.ndarray, spacing: np.ndarray) -> float:
    vals = _finite_values(field, mask)
    if vals.size < 8:
        return np.nan
    idx = np.argwhere(mask & np.isfinite(field))
    v = field[mask & np.isfinite(field)].astype(float)
    if len(v) != len(idx):
        return np.nan
    threshold = np.percentile(v, 75)
    high = v >= threshold
    if not np.any(high):
        return 0.0
    geom = idx.astype(float) * spacing[None, :]
    c0 = geom.mean(axis=0)
    c1 = geom[high].mean(axis=0)
    eq_radius = ((3.0 * len(idx) * np.prod(spacing)) / (4.0 * math.pi) + EPS) ** (1/3)
    return float(np.linalg.norm(c1 - c0) / (eq_radius + EPS))


def _phase_features(phase: str, arr: Optional[np.ndarray], mask: np.ndarray, spacing: np.ndarray,
                    shells: Dict[str, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {f"present_{phase}": float(arr is not None)}
    if arr is None:
        return out
    vals = _finite_values(arr, mask)
    out.update(_intensity_stats(vals, prefix=f"phase_{phase}_"))
    for sname, smask in shells.items():
        sv = _finite_values(arr, smask)
        out.update(_intensity_stats(sv, prefix=f"shell_{phase}_{sname}_"))
        out[f"shell_{phase}_{sname}_fraction"] = float(np.count_nonzero(smask) / (np.count_nonzero(mask) + EPS))
    # Rim-core contrast is a simple proxy for peripheral enhancement/capsule behavior.
    rim = _finite_values(arr, shells["rim"])
    core = _finite_values(arr, shells["core"])
    if rim.size and core.size:
        out[f"shell_{phase}_rim_minus_core"] = float(np.mean(rim) - np.mean(core))
        out[f"shell_{phase}_rim_to_core_ratio"] = float(np.mean(rim) / (abs(np.mean(core)) + EPS))
    else:
        out[f"shell_{phase}_rim_minus_core"] = np.nan
        out[f"shell_{phase}_rim_to_core_ratio"] = np.nan
    out.update(_neighbor_spatial_features(arr, mask, prefix=f"spatial_{phase}_"))
    out[f"spatial_{phase}_hotspot_com_offset"] = _center_of_mass_offset(arr, mask, spacing)
    return out


def _difference_features(name: str, a: Optional[np.ndarray], b: Optional[np.ndarray], mask: np.ndarray,
                         spacing: np.ndarray, shells: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Features of field a-b (e.g., ART-DRY enhancement or VEN-ART washout)."""
    out: Dict[str, float] = {}
    prefix = f"delta_{name}_"
    if a is None or b is None:
        return out
    d = a.astype(np.float32) - b.astype(np.float32)
    vals = _finite_values(d, mask)
    out.update(_intensity_stats(vals, prefix=prefix))
    if vals.size:
        out[f"{prefix}positive_fraction"] = float(np.mean(vals > 0))
        out[f"{prefix}negative_fraction"] = float(np.mean(vals < 0))
    for sname, smask in shells.items():
        sv = _finite_values(d, smask)
        if sv.size:
            out[f"{prefix}{sname}_mean"] = float(np.mean(sv))
            out[f"{prefix}{sname}_std"] = float(np.std(sv))
            out[f"{prefix}{sname}_positive_fraction"] = float(np.mean(sv > 0))
        else:
            out[f"{prefix}{sname}_mean"] = np.nan
            out[f"{prefix}{sname}_std"] = np.nan
            out[f"{prefix}{sname}_positive_fraction"] = np.nan
    if _finite_values(d, shells["rim"]).size and _finite_values(d, shells["core"]).size:
        out[f"{prefix}rim_minus_core"] = float(np.mean(_finite_values(d, shells["rim"])) - np.mean(_finite_values(d, shells["core"])))
    else:
        out[f"{prefix}rim_minus_core"] = np.nan
    out[f"{prefix}hotspot_com_offset"] = _center_of_mass_offset(d, mask, spacing)
    out.update(_neighbor_spatial_features(d, mask, prefix=f"{prefix}spatial_"))
    return out


def _trajectory_features(images: Dict[str, Optional[np.ndarray]], mask: np.ndarray,
                         shells: Dict[str, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    order = list(PHASES)
    regions = {"whole": mask, **shells}
    for rname, rmask in regions.items():
        means = []
        phases = []
        for i, p in enumerate(order):
            arr = images.get(p)
            if arr is None:
                continue
            v = _finite_values(arr, rmask)
            if v.size:
                phases.append(i)
                means.append(float(np.mean(v)))
        if len(means) >= 2:
            x = np.asarray(phases, dtype=float)
            y = np.asarray(means, dtype=float)
            out[f"traj_{rname}_dynamic_range"] = float(np.max(y) - np.min(y))
            out[f"traj_{rname}_path_length"] = float(np.sum(np.abs(np.diff(y))))
            out[f"traj_{rname}_phase_auc"] = float(np.trapz(y, x=x) / (x[-1] - x[0] + EPS))
            peak_idx = int(np.argmax(y))
            out[f"traj_{rname}_peak_phase_index"] = float(x[peak_idx])
            out[f"traj_{rname}_end_minus_start"] = float(y[-1] - y[0])
        if len(means) >= 3:
            y = np.asarray(means, dtype=float)
            out[f"traj_{rname}_mean_abs_curvature"] = float(np.mean(np.abs(np.diff(y, n=2))))
    return out


def extract_case_features(case_dir: str | Path, case_id: Optional[str] = None,
                          margin_vox: int = 3) -> Dict[str, float]:
    """Extract inference-safe features from one AMPLIFAI case.

    Returns a flat dictionary. Feature names are grouped by prefixes:
    geom_, spectral_, topology_, fractal_, phase_, shell_, spatial_, delta_, traj_.
    """
    case_dir = Path(case_dir)
    case_id = case_id or case_dir.name
    base: Dict[str, float] = {"case_id": str(case_id)}
    loaded = _load_case_on_art_grid(case_dir, case_id, margin_vox=margin_vox)
    if loaded is None:
        base["geom_voxel_count"] = 0.0
        base["has_lesion"] = 0.0
        for p in PHASES:
            base[f"present_{p}"] = float(find_phase(case_dir, case_id, p) is not None)
        return base

    mask = loaded["mask"]
    spacing = loaded["spacing"]
    images = loaded["images"]
    base["has_lesion"] = 1.0
    base.update(_geometry_features(mask, spacing))
    shells, _ = _shell_masks(mask, spacing)

    for p in PHASES:
        base.update(_phase_features(p, images.get(p), mask, spacing, shells))

    # Clinically meaningful enhancement trajectories. Missing phases simply yield missing features.
    pairs = [
        ("ART_MINUS_DRY", "ART", "DRY"),
        ("VEN_MINUS_ART", "VEN", "ART"),
        ("DEL_MINUS_ART", "DEL", "ART"),
        ("DEL_MINUS_VEN", "DEL", "VEN"),
    ]
    for name, pa, pb in pairs:
        base.update(_difference_features(name, images.get(pa), images.get(pb), mask, spacing, shells))

    base.update(_trajectory_features(images, mask, shells))
    return base


def _annotation_candidates(annotation_key: str):
    mapping = {
        "aphe": ["aphe.nii.gz", "aphe.nii"],
        "washout_venous": ["venous_washout.nii.gz", "venous_washout.nii", "washout_venous.nii.gz", "washout_venous.nii"],
        "washout_delayed": ["delayed_washout.nii.gz", "delayed_washout.nii", "washout_delayed.nii.gz", "washout_delayed.nii"],
        "capsule_venous": ["venous_capsule.nii.gz", "venous_capsule.nii", "capsule_venous.nii.gz", "capsule_venous.nii"],
        "capsule_delayed": ["delayed_capsule.nii.gz", "delayed_capsule.nii", "capsule_delayed.nii.gz", "capsule_delayed.nii"],
    }
    return mapping.get(annotation_key, [])


def extract_annotation_burdens(case_dir: str | Path, case_id: Optional[str] = None) -> Dict[str, float]:
    """Training-only soft targets: fraction of lesion occupied by annotated major-feature masks.

    These values MUST NOT be used directly as test-time predictors because the private test set
    provides only the lesion segmentation. They are intended as optional distillation targets.
    """
    _require_nibabel()
    case_dir = Path(case_dir)
    case_id = case_id or case_dir.name
    art_path = find_phase(case_dir, case_id, "ART")
    lesion_path = find_lesion_mask(case_dir)
    out = {"case_id": str(case_id)}
    if art_path is None or lesion_path is None:
        return out
    art = nib.load(str(art_path))
    target = (art.shape[:3], art.affine)
    lesion = _resample_mask_to(nib.load(str(lesion_path)), target)
    denom = int(np.count_nonzero(lesion))
    if denom == 0:
        return out
    ann_dir = case_dir / "annotations"
    for key in ("aphe", "washout_venous", "washout_delayed", "capsule_venous", "capsule_delayed"):
        p = _first_existing([ann_dir / x for x in _annotation_candidates(key)])
        if p is None:
            out[f"burden_{key}"] = np.nan
            continue
        m = _resample_mask_to(nib.load(str(p)), target)
        out[f"burden_{key}"] = float(np.count_nonzero(m & lesion) / (denom + EPS))
    return out
