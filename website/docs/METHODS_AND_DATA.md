# PHORA — Research website data and visualization methods

This interactive research companion accompanies:

> Raghavan, D., Madabhushi, A., Singh, A., Lebowitz, M., Modanwal, G., and Madabhushi, A. **PHORA: Physiology-Guided Hierarchical Ordinal Radiomics for LI-RADS Classification in Multiphase CT.** MICCAI 2026 Satellite Events, AMPLIFAI. [Official paper](https://papers.miccai.org/miccai-2026-sat/paper/AMPLIFAI_007.pdf) · [Code](https://github.com/rdharini2001/PHORA)

## What the case viewer actually displays

The two supplied case archives contain NIfTI CT images on four contrast phases (DRY, ART, VEN and DEL). Case 1 also contains radiologist-provided lesion, APHE, venous-washout and delayed-washout masks; Case 2 contains a lesion mask only. The viewer renders 21 selected axial slices per case. These slices span the actual segmented lesion, with the slice of greatest segmented area included. Image exports are grayscale WebP; masks are aligned RGBA PNGs. The original volumes are **not** uploaded to GitHub Pages; the supplied NIfTI archives remain the authoritative full-resolution imaging source.

Images use a fixed abdominal CT display window (center 85 HU, width 390 HU). The interface exposes real lesion masks and the available feature annotations. Note that the same image grid in the supplied archives is used for visual overlays; the website is not a separate deformable image-registration experiment.

### Phase-specific intensity evidence

The plotted phase trajectory is the mean of all voxels in the supplied 3-D lesion annotation for each source CT phase. PHORA's published method uses richer descriptors than this single mean: distributional features, local spatial differences, habitat statistics, temporal changes, and radiomics. The chart is an accessible direct illustration of one physiologic component, **not the complete PHORA feature vector**.

The amber evidence map is defined at the displayed axial position by the **absolute voxelwise HU difference relative to the DRY phase, masked to the supplied lesion**:

\[
H_p(v) = \mathbf{1}_{v \in \Omega}\; \min\left(1,\;\frac{|I_p(v)-I_{DRY}(v)|}{140}\right)
\]

The normalized value controls pixel opacity only. It is an image-derived *phase-difference visualization*, **not learned model attention**, gradient attribution, SHAP, a saliency map, a radiologist annotation, or a model explanation. Do not interpret the heatmap as the spatial influence of each voxel on the predicted LI-RADS class.

### What the two cases cannot show

PHORA makes **a single case-level inference across available phases** rather than issuing a separate clinical category at each phase. Thus, cycling phases illustrates changing imaging evidence, not a changing PHORA category prediction. The two supplied cases are not among the published 59 validation prediction rows provided in `results/validation/phora_predictions.csv`, and the website does not invent their output labels or probabilities. To add genuine per-case predictions, run the repository's feature extraction and frozen model on the exact supplied volumes and record the resulting outputs with their model version. Ground-truth concept masks are never passed directly to the inference model.

The web assets are research illustrations intended for studying image characteristics. They are not clinical diagnostic images or diagnostic software. The relative HU graphics must not be used for patient care.

## Rebuilding the viewer

1. Place the original uploaded ZIPs at `CASE00001.zip` and `CASE00002.zip` in `/mnt/data` (or adapt the paths in `scripts/prepare_ct.py`).
2. Ensure Python 3.10+, NumPy, SciPy and Pillow are installed. A lightweight NIfTI-1 parser is built into the script; nibabel is not required.
3. Run `python scripts/prepare_ct.py` from the site package root.
4. Open `docs/index.html` or deploy the `docs/` directory through GitHub Pages.

The workflow exports a manifest to `docs/assets/case_data.json` and `case_data.js`. The JS mirror permits local file viewing without a fetch/CORS dependency. The complete original CT volumes are intentionally excluded from this website package.

## Reproduced descriptive values for the two supplied cases

| Case | Voxels in lesion mask | DRY mean HU | ART mean HU | VEN mean HU | DEL mean HU | Supplied masks |
|---|---:|---:|---:|---:|---:|---|
| CASE00001 | 94,878 | 46.8 | 79.8 | 94.4 | 71.0 | lesion, APHE, venous washout, delayed washout |
| CASE00002 | 9,326 | −35.5 | 11.1 | 26.0 | 2.7 | lesion |

Values are calculated directly over the source lesion mask volume and are rounded to one decimal. They are descriptive imaging measurements, **not PHORA classification outputs**.

## Validation and source-of-truth

All model performance shown on the website comes from the publication and `results/tables/model_comparison_validation.csv` and `results/validation/concept_recovery.csv` in the connected source repository. The code repository preserves the models, original validation records and official challenge score definition. See `REPRODUCIBILITY_AUDIT.md` in the site package for the complete comparison and an identified router-metadata inconsistency.

- [PHORA open-access manuscript](https://papers.miccai.org/miccai-2026-sat/paper/AMPLIFAI_007.pdf)
- [PHORA source repository](https://github.com/rdharini2001/PHORA)
- [MICCAI 2026 PHORA proceedings record](https://papers.miccai.org/miccai-2026-sat/AMPLIFAI_007.html)
