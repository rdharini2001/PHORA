# PHORA — Research website data and visualization methods

This interactive research companion accompanies:

> Raghavan, D., Madabhushi, A., Singh, A., Lebowitz, M., Modanwal, G., and Madabhushi, A. **PHORA: Physiology-Guided Hierarchical Ordinal Radiomics for LI-RADS Classification in Multiphase CT.** MICCAI 2026 Satellite Events, AMPLIFAI. [Official paper](https://papers.miccai.org/miccai-2026-sat/paper/AMPLIFAI_007.pdf) · [Code](https://github.com/rdharini2001/PHORA)

## What the case viewer actually displays

The two supplied case archives contain NIfTI CT images on four contrast phases (DRY, ART, VEN and DEL). Case 1 also contains radiologist-provided lesion, APHE, venous-washout and delayed-washout masks; Case 2 contains a lesion mask only. The viewer renders 21 selected axial slices per case. These slices span the actual segmented lesion, with the slice of greatest segmented area included. Image exports are grayscale WebP; masks are aligned RGBA PNGs. The original volumes are **not** uploaded to GitHub Pages; the supplied NIfTI archives remain the authoritative full-resolution imaging source.

Images use a fixed abdominal CT display window (center 85 HU, width 390 HU). The interface exposes real lesion masks and the available feature annotations. Note that the same image grid in the supplied archives is used for visual overlays; the website is not a separate deformable image-registration experiment.

### Phase-specific intensity evidence and clinical reading

The lesion-enhancement trajectory is calculated as the mean CT intensity across the supplied 3-D lesion annotation for each source CT phase. PHORA uses richer image-derived descriptors than this mean curve, including radiomics, spatial habitats, and phase kinetics.

The active website viewer presents **signed HU changes between successive acquisitions** within the lesion mask: `ART − DRY`, `VEN − ART`, and `DEL − VEN`. Warm-colored pixels represent increasing attenuation, cyan pixels decreasing attenuation, and variations smaller than ±10 HU are suppressed visually. Saturation of the color opacity increases with absolute change up to a 100 HU display span; this does not alter the source intensity data. The proportion of segmented lesion voxels exceeding +10 HU or falling below −10 HU is reported alongside each phase.

The older `*_heat_*.png` assets are retained as magnitude maps for research comparison; the public viewer uses the separately exported `*_signed_*.png` assets. Maps express imaging contrast, not voxel attribution to PHORA's classification function. Clinical APHE and washout interpretation also depends on relative behavior against liver and on standard acquisition criteria.

### Clinical-feature overlays

Case 1 contains lesion, arterial hyperenhancement, venous washout and delayed washout reference annotations; the phase controls select the matching concept mask. Case 2 contains a lesion mask only. The viewer does not infer concept masks for Case 2.

### PHORA case-level predictions

The decision sequence presents five archived PHORA public-validation records. Its complete-input outputs include six clinical concept scores, five burden scores, the direct and factorized special-category gates, the ordinal experts, and the seven class outputs. Phase selectors organize those concept estimates for radiologist interpretation. The published model consumes the full four-phase image representation; the sequence is not an experiment in sequential acquisition or repeated classification after each phase.

CASE00001 and CASE00002 do not occur in the repository's public-validation prediction table. The site therefore illustrates their imaging evidence without assigning PHORA predictions. The separate frozen-model outputs are traced to the named validation records. See `CLINICAL_VISUALIZATION_METHODS.md` for details.

These materials support research interpretation and method communication, not clinical decision-making.

## Rebuilding the viewer

1. Place the original uploaded ZIPs at `CASE00001.zip` and `CASE00002.zip` in `/mnt/data` (or adapt the paths in `scripts/prepare_ct.py`).
2. Ensure Python 3.10+, NumPy, SciPy and Pillow are installed. A lightweight NIfTI-1 parser is built into the script; nibabel is not required.
3. Run `python scripts/prepare_ct.py`, followed by `python scripts/add_signed_maps.py`, from the site package root.
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
