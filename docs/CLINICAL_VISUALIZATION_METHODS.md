# PHORA | Clinical visualization methods

This website accompanies *PHORA: Physiology-Guided Hierarchical Ordinal Radiomics for LI-RADS Classification in Multiphase CT*, MICCAI 2026 AMPLIFAI Satellite Events. The CT atlas explores two supplied annotated cases and the prediction theatre displays five distinct frozen public-validation inference records.

## Interpreting the CT atlas

CT is displayed with a fixed center 85 HU and width 390 HU. Each case has 21 aligned axial slices selected to traverse its segmented lesion, across noncontrast (DRY), arterial (ART), portal venous (VEN) and delayed (DEL) phases. The website colors are editorial; CT attenuation is measured in Hounsfield units.

The signed heatmaps use *adjacent phase comparisons* for a clearer physiologic story:

- Arterial change: ART minus DRY.
- Portal venous change: VEN minus ART.
- Delayed change: DEL minus VEN.

At each selected axial slice, the signed change is computed on the native voxel grid, spatially restricted to the supplied lesion annotation, then resampled for display. Increased attenuation is amber; decreased attenuation is cyan. Changes smaller than 10 HU are visually suppressed, and stronger changes appear more opaque. The numerical percentages displayed under the heatmap report the fraction of **all annotated lesion voxels** increasing by at least 10 HU or decreasing by at least 10 HU between the two phases. The summaries are independent of the axial slice being viewed.

These maps reveal within-lesion temporal heterogeneity. They do not represent attention, saliency, gradient attribution, or the influence of individual voxels on PHORA's final classification. Neither increasing enhancement nor decreasing HU alone establishes a LI-RADS major feature: APHE and washout are interpreted according to their clinical definitions and relevant comparison with adjacent liver.

The reference annotation overlays were supplied with the case volumes. For Case 1, the website presents lesion annotation in every phase, APHE mask in the arterial phase, venous-washout mask in the portal venous phase, and delayed-washout mask in the delayed phase. Case 2 includes a lesion mask only. These are **reference annotations**, not PHORA segmentation predictions.

## Interpreting the PHORA decision theatre

Five genuine validation prediction records (`CASE00015`, `CASE00025`, `CASE00035`, `CASE00084`, `CASE00099`) were transcribed from the public repository's `results/validation/phora_predictions.csv`. The theatre displays their recorded final routed label, reference label, seven class scores, ordinal-regression and all-threshold outputs, direct and factorized special-category gate scores, six concept predictions, and relevant spatial burden estimates.

The five stages organize a **single complete multiphase inference** by the clinical concepts they describe:

1. Noncontrast: morphology and baseline attenuation context.
2. Arterial: non-rim APHE, rim APHE and APHE burden.
3. Portal venous: venous washout, venous capsule and venous washout burden.
4. Delayed: delayed washout, delayed capsule and delayed washout burden.
5. Final decision: direct/factorized special-category gates, ordinal experts, and recorded final category.

The animation **does not** calculate successive prefix-stage scores and does not imply PHORA is re-run after each phase; this would be a different, unvalidated inference task. All displayed probabilities are the model's original complete-input outputs. The two illustrated CT cases do not occur in the 59-case public-validation prediction table and are not assigned generated PHORA labels by this viewer.

## Image and label scope

Case-derived maps are descriptive of the supplied CT volumes, for the purposes of research-method communication. This website is not medical device software and is not a substitute for radiologist assessment. The 59-case model performance is taken from the public validation CSV and original manuscript, not estimated from the two displayed scans.

Sources: [Published PHORA paper](https://papers.miccai.org/miccai-2026-sat/paper/AMPLIFAI_007.pdf) · [PHORA repository](https://github.com/rdharini2001/PHORA) · [Saved public validation predictions](https://github.com/rdharini2001/PHORA/blob/main/results/validation/phora_predictions.csv).
