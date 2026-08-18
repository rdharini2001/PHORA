# PHORA method

PHORA treats LI-RADS as a structured decision problem rather than seven unrelated classes.

## 1. Imaging representation

The supplied lesion mask and available DRY/ART/VEN/DEL CT phases are converted into three complementary feature views:

1. **Handcrafted radiomics** — 3-D shape, first-order, GLCM, GLRLM, GLSZM, GLDM and NGTDM features after phase-specific alignment, a 5-mm padded crop and 1-mm isotropic resampling. Intensities use a fixed 25-HU discretization.
2. **Spatial physiology** — whole-lesion morphology, rim/middle/core habitats, graph total variation and Dirichlet energy, semivariograms, hotspot displacement/topology, voxelwise inter-phase changes and enhancement-trajectory descriptors.
3. **Local context and distribution shift** — lesion-minus-perilesional reference kinetics, Moran's I, Geary's C, hotspot topology, a vascular-contact surrogate, Wasserstein distance and Jensen-Shannon divergence.

The complete candidate pool contains about **900 inference-safe imaging variables**. Nested screening is performed inside each training fold, removing high-missingness/near-constant variables and ranking candidates for the special-category gate, ordinal severity and clinical concepts. Correlated variables are pruned, leaving at most **140 imaging features**. In the frozen Codabench model these 140 features comprise 70 handcrafted-radiomics, 60 spatial-physiology and 10 context/OT variables.

## 2. Cross-fitted clinical supervision

Training-only radiologist labels supervise six concept heads: non-rim APHE, rim APHE, venous washout, delayed washout, venous capsule and delayed capsule. Five voxel-mask burdens are also distilled. Training cases receive only out-of-fold auxiliary predictions; validation/test cases receive predictions from heads fit on the training partition. Ground-truth concept labels/masks are therefore never direct inference inputs.

The frozen challenge model appends 11 latent concept/burden variables and 24 soft clinical interaction terms to the 140 selected imaging variables, yielding a **175-dimensional unified tabular representation**. It is a concatenated interpretable feature space, not a learned neural embedding.

## 3. Hierarchical ordinal prediction

The unified representation feeds four downstream experts:

- a direct three-way gate over ordinal / LR-M / LR-TIV;
- a factorized special gate: special-vs-ordinal followed by LR-TIV-vs-LR-M;
- an ordinal XGBoost regression expert for LR-1--LR-5;
- a shared all-threshold ordinal expert estimating `P(Y > k)` for four ordered thresholds.

A metric-aware router blends direct/factorized special evidence and the two ordinal estimates. Ordered cut-points map the ordinal score to LR-1--LR-5. Predicted rim APHE softly supports LR-M; LR-5 is constrained by lesion size and predicted non-rim APHE. Router parameters are tuned only on inner out-of-fold predictions using the AMPLIFAI composite objective.
