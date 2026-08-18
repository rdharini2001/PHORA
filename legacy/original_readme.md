# PHORA+ v2 — updated AMPLIFAI new-data notebook

## Files
- `PHORA_LI_NEW_DATA_PAPER.ipynb` — main end-to-end notebook
- `PHORA_LI_NEW_DATA_PAPER_cells.py` — same cells as a `# %%` Python file
- `phora_plus_v2.py` — PHORA+ v2 modeling, routing, context/OT features, baselines, multi-view stack
- `phora_features.py` — spatial/physiology feature extraction and annotation-burden extraction
- `extract_amplifai_pyradiomics.py` — optional conventional PyRadiomics extraction

## Expected data layout

The notebook defaults to:

`E:\HCC_MICCAI26\new_data`

containing:
- `batch_001.zip`
- `batch_002.zip`
- `batch_003.zip`
- `batch_004.zip`
- `train_metadata.csv`
- `val_metadata.csv`

ZIPs may contain arbitrary repeated batch nesting. Case discovery is recursive.

## First run

1. Put all code files beside the notebook.
2. Open `PHORA_LI_NEW_DATA_PAPER.ipynb`.
3. Verify `NEW_DATA_ROOT`.
4. Start with `FAST_MODE=True`.
5. Run top-to-bottom.
6. Once the pipeline is stable, set:
   - `FAST_MODE=False`
   - `RUN_ARCH_ABLATIONS=True`
   - `RUN_FEATURE_ABLATIONS=True`
   - `RUN_REPEATABILITY=True`
7. Restart the kernel and run top-to-bottom for manuscript numbers.

Feature extraction is cached in `phora_paper_outputs_newdata`.

## Evaluation protocol

- Architecture/model selection and ablations: training split only.
- Primary holdout result: official validation split.
- Final train+validation refit: only after the method is frozen.

The notebook excludes any `No lesion` rows from the seven-class LI-RADS classifier because they are not legal challenge output labels.

## Main new experiments

- strong classical ML baselines
- flat XGBoost
- simple hierarchy
- cross-fitted multi-view stacking
- PHORA+ v2
- oracle concept upper bound (analysis only)
- architecture ablations
- feature-family ablations
- concept AUROC/AUPRC
- spatial annotation-burden distillation
- per-class recall/confusion matrix
- bootstrap CIs and paired bootstrap
- uncertainty via ordinal-expert disagreement
- train→validation MMD/domain classifier
- optional leave-one-batch-out robustness
- repeated seed sensitivity

## Main theoretical additions

- local perilesional reference kinetics
- optimal transport between phase intensity distributions
- spatial autocorrelation and hotspot topology
- cross-fitted concept bottleneck
- cross-fitted voxel-mask burden distillation
- factorized special-category gate
- shared all-threshold ordinal model
- metric-aware routing
- probabilistic LI-RADS clinical constraints
