# Released experimental outputs

This directory contains the **public train/CV and 59-case public-validation outputs** used in the manuscript. It does not contain private challenge-test predictions or metrics.

- `tables/` — primary model comparison and baseline tables
- `validation/` — PHORA/multi-view predictions, concept recovery, bootstrap, uncertainty and burden analyses
- `cv/` — architecture/feature ablations and repeated-seed CV
- `domain/` — train→validation shift and exploratory source robustness
- `features/` — cached inference-safe public feature tables and annotation-burden targets
- `latex/` — original LaTeX table exports

The curated model-comparison table fills the multi-view stack's Macro-F1 directly from `multiview_predictions.csv` (0.405128...) and uses the final paper terminology “handcrafted radiomics” rather than the historical internal “PyRadiomics” label.
