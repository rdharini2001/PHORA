# PHORA manuscript ↔ repository ↔ two provided CT cases

**Checked against the MICCAI 2026 AMPLIFAI paper and the connected `rdharini2001/PHORA` repository (main branch), 22 October 2026.** This is an audit of available evidence; it does not claim to re-train PHORA or to run its frozen inference model on the two uploaded cases.

## Public-validation results: agreement at published precision

From Table 1 of the 11-page official paper and `results/tables/model_comparison_validation.csv`:

| Model | Published S | Repo S | Published QWK | Repo QWK | Published SCR | Repo SCR | Audit |
|---|---:|---:|---:|---:|---:|---:|---|
| Majority class | .050 | .050 | .000 | .000 | .333 | .333 | aligns |
| RBF-SVM | .281 | .281 | .232 | .232 | .555 | .555 | aligns |
| HistGradientBoosting | .200 | .200 | .113 | .113 | .692 | .692 | aligns |
| Elastic-net logistic | .690 | .690 | .702 | .702 | .620 | .620 | aligns |
| Flat XGBoost | .687 | .687 | .686 | .686 | .692 | .692 | aligns |
| Random Forest | .705 | .705 | .709 | .709 | .683 | .683 | aligns |
| Extra Trees | .763 | .763 | .785 | .785 | .635 | .635 | aligns |
| Simple hierarchical XGB | .692 | .692 | .703 | .703 | .629 | .629 | aligns |
| Cross-fitted multi-view stack | .734 | .734 | .741 | .741 | .694 | .694 | aligns |
| **PHORA** | **.718** | **.718** | **.738** | **.738** | **.605** | **.605** | **aligns** |
| Oracle concepts (non-deployable) | .935 | .935 | .947 | .947 | .870 | .870 | aligns |

The official objective is `S = 0.85 * adjusted QWK + 0.15 * special-category recognition`. PHORA's unrounded repository values (0.737732657614929 and 0.6052631578947368) yield composite 0.7178622326569002 and round to the manuscript's 0.718. The site names the oracle clearly as *non-deployable*.

## Architecture and concept agreement

- **Architecture CV**: The seven configurations and their published 3-decimal Composite/QWK/SCR numbers correspond to `results/cv/architecture_ablation.csv`; note the full-model row is labeled `PHORA+ v2 full` in code and `PHORA full` in the article.
- **Concept recovery**: `results/validation/concept_recovery.csv` agrees at 3 decimals with all six AUROC/AUPRC values reported for APHE, washout and capsule concepts. The AUROC values 0.915 (non-rim APHE) and 0.905 (delayed washout) agree with the abstract.
- **Per-class recall**: Seven entries in `results/validation/per_class_recall.csv` agree with the integer fractions printed in Results §5.2.
- **Uncertainty quartiles**: Four accuracies match 0.800, 0.533, 0.643 and 0.467 after 3-decimal rounding; the quartile sizes (15/15/14/15) sum to 59.
- **Bootstrap**: Mean/interval values in `results/validation/bootstrap_confidence_intervals.csv` and paired comparison reproduce the published rounded values.
- **Feature composition**: `results/model/model_summary.json` records 140 screened imaging features (70 radiomics, 60 spatial physiology, 10 context/OT), 11 concept/burden variables and 24 interactions = 175 dimensions, consistent with the code documentation.

## Discrepancy requiring repository clarification — published router configuration

The paper §3.5 states the **fixed public-validation router** uses:

`beta=0.25, alpha=0.50, cutpoints=(1.373,3.009,3.693,4.183), thresholds LR-M=0.327 and LR-TIV=0.699, special margin=0.080, APHE consistency threshold=0.291, rim-APHE reweighting=0.971`.

But `results/model/model_summary.json` contains:

`beta=0.50, alpha=0.25, cutpoints=(1.63453,2.84841,3.42849,3.82331), LR-M=0.50712, LR-TIV=0.17818, margin=-0.01269, APHE threshold=0.22223, rim reweighting=1.86504`.

**These are not the same parameter values.** The stored model summary also contains an `inner_training_metrics` score of 0.665, not the publication's validation score 0.718. A legitimate possibility is that the JSON describes an earlier/intermediate fit rather than the frozen published validation router; the record is not labeled clearly enough to establish this. Do not silently substitute the JSON values for the publication's fixed settings. Investigate which serialized model/configuration produced the 59-case PHORA predictions and label/version the JSON accordingly.

This discrepancy does **not** change the demonstrated row-by-row agreement between the published validation metrics and the repository's saved results CSV.

## The two newly supplied NIfTI cases

- `CASE00001`: CT DRY / ART / VEN / DEL, lesion mask, APHE mask, venous washout mask and delayed washout mask. CT shape 512×512×91, mask voxels 94,878.
- `CASE00002`: CT DRY / ART / VEN / DEL, lesion mask only. CT shape 512×512×194, mask voxels 9,326.
- For each case, image volumes and annotations share dimensions; the website uses a fixed slice grid for overlays. The NIfTI image arrays have HU-like values and support descriptive enhancement calculation.
- The cases are **not present among the 59 rows** in `results/validation/phora_predictions.csv` (confirm when integrated with the repository); no author-verified PHORA predictions for these two cases were provided. The website shows phase evidence, not fabricated case-level labels or fabricated attention.
- **Visualization method (revised website)**: signed change maps use `ART − DRY`, `VEN − ART`, and `DEL − VEN` within the 3-D lesion mask. Amber indicates increasing attenuation, cyan indicates decreasing attenuation, and changes smaller than ±10 HU are suppressed for display. The original unsigned images remain archived but are not the active viewer layer. Source NIfTI data and phase-matched masks remain unchanged.

## Publication / website guidance

Use the canonical [MICCAI published PDF](https://papers.miccai.org/miccai-2026-sat/paper/AMPLIFAI_007.pdf) as the manuscript link. Keep exploratory NIfTI examples, validation model scores, and router parameter metadata distinct. The website is not a clinical decision-support system.


## Updated phase-to-decision viewer

The revised website adds a five-stage theatre sourced from saved PHORA public-validation predictions. Its examples are `CASE00015`, `CASE00025`, `CASE00035`, `CASE00084`, and `CASE00099`, each verified against the source CSV’s routed label, concept estimates, burden estimates, special-category gate scores, ordinal experts and seven category outputs. Each original record reflects full multiphase inference. The theatre highlights named phase-linked clinical concepts rather than asserting that classifications were recalculated after individual acquisitions. The two uploaded CT cases remain separate, unscored imaging examples.

The signed voxelwise maps and phase-trajectory HU statistics are generated from the provided NIfTI volumes; see `docs/CLINICAL_VISUALIZATION_METHODS.md` and `scripts/add_signed_maps.py`.
