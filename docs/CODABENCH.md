# AMPLIFAI Codabench submission

`challenge/PHORA_AMPLIFAI_Codabench.zip` is the self-contained inference submission corresponding to the paper method.

The ZIP contains `run.py` and `metadata` at the archive root and writes `/app/output/predictions.csv`. The fitted XGBoost ensembles were converted to a portable pure-Python/NumPy tree representation, so the challenge runtime does not need XGBoost, PyRadiomics or SimpleITK. No network access or runtime installation is required.

Rebuild the archive after editing the source with:

```bash
python scripts/build_codabench_submission.py
```

Private-test metrics and leaderboard rank are intentionally not included in this public repository during the AMPLIFAI publication embargo.
