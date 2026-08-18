PHORA+ v2 — AMPLIFAI Codabench inference submission

Upload the ZIP containing these files directly to the AMPLIFAI Codabench phase.
Required root files are run.py and metadata. The program reads
/app/input_data/sample_cases.csv, processes /app/data/cases/<case_id>/, and writes
/app/output/predictions.csv with case_id,prediction.

This package is dependency-safe for the official codalab/codalab-legacy:gpu310
image: it uses the image's NumPy/Pandas/SciPy and bundles a small NIfTI reader plus
a pure-Python evaluator for the fitted XGBoost trees. No pip install or network is
required at runtime.

Model: PHORA+ v2, refit on all eligible public labeled lesion cases after method
freeze. Private-test labels are never used.
