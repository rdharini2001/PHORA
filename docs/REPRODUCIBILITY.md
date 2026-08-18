# Reproducing the paper experiments

## Installation

```bash
git clone <YOUR-GITHUB-URL>
cd PHORA
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

Place the AMPLIFAI public data under `data/` as described in [DATA.md](DATA.md).

## Fast smoke test

```bash
python scripts/run_reproduction.py --data-root data --mode fast
```

This uses one ensemble seed, 100 screened features, 900 router trials and 1,500 bootstrap replicates; expensive ablations are disabled.

## Full paper run

```bash
python scripts/run_reproduction.py --data-root data --mode paper
```

Paper mode uses 140 screened features, ensemble seeds `{17,37,61}`, 5,000 router trials, 10,000 bootstrap replicates, architecture ablations, feature ablations, repeated-seed CV and source-robustness analysis. Feature extraction is cached in the selected output directory.

Equivalent Make targets are available:

```bash
make fast DATA_ROOT=/path/to/amplifai
make paper DATA_ROOT=/path/to/amplifai
make results
make verify
```

The stored public-validation outputs in `results/` are included so figures/tables and metric calculations can be inspected without rerunning CT feature extraction.
