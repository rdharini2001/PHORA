PYTHON ?= python
DATA_ROOT ?= data

.PHONY: install fast paper results verify submission test
install:
	$(PYTHON) -m pip install -e ".[dev]"
fast:
	$(PYTHON) scripts/run_reproduction.py --data-root "$(DATA_ROOT)" --mode fast
paper:
	$(PYTHON) scripts/run_reproduction.py --data-root "$(DATA_ROOT)" --mode paper
results:
	$(PYTHON) scripts/show_results.py
verify:
	$(PYTHON) scripts/verify_results.py
submission:
	$(PYTHON) scripts/build_codabench_submission.py
test:
	pytest -q
