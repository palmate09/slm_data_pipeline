# Run pipeline and tests with the project virtualenv so dependencies
# (justext, requests, datasketch, ...) are always available.
PYTHON ?= .venv/bin/python

.PHONY: install test run clean

install:
	uv pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) -m slm_pipeline.cli --config configs/machine_learning.yaml --stage all

clean:
	rm -rf .pytest_cache slm_pipeline/__pycache__ slm_pipeline/*/__pycache__ tests/__pycache__
