.PHONY: test rank validate benchmark

PYTHON ?= python
CANDIDATES ?= candidates.jsonl
OUT ?= submission.csv

test:
	$(PYTHON) -m pytest

rank:
	$(PYTHON) rank.py --candidates $(CANDIDATES) --out $(OUT)

validate:
	$(PYTHON) validate_submission.py $(OUT)

benchmark:
	$(PYTHON) scripts/benchmark.py --candidates $(CANDIDATES)
