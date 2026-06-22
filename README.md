# TalentRank Intelligence

Explainable hybrid candidate-ranking engine for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

This repository is currently a production-oriented skeleton. It defines the CLI, package boundaries, validation contracts, and deterministic execution surface, but it intentionally does not implement scoring yet.

## Requirements

- Input: `candidates.jsonl` with candidate profiles.
- Output: `submission.csv` with exactly these columns:
  - `candidate_id`
  - `rank`
  - `score`
  - `reasoning`
- Output must contain exactly 100 ranked rows.
- Ranking target: under 5 minutes on CPU with 16 GB RAM.
- No network calls during ranking.
- No hosted LLM calls during ranking.
- Deterministic and reproducible output.

## Project Layout

```text
.
├── rank.py
├── src/
│   ├── config.py
│   ├── evidence_extractor.py
│   ├── io.py
│   ├── normalization.py
│   ├── penalties.py
│   ├── reasoning.py
│   ├── scoring.py
│   └── validation.py
├── tests/
│   ├── test_config.py
│   └── test_validation.py
├── README.md
└── submission_metadata.yaml
```

## Usage

```bash
python rank.py --input candidates.jsonl --output submission.csv
```

The skeleton parses arguments and configures logging, then stops at the unimplemented ranking stage. Future implementation should fill in the TODOs inside `src/` without changing the output contract.

## Development

Run tests with:

```bash
python -m pytest
```

The tests included at this stage cover configuration defaults and submission contract validation only.
