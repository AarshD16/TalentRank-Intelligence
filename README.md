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
python rank.py --candidates candidates.jsonl --out submission.csv
```

## Sandbox Demo

Run the lightweight recruiter-facing Streamlit demo with:

```bash
streamlit run streamlit_app.py
```

The app accepts a JSON or JSONL candidate sample of up to 100 profiles and uses the same deterministic scoring, penalty, and reasoning logic as the CLI.

## Development

Run tests with:

```bash
python -m pytest
```

The tests included at this stage cover configuration defaults and submission contract validation only.
