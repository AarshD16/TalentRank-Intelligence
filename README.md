# TalentRank Intelligence

TalentRank Intelligence is a production-style candidate ranking application for recruiter-facing shortlisting.

The system compiles each job description into a validated RoleSpec scoring policy, extracts reusable career evidence from candidate profiles, and applies Redrob hireability signals only as bounded modifiers. Ranking is deterministic after the RoleSpec is created.

## What It Does

1. Upload a job description.
2. Upload candidates as JSONL or JSON.
3. The backend checks whether this JD already has a cached LLM RoleSpec.
4. If cached, it reuses the policy; if not, it calls the configured Ollama-compatible model once, validates the RoleSpec, and caches it.
5. The offline ranking engine scores candidates and exports `submission.csv`.

The LLM never ranks candidates and never generates Python code.

## Docker

Create a local `.env` file from the template:

```bash
cp .env.example .env
```

Set your Ollama-compatible endpoint:

```env
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_MODEL=your-model-name
OLLAMA_API_KEY=your-api-key
```

Run the app:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8080
```

Stop it:

```bash
docker compose down
```

## Local Development

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI backend:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8080
```

Run the React frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

The Vite dev server proxies `/api` requests to FastAPI on port `8080`.

## CLI

The original CLI flow is preserved:

```bash
python rank.py --candidates candidates.jsonl --role configs/rolespec_redrob_senior_ai_engineer.yaml --out submission.csv
```

You can also compile a RoleSpec from a JD first:

```bash
python scripts/generate_rolespec.py --jd job_description.docx --out configs/rolespec_redrob_senior_ai_engineer.yaml
python rank.py --candidates candidates.jsonl --role configs/rolespec_redrob_senior_ai_engineer.yaml --out submission.csv --debug
```

## API

```text
GET  /api/health
POST /api/jobs
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/results
GET  /api/jobs/{job_id}/download/submission
GET  /api/jobs/{job_id}/download/debug
```

The web app submits `rolespec_mode=auto_llm`, which means cache-first RoleSpec reuse and LLM compilation only on cache miss.

## Output Format

`submission.csv` uses the challenge contract:

```csv
candidate_id,rank,score,reasoning
```

Validate an export:

```bash
python validate_submission.py submission.csv
```

## Project Structure

```text
backend/          FastAPI routes, background jobs, upload and ranking services
frontend/         React + Vite + TypeScript recruiter dashboard
src/              Deterministic evidence extraction, scoring, penalties, reasoning
scripts/          RoleSpec generation and ranker diagnostics
configs/          Saved RoleSpec and taxonomy configs
tests/            Unit and architecture tests
rank.py           CLI entrypoint plus reusable run_ranking function
Dockerfile        Multi-stage React build + Python runtime
docker-compose.yml
```

## Tests

```bash
python -m pytest
```
