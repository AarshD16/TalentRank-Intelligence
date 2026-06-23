# Ranking Methodology

## Not Keyword Matching

This ranker is deliberately not a "count AI words" system. The job description warns that keyword stuffing is a trap, so skills and summaries are treated as supporting evidence rather than the main signal.

The strongest signals come from career-history evidence: shipped production retrieval, search, ranking, recommendation, ML systems, Python-heavy engineering work, evaluation frameworks, and product-company context. Skill names help only when they agree with titles, role descriptions, Redrob assessments, and behavioral signals. Penalties reduce profiles that list many AI terms without matching career evidence.

## Four Pillars

### Accuracy

Accuracy is approximated through transparent proxy signals because hidden relevance labels are unavailable. The model prioritizes the JD's stated needs:

- production retrieval, search, ranking, or recommendation experience;
- AI/ML engineering title relevance;
- 5-9 years preferred, with flexibility for strong 4+ or 10+ profiles;
- Python and ML systems depth;
- product-company experience;
- evaluation framework experience;
- recent hands-on technical work.

The system also applies penalties for honeypot-like patterns: expert skills with zero duration, AI keyword stuffing, inconsistent timelines, services-only backgrounds, pure research without deployment, demo-only LangChain/OpenAI experience, weak availability, and poor profile trust.

### Efficiency

The ranking step is CPU-only and streaming. It reads `candidates.jsonl` line by line, scores one candidate at a time, and keeps only the current top candidates in a heap. The normal path retains 100 candidates; debug mode retains 200 for inspection.

There are no hosted LLM calls, no network calls, no embedding generation, and no GPU dependency during ranking. The model uses deterministic dictionaries, phrase matching, structured scoring, and bounded debug output.

### Practicality

Recruiting fit is more than static resume quality. Redrob signals are used as practical hireability modifiers: recent activity, recruiter response rate, response time, saved-by-recruiters, notice period, willingness to relocate, verification, and profile completeness.

A great-on-paper candidate who has been inactive for months and rarely responds is down-weighted. A candidate with strong technical fit and healthy Redrob engagement is easier for a recruiter to act on.

### Explainability

Every score is decomposed into named components:

- role fit
- production ML
- retrieval/ranking
- evaluation
- experience
- company context
- skill quality
- Redrob availability
- trust/consistency
- logistics

Evidence extraction stores numeric values, snippets, and source fields. Reasoning text is generated from those facts only and acknowledges concerns when relevant. Debug artifacts expose component scores and penalty reasons for inspection.

## Known Limitations

- The system uses deterministic rules, so it can miss subtle plain-language evidence that does not match the dictionaries.
- Company classification is approximate; some product teams inside services companies may be under-valued.
- The model cannot validate real company founding dates or external GitHub/LinkedIn evidence because ranking must run without network access.
- Salary fit is lightly modeled because the JD does not specify a hard compensation band.
- Some role descriptions may be synthetic or incomplete, so profile trust and consistency checks are useful but not perfect.
- Without labels, proxy evaluation can identify obvious failure modes but cannot estimate true NDCG or MAP.

## Decision Support, Not Replacement

This system is intended to help recruiters triage a 100,000-candidate pool into a smaller review set. It does not make a hiring decision.

The output should be read as ranked evidence: why a candidate appears relevant, what concerns exist, and where a recruiter should inspect the profile. Final decisions still require human judgment, recruiter outreach, candidate conversation, compensation alignment, and interview evaluation.
