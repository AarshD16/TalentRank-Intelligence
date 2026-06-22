# Feature Audit

This audit is based on `candidate_schema.json`, the sample candidates, the Senior AI Engineer job description, the Redrob signal reference, and the submission specification.

## Useful Fields For Ranking

The strongest ranking fields are the ones that connect directly to the job description and can be checked against profile evidence:

- `career_history[].title`, `career_history[].description`, `career_history[].industry`, and `career_history[].company`: best source for production evidence. The JD values shipped retrieval, ranking, search, recommendation, embeddings, vector infrastructure, evaluation frameworks, and product engineering over keyword lists.
- `profile.current_title`, `profile.headline`, and `profile.summary`: useful high-level signals, but should be cross-checked against career history.
- `profile.years_of_experience`: useful for the target band around 5-9 years, with flexibility for strong adjacent evidence.
- `skills[].name`, `skills[].proficiency`, `skills[].duration_months`, and `skills[].endorsements`: useful when skills are supported by role descriptions and assessment scores.
- `education[].tier`, `education[].degree`, and `education[].field_of_study`: secondary signal. Tier 1/2 CS or adjacent education helps, but should not dominate production experience.
- `certifications[]`: weak-to-moderate supporting evidence, especially cloud, ML, search, or data infra credentials.
- `redrob_signals.skill_assessment_scores`: useful validation for claimed skills, especially Python, ML, NLP, ranking, search, data engineering, and related assessments.
- `redrob_signals.github_activity_score`: useful for external technical activity, but absence of GitHub should not be a hard reject.

## Risky Or Noisy Fields

These fields can help, but should be down-weighted or used only with corroboration:

- `skills[].name` alone: the JD explicitly warns that keyword matching is a trap. A Marketing Manager with AI keywords should not outrank a product engineer with real ranking/search history.
- `profile.summary`: can be self-promotional and keyword dense. Use it for recall, then verify against career history.
- `profile.current_company`: company names are useful context, but ranking solely by brand can miss strong startup/product candidates.
- `profile.current_industry`: helpful for filtering services-only or non-product paths, but not definitive.
- `endorsements_received`, `connection_count`, `profile_views_received_30d`, and `search_appearance_30d`: popularity signals can reflect platform dynamics rather than fit.
- `expected_salary_range_inr_lpa`: useful for feasibility only. Avoid penalizing strongly without compensation constraints.
- `applications_submitted_30d`: ambiguous. Some applications indicate market availability; very high counts may indicate low selectivity.
- `languages`: likely not central for this JD unless communication or location constraints are added later.

## Redrob Signals As Multipliers

Redrob behavioral signals should modify a profile-fit score rather than replace it.

Positive multipliers:

- `last_active_date`: recent activity means the candidate can plausibly be reached.
- `open_to_work_flag`: direct availability signal.
- `recruiter_response_rate`: strong hireability signal. A low response rate should materially reduce otherwise strong profiles.
- `avg_response_time_hours`: faster responses improve practical hiring odds.
- `notice_period_days`: sub-30-day notice is preferred by the JD; long notice should reduce rank unless technical fit is exceptional.
- `willing_to_relocate`: important because Pune/Noida is preferred and relocation from Tier-1 Indian cities is acceptable.
- `preferred_work_mode`: hybrid/flexible/onsite can help for Pune/Noida expectations; remote-only may be less aligned.
- `interview_completion_rate`: high completion suggests reliable funnel behavior.
- `offer_acceptance_rate`: positive history helps; `-1` should be treated as unknown, not bad.
- `verified_email`, `verified_phone`, and `linkedin_connected`: trust/contactability multipliers.
- `saved_by_recruiters_30d`: recruiter demand signal, useful when not contradicted by weak fit.
- `skill_assessment_scores`: strong multiplier when assessment names align with JD-critical skills.
- `github_activity_score`: positive multiplier for AI/ML engineering credibility, especially for open-source expectations.

Signals to cap or smooth:

- `profile_views_received_30d`, `search_appearance_30d`, `connection_count`, and `endorsements_received` should use capped or log-scaled effects so popularity cannot swamp evidence.
- `applications_submitted_30d` should likely have a moderate optimum rather than monotonic benefit.

## Honeypot And Inconsistency Detection

The submission spec warns about honeypots with subtly impossible profiles. Useful detection fields include:

- `career_history[].start_date`, `career_history[].end_date`, and `career_history[].duration_months`: check that stated durations match dates and that total role duration roughly matches `profile.years_of_experience`.
- `career_history[].is_current`: normally exactly one current role should exist.
- `skills[].proficiency` plus `skills[].duration_months`: expert skills with zero months are suspicious.
- `skills[]` count and AI keyword density: very large skill lists, especially with many AI terms and weak career evidence, suggest keyword stuffing.
- `profile.current_title` versus `skills[].name`: AI-heavy skills attached to a non-technical or unrelated title should be scrutinized.
- `profile.years_of_experience` versus seniority claims: low experience with many expert AI/ML skills is suspicious.
- `redrob_signals.expected_salary_range_inr_lpa.min/max`: min greater than max is inconsistent.
- `redrob_signals.signup_date` and `last_active_date`: future dates or last active before signup should be flagged.
- `redrob_signals.skill_assessment_scores` versus skills: many claimed expert skills with no corresponding assessments or weak scores should be down-weighted.
- `career_history[].company`, `career_history[].industry`, and `profile.current_industry`: services-only careers are explicitly risky for this JD unless there is strong product-company evidence elsewhere.

The ranking engine should treat these checks as penalties or eligibility gates for top-100 inclusion. They should also feed the `reasoning` column so manual reviewers can see honest concerns rather than generic praise.
