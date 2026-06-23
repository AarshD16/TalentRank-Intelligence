"""Streamlit sandbox demo for recruiter-facing candidate ranking."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any

try:
    import streamlit as st
except ImportError:  # pragma: no cover - exercised only when Streamlit is not installed.
    st = None  # type: ignore[assignment]

from src.config import RankingConfig
from src.reasoning import build_reasoning
from src.scoring import COMPONENT_NAMES, ScoreBreakdown, score_candidate_with_components


DISPLAY_COMPONENTS = (
    "role_fit_score",
    "production_ml_score",
    "retrieval_ranking_score",
    "evaluation_score",
    "skill_quality_score",
    "redrob_availability_score",
)


@dataclass(frozen=True, slots=True)
class DemoResult:
    candidate: dict[str, Any]
    breakdown: ScoreBreakdown
    rank: int
    reasoning: str


def main() -> None:
    if st is None:
        raise RuntimeError("Streamlit is required to run the sandbox app. Install requirements.txt first.")
    st.set_page_config(page_title="TalentRank Sandbox", page_icon="TR", layout="wide")
    _apply_styles()

    st.title("TalentRank Sandbox")
    st.caption("Candidate ranking demo for recruiter review. Upload up to 100 candidate profiles as JSON or JSONL.")

    uploaded = st.file_uploader(
        "Candidate sample",
        type=("json", "jsonl"),
        help="Upload a JSON array or newline-delimited JSON file using the Redrob candidate schema.",
    )

    if uploaded is None:
        _empty_state()
        return

    try:
        candidates = _parse_upload(uploaded.read())
    except ValueError as exc:
        st.error(str(exc))
        return

    if len(candidates) > 100:
        st.error(f"Sandbox accepts at most 100 candidates; uploaded {len(candidates)}.")
        return
    if not candidates:
        st.error("No candidate objects found in the uploaded file.")
        return

    with st.spinner("Ranking candidates..."):
        results = _rank_for_demo(candidates)

    _summary_metrics(results)
    _ranked_table(results)
    _download_button(results)
    _candidate_explainers(results)


def _parse_upload(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of candidate objects.")
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("Every JSON array item must be a candidate object.")
        return data

    candidates: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}.") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Expected candidate object at line {line_number}.")
        candidates.append(item)
    return candidates


def _rank_for_demo(candidates: list[dict[str, Any]]) -> list[DemoResult]:
    config = RankingConfig()
    scored = [(candidate, score_candidate_with_components(candidate, config)) for candidate in candidates]
    scored.sort(key=lambda item: (-item[1].final_score, item[1].candidate_id))

    results: list[DemoResult] = []
    for rank, (candidate, breakdown) in enumerate(scored, start=1):
        reasoning = build_reasoning(
            evidence=breakdown.evidence,
            penalties=breakdown.penalties,
            rank=rank,
            component_scores=breakdown.component_scores,
            candidate=candidate,
        )
        results.append(DemoResult(candidate=candidate, breakdown=breakdown, rank=rank, reasoning=reasoning))
    return results


def _summary_metrics(results: list[DemoResult]) -> None:
    top = results[0]
    hard_flag_count = sum(1 for result in results if result.breakdown.penalties.hard_flags)
    retrieval_count = sum(
        1 for result in results if result.breakdown.evidence.value("retrieval_search_ranking_experience") >= 0.45
    )
    avg_response = _mean(
        _safe_float(_signals(result.candidate).get("recruiter_response_rate")) for result in results
    )

    cols = st.columns(4)
    cols[0].metric("Candidates", len(results))
    cols[1].metric("Top Score", f"{top.breakdown.final_score:.3f}")
    cols[2].metric("Retrieval Evidence", f"{retrieval_count}/{len(results)}")
    cols[3].metric("Hard Flags", hard_flag_count)
    st.caption(f"Average recruiter response rate: {avg_response:.2f}")


def _ranked_table(results: list[DemoResult]) -> None:
    st.subheader("Ranked Candidates")
    rows = []
    for result in results:
        profile = _profile(result.candidate)
        penalties = result.breakdown.penalties
        row: dict[str, Any] = {
            "rank": result.rank,
            "candidate_id": result.breakdown.candidate_id,
            "score": round(result.breakdown.final_score, 4),
            "title": profile.get("current_title", ""),
            "years": profile.get("years_of_experience", ""),
            "reasoning": result.reasoning,
            "penalty_flags": " | ".join((*penalties.hard_flags, *penalties.soft_flags)),
        }
        row.update(
            {
                name.replace("_score", ""): round(result.breakdown.component_scores.get(name, 0.0), 3)
                for name in DISPLAY_COMPONENTS
            }
        )
        rows.append(row)
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _download_button(results: list[DemoResult]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    for result in results:
        writer.writerow(
            {
                "candidate_id": result.breakdown.candidate_id,
                "rank": result.rank,
                "score": f"{result.breakdown.final_score:.6f}",
                "reasoning": result.reasoning,
            }
        )
    st.download_button(
        "Download ranked CSV",
        data=buffer.getvalue(),
        file_name="sandbox_ranked_candidates.csv",
        mime="text/csv",
    )


def _candidate_explainers(results: list[DemoResult]) -> None:
    st.subheader("Candidate Explanations")
    for result in results:
        profile = _profile(result.candidate)
        label = (
            f"#{result.rank} {result.breakdown.candidate_id} - "
            f"{profile.get('current_title', 'Unknown title')} - {result.breakdown.final_score:.3f}"
        )
        with st.expander(label):
            st.write(result.reasoning)
            left, middle, right = st.columns([1.1, 1, 1])
            with left:
                st.markdown("**Evidence snippets**")
                snippets = _evidence_snippets(result.breakdown)
                if snippets:
                    for signal_name, snippet in snippets:
                        st.markdown(f"- `{signal_name}`: {snippet}")
                else:
                    st.caption("No strong snippets extracted.")
            with middle:
                st.markdown("**Score breakdown**")
                for name in COMPONENT_NAMES:
                    st.progress(
                        result.breakdown.component_scores.get(name, 0.0),
                        text=f"{name.replace('_score', '').replace('_', ' ')}: "
                        f"{result.breakdown.component_scores.get(name, 0.0):.2f}",
                    )
            with right:
                st.markdown("**Redrob availability**")
                signals = _signals(result.candidate)
                availability_rows = {
                    "open_to_work": signals.get("open_to_work_flag"),
                    "last_active": signals.get("last_active_date"),
                    "response_rate": signals.get("recruiter_response_rate"),
                    "avg_response_hours": signals.get("avg_response_time_hours"),
                    "notice_days": signals.get("notice_period_days"),
                    "willing_to_relocate": signals.get("willing_to_relocate"),
                }
                st.json(availability_rows, expanded=False)
                st.markdown("**Concerns**")
                penalties = result.breakdown.penalties
                concerns = penalties.penalty_reasons or ("No major concerns detected.",)
                for concern in concerns:
                    st.markdown(f"- {concern}")


def _evidence_snippets(breakdown: ScoreBreakdown) -> list[tuple[str, str]]:
    priority = (
        "retrieval_search_ranking_experience",
        "production_ai_experience",
        "evaluation_framework_experience",
        "python_strength",
        "product_company_experience",
        "availability_signal",
    )
    snippets: list[tuple[str, str]] = []
    for signal_name in priority:
        signal = breakdown.evidence.signals.get(signal_name)
        if signal and signal.snippets:
            snippets.append((signal_name, signal.snippets[0]))
    return snippets[:6]


def _empty_state() -> None:
    st.info("Upload a small candidate sample to inspect ranking, reasoning, evidence, and penalties.")
    st.markdown(
        """
        **Expected input**
        - JSON array of candidate objects, or JSONL with one candidate object per line.
        - Maximum 100 candidates for the sandbox.
        - Same schema used by the ranking CLI.
        """
    )


def _profile(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate.get("profile", {})
    return profile if isinstance(profile, dict) else {}


def _signals(candidate: dict[str, Any]) -> dict[str, Any]:
    signals = candidate.get("redrob_signals", {})
    return signals if isinstance(signals, dict) else {}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Any) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            max-width: 1280px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #d8dee4;
            border-radius: 8px;
            padding: 12px 14px;
            background: #fafbfc;
        }
        div[data-testid="stExpander"] {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
