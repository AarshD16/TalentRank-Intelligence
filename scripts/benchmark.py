"""Benchmark the ranking hot path on a candidate JSONL file."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rank import DEBUG_TOP_K, PipelineTimings, _score_streaming, _sort_best_first
from src.config import RankingConfig
from src.validation import validate_input_path


LOGGER = logging.getLogger(__name__)
TARGET_CANDIDATES = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ranker throughput.")
    parser.add_argument("--candidates", default=Path("candidates.jsonl"), type=Path)
    parser.add_argument("--limit", default=None, type=int, help="Only process first N candidates.")
    parser.add_argument(
        "--debug-heap",
        action="store_true",
        help="Use the debug heap size of 200 instead of final heap size of 100.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    validate_input_path(args.candidates)

    config = RankingConfig()
    timings = PipelineTimings()
    keep_k = DEBUG_TOP_K if args.debug_heap else config.top_k

    memory_before = _memory_mb()
    start = time.perf_counter()
    top_candidates, processed = _score_streaming(
        args.candidates,
        config=config,
        keep_k=keep_k,
        timings=timings,
        limit=args.limit,
    )
    sort_start = time.perf_counter()
    _sort_best_first(top_candidates)
    timings.sort_seconds += time.perf_counter() - sort_start
    elapsed = time.perf_counter() - start
    memory_after = _memory_mb()

    candidates_per_second = processed / elapsed if elapsed > 0 else 0.0
    estimated_100k_seconds = TARGET_CANDIDATES / candidates_per_second if candidates_per_second > 0 else 0.0
    slowest_stage, slowest_seconds = max(timings.as_dict().items(), key=lambda item: item[1])

    print("# Ranker Benchmark")
    print(f"Candidates processed: {processed}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print(f"Candidates/sec: {candidates_per_second:.1f}")
    print(f"Estimated runtime for 100,000 candidates: {_format_duration(estimated_100k_seconds)}")
    print(f"Slowest pipeline stage: {slowest_stage} ({slowest_seconds:.3f}s)")
    print("Stage timings:")
    for stage, seconds in sorted(timings.as_dict().items(), key=lambda item: item[1], reverse=True):
        if seconds > 0:
            pct = seconds / elapsed * 100 if elapsed > 0 else 0.0
            print(f"  {stage}: {seconds:.3f}s ({pct:.1f}%)")
    if memory_after is None:
        print("Memory usage: unavailable (psutil not installed)")
    else:
        delta = 0.0 if memory_before is None else memory_after - memory_before
        print(f"Memory RSS: {memory_after:.1f} MB (delta {delta:+.1f} MB)")
    return 0


def _memory_mb() -> float | None:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def _format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    if minutes:
        return f"{minutes}m {remainder:.1f}s"
    return f"{remainder:.1f}s"


if __name__ == "__main__":
    raise SystemExit(main())
