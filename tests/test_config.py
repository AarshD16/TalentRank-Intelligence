from __future__ import annotations

import pytest

from src.config import RankingConfig


def test_ranking_config_requires_exactly_100_rows() -> None:
    with pytest.raises(ValueError, match="exactly 100"):
        RankingConfig(top_k=50)


def test_ranking_config_columns_match_submission_contract() -> None:
    config = RankingConfig()

    assert config.csv_columns == ("candidate_id", "rank", "score", "reasoning")
