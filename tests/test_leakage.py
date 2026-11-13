"""D9.1 leakage test.

Build the features for one season twice: once with all data, once with every row dated on or
after the snapshot removed. If any feature looks into the future, the two will differ.

Run:  python -m pytest tests -v
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import FEATURES, build_features_one
from src.load import load_tables, truncate
from src.snapshot import add_target, build_population, snapshot_date


@pytest.fixture(scope="module")
def tables():
    return load_tables([config.TARGET_LEAGUE])


@pytest.mark.parametrize("season", [2018, 2022])
def test_features_do_not_use_future_data(tables, season):
    pop = add_target(tables, build_population(tables, config.TARGET_LEAGUE, season))
    full = build_features_one(tables, pop)
    cut = build_features_one(truncate(tables, snapshot_date(season)), pop)

    a = full.sort_values(["player_id", "club_id"])[FEATURES].reset_index(drop=True)
    b = cut.sort_values(["player_id", "club_id"])[FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_no_forbidden_player_columns(tables):
    forbidden = {"current_club_id", "contract_expiration_date", "market_value_in_eur",
                 "highest_market_value_in_eur", "international_caps", "last_season"}
    assert forbidden.isdisjoint(tables["players"].columns)


def test_target_only_for_players_who_stayed(tables):
    pop = add_target(tables, build_population(tables, config.TARGET_LEAGUE, 2021))
    assert pop.loc[pop["left_club"] != "stayed", "target"].isna().all()
    assert pop.loc[pop["left_club"] == "stayed", "target"].isin([0, 1]).all()
    assert np.all((pop["age"] >= config.AGE_MIN) & (pop["age"] < config.AGE_MAX))
