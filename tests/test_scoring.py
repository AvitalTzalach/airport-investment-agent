"""Unit tests for scoring.py on small synthetic data (no CSV files needed).
Run from the project root: python -m pytest"""
import math

import numpy as np
import pandas as pd
import pytest

from airport_agent.scoring import (combine, component_scores, normalize, pressure_level, score_airports,
                     sensitivity, validate_weights)
from airport_agent.scoring_config import COMPONENTS, WEIGHTS


def make_metrics(**overrides):
    """Three synthetic airports; values chosen inside the fixed ranges."""
    data = {
        "load_factor_2025": [0.85, 0.80, 0.75],
        "momentum_vs_national": [0.04, 0.00, -0.04],
        "nas_delay_rate": [0.10, 0.06, 0.03],
        "enplanements_2025": [20_000_000, 3_000_000, 700_000],
    }
    data.update(overrides)
    return pd.DataFrame(data, index=["AAA", "BBB", "CCC"])


# ---------- Config and weights ----------

def test_config_weights_sum_to_one_and_match_components():
    assert math.isclose(sum(WEIGHTS.values()), 1.0)
    assert set(WEIGHTS) == set(COMPONENTS)


@pytest.mark.parametrize("bad", [
    {**WEIGHTS, "load_factor": WEIGHTS["load_factor"] + 0.1},            # sum != 1
    {**WEIGHTS, "load_factor": -0.05, "scale": WEIGHTS["scale"] + 0.30},  # negative
    {k: v for k, v in WEIGHTS.items() if k != "scale"},                   # missing key
])
def test_invalid_weights_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_weights(bad)


# ---------- Normalization ----------

def test_normalize_is_linear_and_clipped():
    out = normalize(pd.Series([0.60, 0.70, 0.80, 0.90, 1.00]), 0.70, 0.90)
    assert list(out.round(6)) == [0.0, 0.0, 0.5, 1.0, 1.0]


def test_normalize_keeps_missing_as_missing():
    out = normalize(pd.Series([np.nan, 0.8]), 0.70, 0.90)
    assert math.isnan(out.iloc[0])


def test_scale_uses_log10_of_enplanements():
    # log10 5.7 (~500k) -> 0 ; log10 7.7 (~50M) -> 1 ; halfway -> 0.5
    comps = component_scores(make_metrics(enplanements_2025=[10 ** 7.7, 10 ** 6.7, 10 ** 5.7]))
    assert list(comps["scale"].round(3)) == [1.0, 0.5, 0.0]


# ---------- Combining ----------

def test_points_sum_exactly_to_score():
    scored = score_airports(make_metrics())
    pts = scored.filter(like="pts_").sum(axis=1)
    assert np.allclose(pts, scored["score"])


def test_full_data_score_matches_hand_calculation():
    m = make_metrics().loc[["BBB"]]
    # BBB: lf (0.80-0.70)/0.20=0.5 ; momentum (0+0.08)/0.16=0.5 ; nas (0.06-0.02)/0.10=0.4 ;
    # scale (log10(3e6)-5.7)/2
    scale = (math.log10(3_000_000) - 5.7) / 2
    expected = 100 * (0.25 * 0.5 + 0.25 * 0.5 + 0.35 * 0.4 + 0.15 * scale)
    assert math.isclose(score_airports(m)["score"].iloc[0], expected, rel_tol=1e-9)


def test_missing_component_is_reweighted_flagged_and_never_zero():
    m = make_metrics(nas_delay_rate=[np.nan, 0.06, 0.03])
    s = score_airports(m).loc["AAA"]
    assert s["partial_data"]
    assert s["missing"] == "congestion"
    assert s["pts_congestion"] == 0
    # Re-weighted score is higher than treating the missing component as zero (= score_min)
    assert s["score"] > s["score_min"]
    # Range: max = min + the full congestion weight
    assert math.isclose(s["score_max"] - s["score_min"], WEIGHTS["congestion"] * 100)
    assert s["score_min"] <= s["score"] <= s["score_max"]


def test_full_data_has_zero_width_range():
    s = score_airports(make_metrics())
    assert np.allclose(s["score_min"], s["score"]) and np.allclose(s["score_max"], s["score"])
    assert not s["partial_data"].any()


@pytest.mark.parametrize("metric, low, high", [
    ("load_factor_2025", 0.75, 0.85),
    ("momentum_vs_national", -0.02, 0.03),
    ("nas_delay_rate", 0.04, 0.09),
    ("enplanements_2025", 1_000_000, 9_000_000),
])
def test_score_is_monotonic_in_each_metric(metric, low, high):
    base = make_metrics().loc[["BBB"]]
    better = base.copy()
    base[metric], better[metric] = low, high
    assert score_airports(better)["score"].iloc[0] > score_airports(base)["score"].iloc[0]


def test_score_does_not_depend_on_peer_group():
    m = make_metrics()
    alone = score_airports(m.loc[["BBB"]])["score"].iloc[0]
    with_peers = score_airports(m).loc["BBB", "score"]
    assert math.isclose(alone, with_peers)


# ---------- Pressure levels ----------

@pytest.mark.parametrize("score, partial, expected", [
    (75, False, "High expansion pressure"),
    (60, False, "High expansion pressure"),
    (59.9, False, "Medium expansion pressure"),
    (40, False, "Medium expansion pressure"),
    (39.9, False, "Low expansion pressure"),
    (75, True, "Medium expansion pressure"),   # partial data is capped at Medium
    (30, True, "Low expansion pressure"),
])
def test_pressure_levels(score, partial, expected):
    assert pressure_level(score, partial) == expected


def test_combine_adds_pressure_level_column():
    comps = component_scores(make_metrics())
    assert "pressure_level" in combine(comps, WEIGHTS).columns


# ---------- Sensitivity ----------

def test_sensitivity_rank_range_contains_rank():
    r = sensitivity(make_metrics())
    assert ((r["rank_best"] <= r["rank"]) & (r["rank"] <= r["rank_worst"])).all()
    assert r.loc["AAA", "rank"] == 1          # best on every metric


def test_sensitivity_rejects_invalid_weights():
    with pytest.raises(ValueError):
        sensitivity(make_metrics(), {**WEIGHTS, "scale": 0.5})
