"""Deterministic Investment Opportunity Score (0-100) with a full per-component breakdown.

score = 100 * sum(weight_i * component_i) over the available components, where each component
is a metric normalized to 0-1 against a fixed range (scoring_config.COMPONENTS).
Missing components are never treated as zero: the remaining weights are re-normalized, the airport
is flagged "partial data", and a score range shows the result if the missing parts were 0 or 1.
"""
import numpy as np
import pandas as pd

from .scoring_config import (COMPONENTS, PRESSURE_LEVELS, SENSITIVITY_FACTORS, WEIGHTS,
                             WEIGHTS_TOLERANCE)


def validate_weights(weights: dict) -> None:
    """Raise ValueError if weights are not one value per component, each 0-1, summing to 1."""
    if set(weights) != set(COMPONENTS):
        raise ValueError(f"Weights must have exactly these keys: {sorted(COMPONENTS)}")
    if any(not 0 <= w <= 1 for w in weights.values()):
        raise ValueError("Each weight must be between 0 and 1")
    if abs(sum(weights.values()) - 1) > WEIGHTS_TOLERANCE:
        raise ValueError(f"Weights must sum to 1 (got {sum(weights.values()):.3f})")


def normalize(values: pd.Series, low: float, high: float) -> pd.Series:
    """Map values linearly from [low, high] to [0, 1], clipped. Missing stays missing."""
    return ((values - low) / (high - low)).clip(0, 1)


def component_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    """One 0-1 column per component (NaN where the input is missing or not meaningful)."""
    data = metrics.assign(log10_enplanements_2025=np.log10(metrics["enplanements_2025"]))
    return pd.DataFrame({
        name: normalize(data[c["metric"]], c["low"], c["high"])
        for name, c in COMPONENTS.items()
    }, index=metrics.index)


def pressure_level(score: float, partial_data: bool) -> str:
    """Label from the fixed thresholds; partial-data airports are never labeled High."""
    for threshold, label in PRESSURE_LEVELS:
        if score >= threshold:
            if partial_data and threshold == PRESSURE_LEVELS[0][0]:
                return PRESSURE_LEVELS[1][1]
            return label
    return PRESSURE_LEVELS[-1][1]


def combine(comps: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Weighted score, per-component points, partial-data flag and score range."""
    w = pd.Series(weights)[comps.columns]
    available = comps.notna()
    available_weight = available.mul(w).sum(axis=1)

    # Re-weighted contribution of each component in points; they sum exactly to the score
    points = comps.fillna(0).mul(w).div(available_weight, axis=0) * 100
    points.columns = [f"pts_{c}" for c in comps.columns]

    # Range with the ORIGINAL weights: missing components set to 0 (min) or to 1 (max)
    base = comps.fillna(0).mul(w).sum(axis=1) * 100
    missing_weight = (~available).mul(w).sum(axis=1) * 100

    result = pd.DataFrame({
        "score": points.sum(axis=1),
        "score_min": base,
        "score_max": base + missing_weight,
        "partial_data": ~available.all(axis=1),
        "missing": [", ".join(comps.columns[~row]) for row in available.to_numpy()],
    }, index=comps.index)
    result["pressure_level"] = [pressure_level(s, p) for s, p in
                                zip(result["score"], result["partial_data"])]
    return pd.concat([result, points], axis=1)


def score_airports(metrics: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """Score every airport in `metrics`; returns score, range, flags and breakdown, best first."""
    weights = weights or WEIGHTS
    validate_weights(weights)
    comps = component_scores(metrics)
    scored = combine(comps, weights)
    return pd.concat([scored, comps.add_prefix("norm_")], axis=1).sort_values("score",
                                                                              ascending=False)


def sensitivity(metrics: pd.DataFrame, weights: dict = None,
                factors: tuple = SENSITIVITY_FACTORS) -> pd.DataFrame:
    """Rank range of each airport (within `metrics`) when each weight is scaled by each factor."""
    weights = weights or WEIGHTS
    validate_weights(weights)
    comps = component_scores(metrics)
    ranks = [combine(comps, weights)["score"].rank(ascending=False, method="min")]
    for name in weights:
        for f in factors:
            varied = dict(weights, **{name: weights[name] * f})
            total = sum(varied.values())
            varied = {k: v / total for k, v in varied.items()}
            ranks.append(combine(comps, varied)["score"].rank(ascending=False, method="min"))
    ranks = pd.concat(ranks, axis=1)
    return pd.DataFrame({"rank": ranks.iloc[:, 0].astype(int),
                         "rank_best": ranks.min(axis=1).astype(int),
                         "rank_worst": ranks.max(axis=1).astype(int)})


if __name__ == "__main__":
    from .data_loader import NEW_ENGLAND, build_airport_metrics

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:,.1f}".format)

    metrics = build_airport_metrics()
    ne = metrics[metrics["state"].isin(NEW_ENGLAND)]
    scores = score_airports(ne).join(sensitivity(ne))
    cols = ["score", "pressure_level", "score_min", "score_max", "rank", "rank_best",
            "rank_worst", "missing",
            "pts_load_factor", "pts_momentum", "pts_congestion", "pts_scale"]
    print("New England:")
    print(scores[cols])

    print("\nUS top 10:")
    print(score_airports(metrics)[["score", "pressure_level", "partial_data", "pts_load_factor",
                                   "pts_momentum", "pts_congestion", "pts_scale"]].head(10))
