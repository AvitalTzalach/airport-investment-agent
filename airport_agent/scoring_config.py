"""All scoring parameters in one place (approved 2026-11-13, revised 2026-11-14).
Change only with approval."""

# Component weights (must sum to 1)
WEIGHTS = {
    "load_factor": 0.25,   # demand pressure: how full the planes are
    "momentum": 0.25,      # recent demand: 2026 YTD growth minus national growth
    "congestion": 0.35,    # share of arriving flights delayed by NAS (capacity) causes
    "scale": 0.15,         # airport size (log scale), so size alone does not win
}

# Fixed normalization ranges: metric value at score 0 and at score 1 (clipped outside).
# Chosen against the US distribution (docs/sanity_checks.md), not against the peer group,
# so a score does not change when the set of compared airports changes.
COMPONENTS = {
    "load_factor": {"metric": "load_factor_2025", "low": 0.70, "high": 0.90},
    "momentum": {"metric": "momentum_vs_national", "low": -0.08, "high": 0.08},
    "congestion": {"metric": "nas_delay_rate", "low": 0.02, "high": 0.12},
    # log10 of enplanements: 5.7 = 500k, 7.7 = 50M
    "scale": {"metric": "log10_enplanements_2025", "low": 5.7, "high": 7.7},
}

# Expansion pressure levels by score (fixed thresholds, approved 2026-11-13).
# Hebrew labels: לחץ הרחבה גבוה / בינוני / נמוך.
# An airport with partial data is never labeled "High" (capped at "Medium" and flagged).
PRESSURE_LEVELS = [
    (60, "High expansion pressure"),
    (40, "Medium expansion pressure"),
    (0, "Low expansion pressure"),
]

# Sensitivity check: each weight is multiplied by these factors (then re-normalized to sum 1)
SENSITIVITY_FACTORS = (0.5, 1.5)

# Allowed deviation of the weights sum from 1
WEIGHTS_TOLERANCE = 0.001
