"""Agent tools: validated, deterministic wrappers over data_loader and scoring.

Plain Python (no LangChain): agent.py wraps these functions as tools. Every tool returns a dict:
  success: {"result": ..., "assumptions": [...], "warnings": [...], "data_period": "..."}
  invalid input: {"error": "<code>", "message": "...", "suggestions": [...]}
The LLM never computes numbers; it explains what these functions return.
"""
import difflib
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import live_api
from .data_loader import (DATA_DIR, CARGO_CLASSES, MIN_ENPLANEMENTS, NEW_ENGLAND, PASSENGER_CLASSES,
                         US_STATES, build_airport_metrics, delay_window, load_airport_names,
                         load_delays, load_t100, momentum_months, passenger_flights, us_origins)
from .scoring import score_airports, sensitivity

CONTEXT_FILE = DATA_DIR / "airport_context.json"

# US Census Bureau divisions (standard, defensible regions) + the whole US
REGIONS = {
    "New England": NEW_ENGLAND,
    "Middle Atlantic": ["NJ", "NY", "PA"],
    "East North Central": ["IL", "IN", "MI", "OH", "WI"],
    "West North Central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South Atlantic": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "East South Central": ["AL", "KY", "MS", "TN"],
    "West South Central": ["AR", "LA", "OK", "TX"],
    "Mountain": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "Pacific": ["AK", "CA", "HI", "OR", "WA"],
    "US": US_STATES,
}

# Ambiguous city names -> the main commercial airports (used only for suggestions;
# the agent states which one it assumed, e.g. "LA" = LAX)
METRO_ALIASES = {
    "LA": ["LAX", "BUR", "LGB", "ONT", "SNA"],
    "NYC": ["JFK", "LGA", "EWR"],
    "DC": ["DCA", "IAD", "BWI"],
    "SF": ["SFO", "OAK", "SJC"],
    "CHICAGO": ["ORD", "MDW"],
}

# Input limits (approved 2026-11-14)
LONG_HAUL_MIN_MILES, LONG_HAUL_MAX_MILES, LONG_HAUL_DEFAULT_MILES = 500, 8000, 3000
LONG_HAUL_YEAR = 2025
LONG_HAUL_MODES = ("passenger", "cargo", "all")
COMPARE_MIN, COMPARE_MAX = 2, 5
TOP_N_MAX = 50

BASE_ASSUMPTIONS = [
    "Passenger volume = enplanements (departing passengers, FAA definition); total traffic is ~2x.",
    "Passenger metrics use scheduled passenger flights only (T-100 CLASS F).",
    "The score is a relative expansion-pressure indicator (demand vs capacity), not a profit forecast.",
]
DELAY_NOTE = ("Delay data covers large US reporting carriers only; airports without it are shown "
              "as missing, never as zero.")


# ---------- Cached data (loaded once per process) ----------

@lru_cache(maxsize=1)
def _t100() -> pd.DataFrame:
    return us_origins(load_t100())


@lru_cache(maxsize=1)
def _delays() -> pd.DataFrame:
    return load_delays()


@lru_cache(maxsize=1)
def _all_metrics() -> pd.DataFrame:
    """Metrics for every US airport with scheduled passenger service (no size filter)."""
    return build_airport_metrics(min_enplanements=0, t100=_t100(), delays=_delays())


def _universe() -> pd.DataFrame:
    """Ranking universe: US airports with at least MIN_ENPLANEMENTS in 2025."""
    m = _all_metrics()
    return m[m["enplanements_2025"] >= MIN_ENPLANEMENTS]


@lru_cache(maxsize=1)
def _national_scores() -> pd.DataFrame:
    return score_airports(_universe())


@lru_cache(maxsize=1)
def _names() -> pd.DataFrame:
    return load_airport_names()


@lru_cache(maxsize=1)
def _known_codes() -> frozenset:
    return frozenset(_t100()["ORIGIN"].unique())


@lru_cache(maxsize=1)
def _context() -> dict:
    if not CONTEXT_FILE.exists():
        return {}
    return json.loads(CONTEXT_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _data_period() -> str:
    def fmt(p):
        return pd.Timestamp(year=p // 100, month=p % 100, day=1).strftime("%b %Y")
    last_month = int(momentum_months(passenger_flights(_t100()))[-1])
    window = delay_window(_delays())
    return (f"T-100 2025 (volume, load factor); momentum = Jan-{fmt(202600 + last_month)} "
            f"vs same months 2025; delays {fmt(window[0])} - {fmt(window[-1])}")


# ---------- Helpers ----------

def _error(code: str, message: str, suggestions: list = None) -> dict:
    return {"error": code, "message": message, "suggestions": suggestions or []}


def _response(result, assumptions=(), warnings=(), data_period: str = None) -> dict:
    return {"result": result, "assumptions": BASE_ASSUMPTIONS + list(assumptions),
            "warnings": list(warnings), "data_period": data_period or _data_period()}


def _num(x, digits=3):
    """JSON-friendly rounded number; missing stays None (never 0)."""
    return None if pd.isna(x) else round(float(x), digits)


def _label(code: str) -> str:
    names = _names()
    if code in names.index:
        row = names.loc[code]
        return f"{code} - {row['name']} ({row['municipality']})"
    return code


def _suggest(query: str) -> list:
    """Known US airports matching the query: metro alias, name/city, or a similar code."""
    q = str(query).strip()
    if q.upper() in METRO_ALIASES:
        return [_label(c) for c in METRO_ALIASES[q.upper()]]
    known = _known_codes()
    names = _names()
    names = names[names.index.isin(known)]
    hits = []
    if len(q) >= 3:
        mask = (names["name"].str.contains(q, case=False, regex=False, na=False)
                | names["municipality"].str.contains(q, case=False, regex=False, na=False))
        sizes = _all_metrics()["enplanements_2025"]
        hits = sorted(names.index[mask], key=lambda c: -sizes.get(c, 0))[:3]
    if len(hits) < 3 and len(q) == 3:
        # Likely a typo in a code: passenger airports sharing 2 of 3 letters, largest first
        sizes = _all_metrics()["enplanements_2025"].sort_values(ascending=False)
        similar = [c for c in sizes.index if c not in hits and (
            sum(a == b for a, b in zip(c, q.upper())) >= 2 or sorted(c) == sorted(q.upper()))]
        hits += similar[:3 - len(hits)]
    if not hits and len(q) == 3:
        hits = difflib.get_close_matches(q.upper(), known, n=3, cutoff=0.6)
    return [_label(c) for c in hits[:3]]


def _check_airport(code):
    """Return (CODE, None) for a known US airport, else (None, error dict)."""
    q = str(code).strip().upper() if code is not None else ""
    if len(q) == 3 and q in _known_codes():
        return q, None
    return None, _error("unknown_airport",
                        f"'{code}' is not a US airport in our data (expected a 3-letter IATA code).",
                        _suggest(code) if q else [])


def _score_rows(scored: pd.DataFrame, metrics: pd.DataFrame, ranks: pd.DataFrame = None) -> list:
    """Turn scored rows into JSON-friendly dicts with the full breakdown."""
    rows = []
    for code, s in scored.iterrows():
        m = metrics.loc[code]
        row = {
            "airport": _label(code),
            "state": m["state"],
            "score": _num(s["score"], 1),
            "pressure_level": s["pressure_level"],
            "partial_data": bool(s["partial_data"]),
            "points": {c[4:]: _num(s[c], 1) for c in scored.columns if c.startswith("pts_")},
            "metrics": {
                "enplanements_2025": int(m["enplanements_2025"]),
                "load_factor_2025": _num(m["load_factor_2025"]),
                "momentum_vs_national": _num(m["momentum_vs_national"]),
                "nas_delay_rate": _num(m["nas_delay_rate"]),
            },
        }
        if s["partial_data"]:
            row["missing_components"] = s["missing"]
            row["score_range"] = [_num(s["score_min"], 1), _num(s["score_max"], 1)]
        if ranks is not None:
            r = ranks.loc[code]
            row["rank"] = int(r["rank"])
            row["rank_range_under_weight_changes"] = [int(r["rank_best"]), int(r["rank_worst"])]
        rows.append(row)
    return rows


def _tie_warnings(ranks: pd.DataFrame) -> list:
    """Airports whose rank ranges overlap under weight changes are a tie, not a ranking."""
    r = ranks.sort_values("rank")
    codes = list(r.index)
    out = []
    for a, b in zip(codes, codes[1:]):
        if r.loc[a, "rank_worst"] >= r.loc[b, "rank_best"]:
            out.append(f"{a} and {b} swap places under moderate weight changes: treat as a tie.")
    return out


def _partial_warnings(scored: pd.DataFrame) -> list:
    return [f"{c}: partial data (missing {s['missing']}); score range "
            f"{s['score_min']:.0f}-{s['score_max']:.0f}; capped at Medium."
            for c, s in scored[scored["partial_data"]].iterrows()]


def _context_for(code: str) -> list:
    return _context().get(code, [])


# ---------- Tools ----------

def get_airport_profile(airport: str) -> dict:
    """All metrics for one US airport, its score breakdown and national rank, plus background."""
    code, err = _check_airport(airport)
    if err:
        return err
    all_m = _all_metrics()
    if code not in all_m.index:
        return _response({"airport": _label(code), "scored": False,
                          "note": "No scheduled passenger service in T-100 2025/2026 data."})
    m = all_m.loc[code]
    raw = {k: (_num(v) if isinstance(v, float) else v) for k, v in m.items()}
    raw["enplanements_2025"] = int(m["enplanements_2025"])

    if m["enplanements_2025"] < MIN_ENPLANEMENTS:
        return _response(
            {"airport": _label(code), "scored": False, "metrics": raw,
             "note": f"Not scored: {int(m['enplanements_2025']):,} enplanements in 2025 is below "
                     f"the {MIN_ENPLANEMENTS:,} minimum for terminal-investment ranking.",
             "background": _context_for(code)},
            assumptions=[DELAY_NOTE])

    national = _national_scores()
    row = _score_rows(national.loc[[code]], _universe())[0]
    row["national_rank"] = f"{list(national.index).index(code) + 1} of {len(national)}"
    row["metrics"] = raw
    row["scored"] = True
    row["background"] = _context_for(code)
    return _response(row, assumptions=[DELAY_NOTE], warnings=_partial_warnings(national.loc[[code]]))


def rank_airports(region: str = None, states: list = None, top_n: int = 10) -> dict:
    """Rank US airports in a region (Census division or 'US') or a list of states by score."""
    if (region is None) == (states is None):
        return _error("unknown_region", "Give exactly one of: region or states.",
                      [f"Known regions: {', '.join(REGIONS)}"])
    if region is not None:
        key = {r.lower(): r for r in REGIONS}.get(str(region).strip().lower())
        if key is None:
            return _error("unknown_region", f"'{region}' is not a known region.",
                          [f"Known regions: {', '.join(REGIONS)}", "Or pass a list of state codes."])
        state_list, scope = REGIONS[key], key
    else:
        state_list = [str(s).strip().upper() for s in (states or [])]
        bad = [s for s in state_list if s not in US_STATES]
        if not state_list or bad:
            return _error("unknown_region", f"Unknown state codes: {bad or 'empty list'}.",
                          ["Use 2-letter US state codes, e.g. ['CA', 'NV']."])
        scope = ", ".join(state_list)
    if not isinstance(top_n, int) or not 1 <= top_n <= TOP_N_MAX:
        return _error("out_of_range", f"top_n must be an integer from 1 to {TOP_N_MAX}.")

    peers = _universe()[_universe()["state"].isin(state_list)]
    if peers.empty:
        return _response({"scope": scope, "airports": []},
                         warnings=[f"No airport in {scope} has at least {MIN_ENPLANEMENTS:,} "
                                   "enplanements, so none is ranked."])
    scored = score_airports(peers)
    ranks = sensitivity(peers)
    top = scored.head(top_n)
    result = {"scope": scope, "airports_ranked": len(scored), "shown": len(top),
              "airports": _score_rows(top, peers, ranks)}
    return _response(result, assumptions=[DELAY_NOTE],
                     warnings=_partial_warnings(top) + _tie_warnings(ranks.loc[top.index]))


def compare_airports(airports: list) -> dict:
    """Compare 2-5 US airports side by side: score, breakdown and congestion indicators."""
    if not isinstance(airports, (list, tuple)):
        return _error("invalid_count", "airports must be a list of 2-5 airport codes.")
    codes = []
    for a in airports:
        code, err = _check_airport(a)
        if err:
            return err
        codes.append(code)
    if len(set(codes)) != len(codes) or not COMPARE_MIN <= len(codes) <= COMPARE_MAX:
        return _error("invalid_count",
                      f"Compare {COMPARE_MIN}-{COMPARE_MAX} different airports (got {codes}).")

    universe = _universe()
    eligible = [c for c in codes if c in universe.index]
    all_m = _all_metrics()
    excluded = [{"airport": _label(c),
                 "reason": (f"{int(all_m.loc[c, 'enplanements_2025']):,} enplanements in 2025, "
                            f"below the {MIN_ENPLANEMENTS:,} ranking minimum")
                 if c in all_m.index else "no scheduled passenger service"}
                for c in codes if c not in eligible]
    if not eligible:
        return _error("below_ranking_threshold", "None of these airports can be scored.",
                      [e["airport"] + ": " + e["reason"] for e in excluded])

    peers = universe.loc[eligible]
    scored = score_airports(peers)
    ranks = sensitivity(peers) if len(eligible) > 1 else None
    rows = _score_rows(scored, peers, ranks)
    for row, code in zip(rows, scored.index):
        m = peers.loc[code]
        row["congestion_indicators"] = {
            "nas_delay_rate": _num(m["nas_delay_rate"]),
            "delay_rate": _num(m["delay_rate"]),
            "cancel_rate": _num(m["cancel_rate"]),
            "nas_share_of_delays": _num(m["nas_share_of_delays"]),
            "weather_share_of_delays": _num(m["weather_share_of_delays"]),
        }
        row["background"] = _context_for(code)
    warnings = _partial_warnings(scored) + ([] if ranks is None else _tie_warnings(ranks))
    if excluded:
        warnings.append("Excluded from comparison: "
                        + "; ".join(f"{e['airport']} ({e['reason']})" for e in excluded))
    return _response({"airports": rows, "excluded": excluded},
                     assumptions=[DELAY_NOTE], warnings=warnings)


def long_haul_share(airport: str, threshold_miles: float = LONG_HAUL_DEFAULT_MILES,
                    mode: str = "all") -> dict:
    """Share of departing flights at or above a distance threshold, passenger and/or cargo."""
    code, err = _check_airport(airport)
    if err:
        return err
    try:
        threshold = float(threshold_miles)
    except (TypeError, ValueError):
        threshold = None
    if threshold is None or not LONG_HAUL_MIN_MILES <= threshold <= LONG_HAUL_MAX_MILES:
        return _error("out_of_range", f"threshold_miles must be between {LONG_HAUL_MIN_MILES:,} "
                                      f"and {LONG_HAUL_MAX_MILES:,} miles (got {threshold_miles}).")
    mode = str(mode).strip().lower()
    if mode not in LONG_HAUL_MODES:
        return _error("out_of_range", f"mode must be one of {LONG_HAUL_MODES} (got '{mode}').")

    t = _t100()
    t = t[(t["ORIGIN"] == code) & (t["YEAR"] == LONG_HAUL_YEAR)]
    groups = {"passenger": PASSENGER_CLASSES, "cargo": CARGO_CLASSES}
    if mode != "all":
        groups = {mode: groups[mode]}

    result = {"airport": _label(code), "threshold_miles": threshold, "year": LONG_HAUL_YEAR}
    warnings = []
    for name, classes in groups.items():
        g = t[t["CLASS"].isin(classes)]
        flights = g["DEPARTURES_PERFORMED"].sum()
        if flights == 0:
            result[name] = {"flights": 0, "long_haul_share_of_flights": None}
            warnings.append(f"No {name} flights from {code} in {LONG_HAUL_YEAR}.")
            continue
        lh = g[g["DISTANCE"] >= threshold]
        top = (lh.groupby("DEST").agg(flights=("DEPARTURES_PERFORMED", "sum"),
                                      miles=("DISTANCE", "max"))
               .sort_values("flights", ascending=False).head(5))
        entry = {
            "flights": int(flights),
            "long_haul_flights": int(lh["DEPARTURES_PERFORMED"].sum()),
            "long_haul_share_of_flights": _num(lh["DEPARTURES_PERFORMED"].sum() / flights),
            "top_long_haul_destinations": [
                {"destination": d, "miles": int(r["miles"]), "flights": int(r["flights"])}
                for d, r in top.iterrows()],
        }
        if name == "passenger" and g["PASSENGERS"].sum() > 0:
            entry["long_haul_share_of_passengers"] = _num(lh["PASSENGERS"].sum()
                                                          / g["PASSENGERS"].sum())
        result[name] = entry

    assumptions = [
        f"Long haul = segment distance >= {threshold:,.0f} miles (no single industry definition).",
        "Share of departures performed from this airport in the full year 2025.",
        "Passenger = scheduled passenger (CLASS F); cargo = all-cargo (CLASS G scheduled + P "
        "non-scheduled); charters (CLASS L) excluded.",
    ]
    return _response(result, assumptions=assumptions, warnings=warnings,
                     data_period=f"T-100 {LONG_HAUL_YEAR} (departures in the full year)")


def demand_gap(airport: str) -> dict:
    """Proxies for unmet demand (not directly observable), each with its national percentile."""
    code, err = _check_airport(airport)
    if err:
        return err
    all_m = _all_metrics()
    if code not in all_m.index:
        return _error("no_passenger_service",
                      f"{_label(code)} has no scheduled passenger service in our data.")
    universe = _universe()
    m = all_m.loc[code]
    indicators = {}
    for col, meaning in [
        ("load_factor_2025", "share of seats filled (high = planes are full)"),
        ("nas_delay_rate", "share of arrivals delayed by airport/airspace capacity (NAS)"),
        ("delay_rate", "share of arrivals delayed 15+ minutes, all causes"),
        ("cancel_rate", "share of arrivals cancelled"),
        ("nas_share_of_delays", "share of delays caused by NAS"),
        ("weather_share_of_delays", "share of delays caused by extreme weather"),
        ("momentum_vs_national", "2026 YTD passenger growth minus national growth"),
    ]:
        value = m[col]
        pct = None if pd.isna(value) else (universe[col] <= value).mean() * 100
        indicators[col] = {"value": _num(value), "national_percentile": _num(pct, 0),
                           "national_median": _num(universe[col].median()), "meaning": meaning}

    warnings = []
    if pd.isna(m["nas_delay_rate"]):
        warnings.append(f"{code} has no delay data: congestion proxies are missing, not zero.")
    if m["enplanements_2025"] < MIN_ENPLANEMENTS:
        warnings.append(f"{code} is below the {MIN_ENPLANEMENTS:,} ranking minimum; percentiles "
                        "compare it with larger airports.")
    assumptions = [
        "Unmet demand cannot be observed directly; these are proxies. No passenger-count estimate "
        "is given because runway/gate capacity data is not available.",
        f"Percentiles are against the {len(universe)} US airports with at least "
        f"{MIN_ENPLANEMENTS:,} enplanements (100 = highest).",
        DELAY_NOTE,
    ]
    return _response({"airport": _label(code), "indicators": indicators,
                      "peer_airports": len(universe), "background": _context_for(code)},
                     assumptions=assumptions, warnings=warnings)


def live_airport_status(airport: str) -> dict:
    """Current FAA status for one US airport (live public API). Not used in the score."""
    code, err = _check_airport(airport)
    if err:
        return err
    try:
        root = live_api.fetch_nas_status()
        events = live_api.parse_airport_events(root)
        updated = live_api.update_time(root)
    except Exception as e:  # network, HTTP or XML error: report it, never return stale data
        return _error("live_api_unavailable",
                      f"The FAA NAS Status service could not be reached ({type(e).__name__}). "
                      "No live status is shown; historical metrics are unaffected.")
    active = events.get(code, [])
    result = {
        "airport": _label(code),
        "faa_update_time": updated,
        "active_events": active,
        "summary": (f"{len(active)} active FAA event(s)" if active
                    else "No active delays, ground stops or closures reported by the FAA"),
    }
    assumptions = [
        f"Live snapshot from the FAA NAS Status public API ({live_api.FAA_URL}) at {updated}; "
        "it describes this moment only and is not used in the score.",
        "A closure notice can be partial (e.g. closed only to non-scheduled general aviation); "
        "read its Reason text before calling the airport closed.",
    ]
    warnings = [f"{code}: closure notice - check whether it is partial: {e.get('Reason', '')[:120]}"
                for e in active if "Closure" in e["type"]]
    return {"result": result, "assumptions": assumptions, "warnings": warnings,
            "data_period": f"Live (FAA update time {updated})"}
