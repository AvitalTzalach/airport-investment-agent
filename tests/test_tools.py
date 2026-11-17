"""Tests for tools.py on the real BTS data in data/ (skipped if the data files are missing).
Run from the project root: python -m pytest"""
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent / "data"
pytestmark = pytest.mark.skipif(not (DATA_DIR / "t100_2025.csv.gz").exists(),
                                reason="data files not found in data/")

from airport_agent.tools import (compare_airports, demand_gap, get_airport_profile,  # noqa: E402
                   long_haul_share, rank_airports)

SUCCESS_KEYS = {"result", "assumptions", "warnings", "data_period"}


def is_error(r, code):
    return r.get("error") == code and r.get("message")


# ---------- Input validation ----------

def test_unknown_airport_returns_error_with_suggestions():
    r = get_airport_profile("SFX")
    assert is_error(r, "unknown_airport")
    assert r["suggestions"][0].startswith("SFO")


def test_ambiguous_city_lists_metro_airports():
    r = get_airport_profile("LA")
    assert is_error(r, "unknown_airport")
    assert [s[:3] for s in r["suggestions"]][:4] == ["LAX", "BUR", "LGB", "ONT"]


def test_city_name_suggests_the_airport():
    r = get_airport_profile("Santa Ana")
    assert r["suggestions"][0].startswith("SNA")


def test_codes_are_case_insensitive():
    assert set(long_haul_share("anc")) == SUCCESS_KEYS


@pytest.mark.parametrize("airports", [["LAX"], ["LAX", "LAX"], [],
                                      ["LAX", "SNA", "SFO", "BOS", "JFK", "ORD"]])
def test_compare_count_limits(airports):
    assert is_error(compare_airports(airports), "invalid_count")


@pytest.mark.parametrize("threshold", [499, 8001, "far"])
def test_long_haul_threshold_limits(threshold):
    assert is_error(long_haul_share("ANC", threshold), "out_of_range")


def test_long_haul_threshold_bounds_are_inclusive():
    assert "result" in long_haul_share("ANC", 500)
    assert "result" in long_haul_share("ANC", 8000)


def test_long_haul_mode_must_be_known():
    assert is_error(long_haul_share("ANC", mode="freight"), "out_of_range")


def test_unknown_region_and_states():
    assert is_error(rank_airports(region="Narnia"), "unknown_region")
    assert is_error(rank_airports(states=["CA", "ZZ"]), "unknown_region")
    assert is_error(rank_airports(), "unknown_region")


def test_top_n_limits():
    assert is_error(rank_airports(region="US", top_n=0), "out_of_range")
    assert is_error(rank_airports(region="US", top_n=51), "out_of_range")


# ---------- Small airports (below the ranking minimum) ----------

def test_profile_of_small_airport_returns_raw_metrics_without_score():
    r = get_airport_profile("BGR")
    assert r["result"]["scored"] is False
    assert "metrics" in r["result"] and "below" in r["result"]["note"]


def test_compare_excludes_small_airport_and_compares_the_rest():
    r = compare_airports(["BOS", "BGR", "PWM"])
    assert [a["airport"][:3] for a in r["result"]["airports"]] == ["BOS", "PWM"]
    assert r["result"]["excluded"][0]["airport"].startswith("BGR")


# ---------- The four example questions ----------

def test_q1_new_england_ranking():
    r = rank_airports(region="New England")
    airports = r["result"]["airports"]
    assert airports[0]["airport"].startswith("BOS")
    assert airports[-1]["airport"].startswith("MHT")
    hvn = next(a for a in airports if a["airport"].startswith("HVN"))
    assert hvn["partial_data"] and hvn["pressure_level"] != "High expansion pressure"
    assert hvn["metrics"]["nas_delay_rate"] is None        # missing, never zero
    assert any("tie" in w for w in r["warnings"])


def test_q2_compare_lax_sna():
    r = compare_airports(["LAX", "SNA"])
    codes = {a["airport"][:3] for a in r["result"]["airports"]}
    assert codes == {"LAX", "SNA"}
    assert all("congestion_indicators" in a for a in r["result"]["airports"])


def test_q3_anchorage_long_haul_passenger_vs_cargo():
    r = long_haul_share("ANC")["result"]
    pax, cargo = r["passenger"], r["cargo"]
    assert 0 <= pax["long_haul_share_of_flights"] < cargo["long_haul_share_of_flights"] <= 1
    assert all(d["miles"] >= 3000 for d in pax["top_long_haul_destinations"])


def test_q4_sfo_demand_gap_proxies():
    r = demand_gap("SFO")
    nas = r["result"]["indicators"]["nas_delay_rate"]
    assert nas["national_percentile"] >= 99            # SFO has the highest NAS delay rate
    assert any("proxies" in a for a in r["assumptions"])


def test_every_success_response_has_the_same_shape():
    for r in [get_airport_profile("BOS"), rank_airports(region="Pacific"),
              compare_airports(["LAX", "SNA"]), long_haul_share("ANC"), demand_gap("SFO")]:
        assert set(r) == SUCCESS_KEYS
        assert r["assumptions"] and r["data_period"]


# ---------- Live FAA status (offline: sample XML and a simulated outage) ----------

SAMPLE_FAA_XML = """<AIRPORT_STATUS_INFORMATION><Update_Time>Sat Nov 14 21:18:47 2026 GMT</Update_Time>
<Delay_type><Name>Ground Stop Programs</Name><Ground_Stop_List><Program><ARPT>LGA</ARPT>
<Reason>other</Reason><End_Time>5:30 pm EST</End_Time></Program></Ground_Stop_List></Delay_type>
<Delay_type><Name>Airspace Flow Programs</Name><Airspace_Flow_List><Airspace_Flow>
<CTL_Element>FCAHU7</CTL_Element><Reason>other</Reason></Airspace_Flow></Airspace_Flow_List></Delay_type>
<Delay_type><Name>Airport Closures</Name><Airport_Closure_List><Airport><ARPT>LAX</ARPT>
<Reason>!LAX AD AP CLSD TO NON SKED TRANSIENT GA ACFT</Reason></Airport></Airport_Closure_List></Delay_type>
</AIRPORT_STATUS_INFORMATION>"""


@pytest.fixture
def fake_faa(monkeypatch):
    import xml.etree.ElementTree as ET
    from airport_agent import live_api
    monkeypatch.setattr(live_api, "fetch_nas_status", lambda: ET.fromstring(SAMPLE_FAA_XML))


def test_live_status_reports_active_event(fake_faa):
    from airport_agent.tools import live_airport_status
    r = live_airport_status("LGA")["result"]
    assert r["active_events"] == [{"type": "Ground Stop Programs", "Reason": "other",
                                   "End_Time": "5:30 pm EST"}]


def test_live_status_flags_closure_notice_as_possibly_partial(fake_faa):
    from airport_agent.tools import live_airport_status
    r = live_airport_status("LAX")
    assert r["result"]["active_events"][0]["type"] == "Airport Closures"
    assert any("partial" in w for w in r["warnings"])


def test_live_status_quiet_airport_and_airspace_programs_skipped(fake_faa):
    from airport_agent.tools import live_airport_status
    assert live_airport_status("BOS")["result"]["active_events"] == []


def test_live_status_outage_returns_error_not_stale_data(monkeypatch):
    from airport_agent import live_api
    from airport_agent.tools import live_airport_status

    def down():
        raise TimeoutError("simulated outage")
    monkeypatch.setattr(live_api, "fetch_nas_status", down)
    assert is_error(live_airport_status("SFO"), "live_api_unavailable")


def test_live_status_validates_airport():
    from airport_agent.tools import live_airport_status
    assert is_error(live_airport_status("XYZ1"), "unknown_airport")
