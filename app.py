"""Streamlit chat UI for the airport investment agent. Run: streamlit run app.py

Tables and charts are drawn directly from the tool results (deterministic), never from the
LLM's text. The LLM's answer is shown as the explanation above them.
"""
import html
import os
import re

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from airport_agent import tools
from airport_agent.agent import API_KEY_ENV, build_agent, new_thread_id, stream_ask
from airport_agent.scoring_config import WEIGHTS

st.set_page_config(page_title="Airport Investment Intelligence", page_icon="✈️", layout="centered")

BLUE = "#0B6FB3"
EXAMPLE_QUESTIONS = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long haul flights out of Anchorage airport?",
    "What is the unmet flight demand in SFO airport and why?",
]
TOOL_LABELS = {
    "rank_airports": "Ranking airports...",
    "compare_airports": "Comparing airports...",
    "get_airport_profile": "Loading airport profile...",
    "long_haul_share": "Measuring long-haul flights...",
    "demand_gap": "Checking demand signals...",
    "live_airport_status": "Checking live FAA status...",
}
COMPONENT_COLORS = {"Load factor": BLUE, "Momentum vs national": "#5BB46E",
                    "Congestion": "#E58B2E", "Scale": "#7A8494"}
COMPONENT_NAMES = {"load_factor": "Load factor", "momentum": "Momentum vs national",
                   "congestion": "Congestion", "scale": "Scale"}
USER_AVATAR, AI_AVATAR = ":material/person:", ":material/auto_awesome:"
HEBREW = re.compile(r"[֐-׿]")
# Numbers inside Hebrew text: wrap each one in Unicode "isolate" marks so the browser keeps it as
# one left-to-right unit in the right-to-left sentence (e.g. "ב-BOS 81.7%" was shown as "81.7% BOS").
# Markdown link targets (URLs) are left untouched.
# A leading minus belongs to the number only if it does not follow a letter: in "ב-81.7%" the
# hyphen is the Hebrew prefix, not a minus sign.
# A number ends with a digit or %, so a sentence period/comma after it stays outside ("41.0.").
_NUMBER = re.compile(r"(?:(?<![֐-׿\w])[-+])?\d(?:[\d.,]*\d)?%?")
_LINK_TARGET = re.compile(r"(\]\([^)]*\))")


def isolate_numbers(text: str) -> str:
    parts = _LINK_TARGET.split(text)
    return "".join(p if _LINK_TARGET.fullmatch(p) else
                   _NUMBER.sub(lambda m: "\u2066" + m.group(0) + "\u2069", p) for p in parts)

# Line icons (Lucide, MIT license) for the HTML parts of the page
_ICON_PATHS = {
    "plane": '<path d="M2 22h20"/><path d="M6.36 17.4 4 17l-2-4 1.1-.55a2 2 0 0 1 1.8 0l.17.1a2 '
             '2 0 0 0 1.8 0L8 12 5 6l.9-.45a2 2 0 0 1 2.09.2l4.02 3a2 2 0 0 0 2.1.2l4.19-2.06a2.41 '
             '2.41 0 0 1 1.73-.17L21 7a1.4 1.4 0 0 1 .87 1.99l-.38.76c-.23.46-.6.84-1.07 1.08L7.58 '
             '17.2a2 2 0 0 1-1.22.18Z"/>',
    "sliders": '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line '
               'x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" '
               'x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/><line x1="14" '
               'x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="16" x2="16" '
               'y1="18" y2="22"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/>'
                '<path d="M3 12A9 3 0 0 0 21 12"/>',
    "checks": '<path d="m3 17 2 2 4-4"/><path d="m3 7 2 2 4-4"/><path d="M13 6h8"/>'
              '<path d="M13 12h8"/><path d="M13 18h8"/>',
    "followup": '<polyline points="15 10 20 15 15 20"/><path d="M4 4v7a4 4 0 0 0 4 4h12"/>',
}


def icon(name: str, size: int = 16) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>')


st.markdown(f"""
<style>
/* (the Inter font is set in .streamlit/config.toml) */
/* hide Streamlit's own chrome: Deploy, top color line, "running/Stop" widget (we show our own) */
[data-testid="stAppDeployButton"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{display: none !important;}}
header[data-testid="stHeader"] {{background: transparent;}}
.block-container {{max-width: 1000px; padding-top: 1rem; padding-bottom: 2rem;}}

/* top bar */
.st-key-topbar {{border-bottom: 1px solid rgba(128,128,128,0.18); padding-bottom: 0.6rem;
  margin-bottom: 0.6rem;}}
.topbar-title {{display: flex; align-items: center; gap: 0.55rem; font-weight: 700;
  font-size: 1.05rem;}}
.tile {{display: inline-flex; align-items: center; justify-content: center;
  background: rgba(11,111,179,0.12); color: {BLUE}; border-radius: 10px;}}
.st-key-topbar button {{border-radius: 10px; font-size: 0.88rem;}}

/* welcome */
.hero {{text-align: center; margin: 0.8rem 0 0.9rem;}}
.hero .tile {{width: 48px; height: 48px; border-radius: 14px;}}
.hero h1 {{font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em; margin: 0.7rem 0 0.25rem;
  padding: 0;}}
.hero p {{opacity: 0.7; font-size: 1rem; margin-bottom: 0.4rem;}}
.st-key-examples button {{min-height: 84px; width: 100%; border-radius: 14px;
  padding: 0.85rem 1.1rem; justify-content: space-between; align-items: flex-start;
  border: 1px solid rgba(128,128,128,0.25); transition: all .15s ease;}}
.st-key-examples button:hover {{border-color: {BLUE}; box-shadow: 0 4px 14px rgba(11,111,179,.12);
  transform: translateY(-1px);}}
.st-key-examples button p {{text-align: left; font-size: 0.97rem; line-height: 1.45;
  white-space: normal; overflow: visible; text-overflow: clip;}}
.st-key-examples [data-testid="stHorizontalBlock"] {{gap: 0.9rem;}}

/* chat avatars: soft rounded tiles instead of the default colored icons */
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] {{
  background: rgba(11,111,179,0.12) !important; color: {BLUE} !important;
  border-radius: 10px; width: 34px; height: 34px;}}
[data-testid="stChatMessage"] {{background: transparent !important; padding: 0.4rem 0;
  gap: 0.75rem;}}
/* user message: blue pill on the right */
[data-testid="stChatMessage"]:has([class*="st-key-umsg"]) {{flex-direction: row-reverse;}}
[data-testid="stChatMessage"]:has([class*="st-key-umsg"])
  [data-testid^="stChatMessageAvatar"] {{background: rgba(128,128,128,0.12) !important;
  color: inherit !important;}}
[data-testid="stChatMessage"]:has([class*="st-key-umsg"]) [data-testid="stChatMessageContent"] {{
  display: flex; justify-content: flex-end;}}
[class*="st-key-umsg"] {{background: {BLUE}; color: #fff; border-radius: 22px;
  padding: 0.65rem 1.2rem; width: fit-content !important; max-width: 80%; margin-left: auto;
  height: auto !important; overflow: visible;}}
[class*="st-key-umsg"] p {{color: #fff; margin: 0 !important; line-height: 1.5;}}
[class*="st-key-umsg"] [data-testid="stMarkdownContainer"] {{margin: 0 !important;}}
/* assistant message: clean card */
[class*="st-key-amsg"] {{border: 1px solid rgba(128,128,128,0.22); border-radius: 16px;
  padding: 1.1rem 1.25rem 0.6rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);}}
[class*="st-key-rtl"] {{direction: rtl; text-align: right;}}
/* answer text: comfortable reading measure, size, contrast and spacing */
[class*="st-key-ltr"] p, [class*="st-key-rtl"] p,
[class*="st-key-ltr"] li, [class*="st-key-rtl"] li {{
  max-width: 600px; font-size: 1.04rem; line-height: 1.8; color: #1F2328;}}
[class*="st-key-rtl"] p, [class*="st-key-rtl"] li, [class*="st-key-rtl"] td,
[class*="st-key-rtl"] th {{text-align: right !important;}}
[class*="st-key-ltr"] p, [class*="st-key-rtl"] p {{margin: 0 0 0.95rem;}}
[class*="st-key-ltr"] table, [class*="st-key-rtl"] table {{margin: 0.3rem 0 1.1rem;
  font-size: 0.95rem;}}

/* expanders as clean divided rows, not boxes */
[class*="st-key-amsg"] [data-testid="stExpander"] details {{border: none;
  border-top: 1px solid rgba(128,128,128,0.18); border-radius: 0;}}
[class*="st-key-amsg"] [data-testid="stExpander"] summary {{padding-left: 0.1rem;
  font-weight: 500; background: transparent !important;}}

/* thinking indicator */
.thinking {{display: flex; align-items: center; gap: 0.75rem; margin: 0.6rem 0 0.6rem 0;}}
.thinking .tile {{width: 34px; height: 34px;}}
.spinner {{width: 16px; height: 16px; border: 2px solid rgba(11,111,179,0.25);
  border-top-color: {BLUE}; border-radius: 50%; animation: spin .8s linear infinite;}}
@keyframes spin {{to {{transform: rotate(360deg);}}}}
.thinking .step {{opacity: 0.65;}}

/* badges, follow-ups */
.badge {{display: inline-block; background: #FDF0D5; color: #8A5A00; border: 1px solid #F0C36D;
  border-radius: 999px; padding: 0.15rem 0.75rem; font-size: 0.82rem; font-weight: 600;
  margin: 0 0.35rem 0.5rem 0; cursor: help;}}
.followup-label {{display: flex; align-items: center; gap: 0.4rem; opacity: 0.65;
  font-size: 0.88rem; margin: 0.9rem 0 0.4rem;}}
.st-key-followups button {{border-radius: 999px; font-size: 0.86rem;
  border: 1px solid rgba(128,128,128,0.3);}}
.st-key-followups button:hover {{border-color: {BLUE}; color: {BLUE};}}

/* chat input: always looks active (white, blue frame); the cursor is placed in it by a script */
[data-testid="stChatInput"] {{border-radius: 16px; box-shadow: 0 4px 18px rgba(11,111,179,0.10);
  border: 1.5px solid {BLUE} !important; background: #fff !important;}}
[data-testid="stChatInput"] > div, [data-testid="stChatInputTextArea"] {{
  background: #fff !important;}}
.st-key-focus-helper {{display: none;}}

/* sidebar */
.sb-title {{font-weight: 700; font-size: 1.05rem; margin: 0.2rem 0 1.1rem;}}
.sb-label {{display: flex; align-items: center; gap: 0.45rem; font-size: 0.74rem;
  font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.6;
  margin: 1.4rem 0 0.7rem;}}
.weight-row {{display: flex; justify-content: space-between; align-items: center;
  padding: 0.35rem 0; font-size: 0.93rem;}}
.dot {{display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  margin-right: 0.5rem;}}
.weight-bar {{height: 4px; border-radius: 2px; background: rgba(128,128,128,0.18);
  margin-bottom: 0.3rem;}}
.weight-bar span {{display: block; height: 4px; border-radius: 2px;}}
.source-card {{border: 1px solid rgba(128,128,128,0.22); border-radius: 12px;
  padding: 0.65rem 0.8rem; margin-bottom: 0.55rem;}}
.source-card .head {{display: flex; justify-content: space-between; align-items: center;
  gap: 0.4rem; font-weight: 600; font-size: 0.93rem;}}
.pill {{font-family: ui-monospace, monospace; font-size: 0.7rem; font-weight: 500;
  background: rgba(128,128,128,0.13); border-radius: 6px; padding: 0.1rem 0.4rem;
  white-space: nowrap;}}
.pill.live {{background: rgba(11,111,179,0.12); color: {BLUE}; font-weight: 700;}}
.source-card small {{opacity: 0.65;}}
.assumptions {{padding-left: 1.1rem; font-size: 0.88rem; opacity: 0.85;}}
.assumptions li {{margin-bottom: 0.35rem;}}
</style>
""", unsafe_allow_html=True)


# ---------- Setup ----------

def load_cloud_api_key():
    """On Streamlit Community Cloud the API key is an app secret; locally it comes from .env."""
    try:
        key = st.secrets.get(API_KEY_ENV)
    except Exception:  # no secrets configured (local run): keep using .env
        key = None
    if key:
        os.environ.setdefault(API_KEY_ENV, key)


load_cloud_api_key()


@st.cache_resource(show_spinner="Loading data and agent...")
def get_agent():
    tools.rank_airports(region="US", top_n=1)   # load and cache the data once
    return build_agent()


if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()
    st.session_state.history = []
    st.session_state.pending = None


def start_new_conversation():
    st.session_state.thread_id = new_thread_id()
    st.session_state.history = []
    st.session_state.pending = None


def ask_question(question: str):
    st.session_state.pending = question


# ---------- Helpers that turn tool results into tables and charts ----------

def code_of(label: str) -> str:
    return label.split(" - ")[0]


def scored_rows(results: list) -> list:
    """Airports with a score from rank / compare / profile results."""
    rows = []
    for r in results:
        res = r["data"].get("result") or {}
        if r["name"] in ("rank_airports", "compare_airports"):
            rows += res.get("airports", [])
        elif r["name"] == "get_airport_profile" and res.get("scored"):
            rows.append(res)
    return rows


def short_name(label: str) -> str:
    """'BOS - Boston Logan International Airport (Boston)' -> 'BOS · Boston'."""
    code = code_of(label)
    city = label.rsplit("(", 1)[-1].rstrip(")") if "(" in label else ""
    return f"{code} · {city}" if city else code


def pct(value, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:+.1f} pts" if signed else f"{value * 100:.1f}%"


def rank_text(row: dict) -> str:
    best, worst = row.get("rank_range_under_weight_changes", [None, None])
    if best is None:
        return ""
    return str(best) if best == worst else f"{best}-{worst} (tie)"


def score_table(rows: list) -> pd.DataFrame:
    table = pd.DataFrame([{
        "Rank": rank_text(r),
        "Airport": short_name(r["airport"]),
        "Score": f"{r['score']:.1f}" + (f"  ({r['score_range'][0]:.0f}-{r['score_range'][1]:.0f})"
                                         if r.get("partial_data") else ""),
        "Pressure": r["pressure_level"].replace(" expansion pressure", ""),
        "Planes full": pct(r["metrics"]["load_factor_2025"]),
        "Capacity delays": pct(r["metrics"]["nas_delay_rate"]),
        "Passengers 2025": f"{r['metrics']['enplanements_2025'] / 1e6:.1f}M",
    } for r in rows])
    return table if table["Rank"].any() else table.drop(columns="Rank")


DEMAND_LABELS = {
    "load_factor_2025": "Planes full (load factor)",
    "nas_delay_rate": "Arrivals delayed by airport capacity",
    "delay_rate": "Arrivals delayed 15+ min (all causes)",
    "cancel_rate": "Arrivals cancelled",
    "nas_share_of_delays": "Share of delays caused by capacity",
    "weather_share_of_delays": "Share of delays caused by extreme weather",
    "momentum_vs_national": "Passenger growth vs national trend",
}


def breakdown_chart(rows: list):
    data = pd.DataFrame([{"Airport": code_of(r["airport"]), "Component": COMPONENT_NAMES[k],
                          "Points": v or 0, "order": i}
                         for r in rows for i, (k, v) in enumerate(r["points"].items())])
    order = [code_of(r["airport"]) for r in rows]
    return alt.Chart(data).mark_bar(cornerRadius=3).encode(
        x=alt.X("sum(Points):Q", title="Score 0-100", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Airport:N", sort=order, title=None),
        color=alt.Color("Component:N", title=None,
                        scale=alt.Scale(domain=list(COMPONENT_COLORS),
                                        range=list(COMPONENT_COLORS.values())),
                        legend=alt.Legend(orient="top")),
        order=alt.Order("order:Q"),
        tooltip=["Airport", "Component", alt.Tooltip("Points:Q", format=".1f")],
    ).properties(height=alt.Step(34))   # fixed height per airport row


def detail_tables(results: list):
    for r in results:
        data, res = r["data"], r["data"].get("result") or {}
        if "error" in data:
            st.warning(f"{r['name']}: {data['message']}")
        elif r["name"] == "long_haul_share":
            miles = f"{res['threshold_miles']:,.0f}"
            st.caption(f"{short_name(res['airport'])} · departures in {res['year']} · "
                       f"long haul = {miles}+ miles")
            modes = {m: v for m, v in res.items() if m in ("passenger", "cargo")}
            st.dataframe(pd.DataFrame([{
                "Flights": m.capitalize(), "Departures": f"{v.get('flights', 0):,}",
                f"Long haul ({miles}+ mi)": f"{v.get('long_haul_flights', 0):,}",
                "Share": pct(v.get("long_haul_share_of_flights"))}
                for m, v in modes.items()]), hide_index=True)
            for m, v in modes.items():
                top = v.get("top_long_haul_destinations", [])
                if top:
                    st.caption(f"Top {m} long-haul destinations: " + ", ".join(
                        f"{d['destination']} ({d['flights']:,})" for d in top))
        elif r["name"] == "demand_gap":
            st.caption(f"{short_name(res['airport'])} · compared with "
                       f"{res['peer_airports']} US airports")
            st.dataframe(pd.DataFrame([{
                "Signal": DEMAND_LABELS.get(k, k),
                "This airport": pct(v["value"], signed=(k == "momentum_vs_national")),
                "US median": pct(v["national_median"], signed=(k == "momentum_vs_national")),
                "Higher than": (f"{v['national_percentile']:.0f}% of airports"
                                if v["national_percentile"] is not None else "n/a")}
                for k, v in res["indicators"].items()]), hide_index=True)
        elif r["name"] == "live_airport_status":
            st.caption(f"{short_name(res['airport'])} · FAA update {res['faa_update_time']}")
            st.write(res["summary"])
            if res["active_events"]:
                st.dataframe(pd.DataFrame([{"Event": e.get("type", ""),
                                            "Reason": e.get("Reason", "")[:90]}
                                           for e in res["active_events"]]), hide_index=True)
        elif r["name"] == "get_airport_profile" and not res.get("scored"):
            st.caption(short_name(res.get("airport", "")))
            st.write(res.get("note", ""))
    rows = scored_rows(results)
    if rows:
        st.dataframe(score_table(rows), hide_index=True)


MISSING_LABELS = {"congestion": "No delay data", "load_factor": "No seat data",
                  "momentum": "No 2026 growth data", "scale": "No passenger data"}


def partial_badges(results: list) -> str:
    """One badge per airport with a missing score component, e.g. 'HVN · No delay data · ...'."""
    badges = []
    for r in scored_rows(results):
        if not r.get("partial_data"):
            continue
        missing = r.get("missing_components", "")
        label = MISSING_LABELS.get(missing, "Missing data")
        low, high = r["score_range"]
        tip = (f"{label.replace('No', 'Missing')} for this airport, so its score is shown as a "
               f"range ({low:.0f}-{high:.0f}) instead of a single number.")
        badges.append(f'<span class="badge" title="{html.escape(tip)}">{code_of(r["airport"])} · '
                      f'{label} · score could be {low:.0f}-{high:.0f}</span>')
    return "".join(badges)


def follow_ups(results: list) -> list:
    """Deterministic follow-up suggestions based on the last tool used."""
    if not results or "error" in results[-1]["data"]:
        return []
    last = results[-1]
    res = last["data"].get("result") or {}
    if last["name"] == "rank_airports" and len(res.get("airports", [])) >= 2:
        a, b = (code_of(x["airport"]) for x in res["airports"][:2])
        return [f"Why is {b} ranked second?", f"Compare {a} and {b}",
                f"What is the demand gap at {a}?"]
    if last["name"] == "compare_airports" and res.get("airports"):
        a = code_of(res["airports"][0]["airport"])
        return [f"Show the full profile of {a}", f"What is the live FAA status at {a}?"]
    code = code_of(res["airport"]) if res.get("airport") else None
    if not code:
        return []
    return {
        "long_haul_share": [f"Use a 2,500-mile threshold for {code}",
                            f"What is the demand gap at {code}?"],
        "demand_gap": [f"Show the full profile of {code}",
                       f"What is the live FAA status at {code}?"],
        "get_airport_profile": [f"What is the demand gap at {code}?",
                                f"What is the live FAA status at {code}?"],
        "live_airport_status": [f"Show the full profile of {code}"],
    }.get(last["name"], [])


def render_user(text: str, index: int):
    with st.chat_message("user", avatar=USER_AVATAR):
        with st.container(key=f"umsg-{index}"):
            st.markdown(text)


def render_assistant(msg: dict, index: int, is_last: bool):
    results = msg.get("tool_results", [])
    with st.chat_message("assistant", avatar=AI_AVATAR):
        with st.container(key=f"amsg-{index}"):
            rtl = "rtl" if HEBREW.search(msg["content"]) else "ltr"
            with st.container(key=f"{rtl}-{index}"):
                st.markdown(isolate_numbers(msg["content"]) if rtl == "rtl" else msg["content"])
            badges = partial_badges(results)
            if badges:
                st.markdown(badges, unsafe_allow_html=True)
            if results:
                rows = scored_rows(results)
                with st.expander("Details", icon=":material/table_chart:"):
                    detail_tables(results)
                if rows:
                    with st.expander("Score breakdown", icon=":material/bar_chart:"):
                        st.altair_chart(breakdown_chart(rows), width="stretch")
                        st.caption("Points per component; they add up to the score. Weights: "
                                   + ", ".join(f"{COMPONENT_NAMES[k]} {v:.0%}"
                                               for k, v in WEIGHTS.items()))
                # Only what is specific to this answer; general assumptions are in the sidebar
                specific = []
                for r in results:
                    specific += r["data"].get("warnings", [])
                    specific += [a for a in r["data"].get("assumptions", [])
                                 if a not in tools.BASE_ASSUMPTIONS]
                with st.expander("Assumptions & uncertainty", icon=":material/report:"):
                    if specific:
                        st.markdown("\n".join(f"- {n}" for n in dict.fromkeys(specific)))
                    for p in dict.fromkeys(r["data"].get("data_period") for r in results
                                           if r["data"].get("data_period")):
                        st.caption(f"Data: {p}")
        if is_last:
            suggestions = follow_ups(results)
            if suggestions:
                st.markdown(f'<div class="followup-label">{icon("followup", 15)} '
                            f'Suggested follow-ups</div>', unsafe_allow_html=True)
                with st.container(key="followups", horizontal=True):
                    for i, s in enumerate(suggestions):
                        st.button(s, key=f"fu-{index}-{i}", on_click=ask_question, args=(s,))


def thinking_html(step: str) -> str:
    return (f'<div class="thinking"><span class="tile"><span class="spinner"></span></span>'
            f'<b>Analyzing...</b><span class="step">{html.escape(step)}</span></div>')


# ---------- Sidebar ----------

with st.sidebar:
    st.markdown('<div class="sb-title">Analysis settings</div>', unsafe_allow_html=True)

    weights_html = f'<div class="sb-label">{icon("sliders", 14)} Scoring weights</div>'
    for k, v in WEIGHTS.items():
        color = COMPONENT_COLORS[COMPONENT_NAMES[k]]
        weights_html += (f'<div class="weight-row"><span><span class="dot" '
                         f'style="background:{color}"></span>{COMPONENT_NAMES[k]}</span>'
                         f'<b>{v:.0%}</b></div><div class="weight-bar"><span '
                         f'style="width:{v * 100 / 0.4:.0f}%;background:{color}"></span></div>')
    weights_html += '<small style="opacity:.6">Fixed methodology (see design document).</small>'
    st.markdown(weights_html, unsafe_allow_html=True)

    sources_html = f'<div class="sb-label">{icon("database", 14)} Data sources</div>'
    for name, period, desc, live in [
        ("BTS T-100", "2025 · Jan-Aug 2026", "Passengers, seats and flights by route", False),
        ("BTS Delay Causes", "Sep 2025-Aug 2026", "Why flights were late (incl. NAS)", False),
        ("OurAirports", "reference", "Airport names and cities", False),
        ("FAA airport status", "Live", "Current ground stops and delays", True),
    ]:
        sources_html += (f'<div class="source-card"><div class="head">{name}'
                         f'<span class="pill{" live" if live else ""}">{period}</span></div>'
                         f'<small>{desc}</small></div>')
    st.markdown(sources_html, unsafe_allow_html=True)

    st.markdown(f'<div class="sb-label">{icon("checks", 14)} Key assumptions</div>'
                '<ul class="assumptions">'
                '<li>US airports with 500,000+ enplanements are scored.</li>'
                '<li>Passenger volume = enplanements (FAA definition).</li>'
                '<li>Growth is measured relative to the national trend.</li>'
                '<li>Missing data is shown as a score range, never as zero.</li>'
                '<li>The score is expansion pressure, not a profit forecast.</li></ul>',
                unsafe_allow_html=True)


# ---------- Top bar ----------

with st.container(key="topbar"):
    left, right = st.columns([4, 1.4], vertical_alignment="center")
    left.markdown(f'<div class="topbar-title"><span class="tile" style="width:30px;height:30px">'
                  f'{icon("plane", 17)}</span>Airport Investment Intelligence</div>',
                  unsafe_allow_html=True)
    right.button("New conversation", icon=":material/add:", on_click=start_new_conversation,
                 width="stretch")

agent = get_agent()

# Read the input box BEFORE drawing the page, so a typed question hides the welcome screen at once
# (the input box is always pinned to the bottom, wherever it is called).
typed = st.chat_input("Ask about airports, regions or scores...")
if typed:
    st.session_state.pending = typed

# Put the cursor in the input box on every page load / after every answer (no built-in option)
with st.container(key="focus-helper"):
    # the message count makes the script text change after each answer, so it runs again
    components.html(f"<!-- {len(st.session_state.history)} -->" + """<script>
      const doc = window.parent.document;
      const focusBox = () => {
        const box = doc.querySelector('[data-testid="stChatInputTextArea"]');
        if (box && doc.activeElement !== box) box.focus();
      };
      setTimeout(focusBox, 300); setTimeout(focusBox, 1200);
    </script>""", height=0)

# ---------- Welcome screen ----------

if not st.session_state.history and not st.session_state.pending:
    st.markdown(f'<div class="hero"><span class="tile">{icon("plane", 24)}</span>'
                '<h1>Airport Investment Intelligence</h1><p>Find US airports where terminal '
                'expansion pays off, with a transparent 0-100 score.</p></div>',
                unsafe_allow_html=True)
    with st.container(key="examples"):
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            cols[i % 2].button(q, key=f"ex-{i}", on_click=ask_question, args=(q,),
                               icon=":material/north_east:", icon_position="right",
                               width="stretch", wrap=True)

# ---------- Conversation ----------

for i, msg in enumerate(st.session_state.history):
    if msg["role"] == "user":
        render_user(msg["content"], i)
    else:
        render_assistant(msg, i, is_last=(i == len(st.session_state.history) - 1
                                          and not st.session_state.pending))

if st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None
    st.session_state.history.append({"role": "user", "content": question})
    render_user(question, len(st.session_state.history) - 1)
    indicator = st.empty()
    indicator.markdown(thinking_html("Understanding the question..."), unsafe_allow_html=True)
    try:
        final = None
        for event in stream_ask(agent, st.session_state.thread_id, question):
            if event["type"] == "tool":
                indicator.markdown(thinking_html(TOOL_LABELS.get(event["name"], "Working...")),
                                   unsafe_allow_html=True)
            else:
                final = event
    except Exception as e:  # API or network problem: tell the user, keep the app alive
        final = {"answer": f"Sorry, the agent could not answer ({type(e).__name__}). "
                           "Please try again.", "tool_calls": [], "tool_results": []}
    indicator.empty()
    st.session_state.history.append({"role": "assistant", "content": final["answer"],
                                     "tool_calls": final.get("tool_calls", []),
                                     "tool_results": final.get("tool_results", [])})
    st.rerun()
