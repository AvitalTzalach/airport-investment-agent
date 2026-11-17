# Airport Investment Intelligence Agent

A chat assistant for analysts at a firm that invests in US airport modernization. You ask about
airports in plain language, and it tells you where demand is outgrowing capacity, which is where an
expansion is most likely to pay off.

The ranking is not left to the language model. Every airport gets an expansion-pressure score
(0-100) from a scoring model written in Python; the agent's job is to pick the right calculation and
explain the result, including what the data can't tell you.

## Example questions

| Question | What the agent finds |
|---|---|
| Which airports in New England are strong candidates for terminal expansion? | Boston (BOS) is the only airport under high pressure (71.6). Portland (PWM) and Burlington (BTV) come next at about 41, close enough to call it a tie. New Haven has no delay data, so its score is shown as a range. |
| Compare LA and Santa Ana airport congestion levels. | It reads "LA" as LAX and says so. LAX is a bit more congested (5.2% vs 4.4% of arrivals delayed by capacity), but Santa Ana's real limit is regulatory: a passenger cap and a night curfew. |
| What is the percentage of long haul flights out of Anchorage airport? | With long haul defined as 3,000+ miles: 2.8% of passenger departures in 2025, but 52.6% of cargo departures. Anchorage is mainly a cargo hub. |
| What is the unmet flight demand in SFO airport and why? | Unmet demand can't be measured directly, so it uses proxies: 18.8% of arrivals are delayed by capacity (the highest in the US) and planes are 82% full. The reason is the runways, which are only 750 ft apart and force single-file landings in fog. |

Follow-up questions work within the same conversation, e.g. "Why is PWM ranked second?".

## Running it locally

You need Python 3.12 and an Anthropic API key. The data is already in `data/`.

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env           # macOS/Linux: cp .env.example .env
# open .env and add your key:  AIRPORT_AGENT_API_KEY=sk-ant-...
streamlit run app.py
```

The app opens at http://localhost:8501. The first start takes about 10 seconds while the libraries
and data load.

To run the tests: `python -m pytest`. To check the metrics against known facts:
`python -m airport_agent.sanity_checks`. To try the four questions in the terminal:
`python -m airport_agent.agent`.

## How it works

```
question ──> agent (Claude Sonnet 5.5) ──> picks a tool ──> Python tool ──> numbers
                     │                                       (BTS data, score, FAA live API)
                     └────────── explains the result <─────────────────┘
```

The agent has six tools: rank a region, compare airports, an airport profile, long-haul share,
demand-gap indicators, and the live FAA status. All numbers come from these tools, and the tables and
charts in the app are drawn straight from their output rather than from the model's text. When data
is missing, the score is shown as a range instead of being filled in, and airports that are too close
to separate are reported as a tie.

The scoring method, the trade-offs and the limitations are in [DESIGN.md](DESIGN.md).

## Project structure

```
app.py                      the Streamlit chat app (entry point)
airport_agent/
    data_loader.py          loads the BTS and OurAirports files, builds per-airport metrics
    scoring_config.py       weights, ranges and pressure levels, all in one place
    scoring.py              the score, its breakdown by component, the sensitivity check
    tools.py                the agent's tools, including input validation
    live_api.py             calls the FAA NAS Status API
    agent.py                the agent: model, instructions, tools, conversation memory
    sanity_checks.py        checks the metrics against known facts
data/                       BTS and OurAirports data (.csv.gz) and airport_context.json
                            (a few background facts per airport, each with a source)
tests/                      unit tests
docs/sanity_checks.md       output of the sanity checks
```

## Data

[BTS](https://www.transtats.bts.gov) T-100 Segment, all carriers (2025 and January-August 2026),
BTS Airline On-Time Delay Causes (January 2025 - August 2026, the last 12 months are used),
[OurAirports](https://ourairports.com/data/) for airport names, and the FAA NAS Status API for live
status. The analysis covers the 142 US airports with at least 500,000 departing passengers in 2025.

## How I built it

I made the product and design decisions myself: which data to use, how to define and weight the
KPI, which libraries and model to use, and how the app should behave. I wrote the code with an AI
coding assistant (Claude Code), reviewed every change, and checked the results against facts I could
verify independently, for example that SFO comes out as the most capacity-constrained airport in the
US.
