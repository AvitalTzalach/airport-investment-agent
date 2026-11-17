# Design notes

## The question behind the score

An investor in a terminal earns more with every additional passenger, so the interesting airports
are the ones where demand is strong and the airport is already running out of room. I turned that
into an expansion-pressure score from 0 to 100: demand against capacity.

It is a screening tool, not a profit forecast. I don't have construction costs, deal terms or gate
counts, so the score can say where pressure is high, but not whether a specific project will make
money.

## Data

| Source | What I use it for |
|---|---|
| BTS T-100 Segment, all carriers (2025, Jan-Aug 2026) | Passengers, seats, departures, distances |
| BTS Airline On-Time Delay Causes (last 12 months: Sep 2025 - Aug 2026) | Delays and their causes |
| OurAirports | Airport names and cities |
| FAA NAS Status API | Live ground stops, delay programs and closures |
| `data/airport_context.json` | 13 hand-checked background facts with sources, e.g. Santa Ana's passenger cap |

A few decisions here shape everything else. I count only scheduled passenger flights for passenger
metrics, and measure volume as departing passengers (enplanements), which is how the FAA reports it.
T-100 also lists foreign airports with flights to the US, so I keep only US airports (the 50 states,
DC, Puerto Rico and the Virgin Islands) and score only those with at least 500,000 enplanements, which
leaves 142. Missing values are never treated as zero. For delays I use a full 12 months: an earlier
version used only August and overstated them (26% of flights delayed in Aug 2026, against 22% over
the year).

## How the score works

Each metric is mapped onto a 0-1 scale using a fixed range (anything outside it is clipped), and then
weighted:

| Component | What it asks | Metric | Range (0 → 1) | Weight |
|---|---|---|---|---|
| Load factor | Are the planes full? | Passengers / seats, 2025 | 70% → 90% | 25% |
| Momentum | Is demand growing faster than the country? | Jan-Aug 2026 growth minus national growth | -8% → +8% | 25% |
| Congestion | Is the airport hitting its limit? | Share of arrivals delayed by airport or airspace capacity (BTS "NAS" delays) | 2% → 12% | 35% |
| Scale | How big is it? | log10 of 2025 enplanements | 500k → 50M | 15% |

Congestion gets the most weight because it is the clearest sign that an airport has hit its capacity.
Scale gets the least, so that the largest airport doesn't win on size alone. I set the ranges by
looking at how each metric is spread across the 142 airports (`docs/sanity_checks.md`). The points of
the four components always add up to the score, so any score can be taken apart and explained.

Scores of 60 and above count as high pressure, 40-60 as medium, below 40 as low.

When an airport is missing a component, I re-weight the other three, mark it as partial data, show
the range the score could fall in (missing component at 0 and at 1), and never label it high. I also
wanted to know how much the ranking depends on my weights, so every weight is moved up and down by
50% and the resulting rank range is shown. Airports whose ranges overlap are reported as a tie rather
than as first and second.

To check that the numbers make sense, I compared them with things I know independently. SFO comes out
with the highest capacity-delay rate in the country; Boston, LaGuardia, Newark, Reagan National and
O'Hare are all in the top 10; and Santa Ana sits below the median, which fits a limit that is
regulatory rather than physical. All seven checks pass, along with 53 unit tests.

## Architecture and the role of the LLM

It is a single agent (LangChain `create_agent`, running on LangGraph) using Claude Sonnet 5.5 with six
tools, a Streamlit chat app, and in-memory conversation history per session.

I use the model for the parts it is good at: understanding the question, including ambiguous ones
like "LA", choosing the right tool, and explaining the result in plain language, in whichever
language the user writes in. I don't use it for numbers. The score, ranks, shares and percentiles are
all computed in Python, the instructions forbid the model from estimating or inventing figures, and
the app draws its tables and charts from the tool output rather than from the model's text.

Each tool checks its own input: an unknown airport code returns close suggestions, and out-of-range
parameters return a clear error the agent can explain. Before a message reaches the agent, empty or
very long messages are stopped, and the instructions keep the agent to US airport questions. If
someone describes an emergency, it only points them to emergency services.

## Trade-offs

**One agent rather than several.** Six focused tools cover every question in scope. Splitting the
work across agents would add cost, latency and more things that can fail, without a clear gain.

**Fixed ranges rather than ranking against the peer group.** With fixed ranges an airport gets the
same score whether it is compared with its region or the whole country. The price is that I had to
choose the ranges, which I did from the data.

**Historical files plus a live API.** Twelve months of BTS data drive the score. The live FAA feed
only describes the current moment, so I deliberately keep it out of the score.

**Capacity delays rather than all delays.** Total delays include airline and late-aircraft causes that
have nothing to do with the airport's capacity.

**Growth compared with the national trend.** Domestic traffic fell about 1.1% in Jan-Aug 2026, so raw
growth would penalize every airport for something happening across the market.

**No 2019 baseline.** I started with a long-term growth component against 2019 and later removed it.
A pre-COVID base says less about today's investment case, and it needed a special rule for airports
that started from almost nothing (New Haven grew about 15x after Avelo arrived). The cost is that
long-term trends are no longer part of the score.

**Simple deterministic guardrails rather than an AI classifier.** They are cheap, predictable and easy
to test. For a production version I would add a small classifier model in front of the agent.

## Limitations and next steps

The score can't tell a runway problem from a gate or terminal problem. SFO is the clearest example:
its constraint is the runways, which a new terminal wouldn't solve. Capacity data such as gate counts
and runway rates would close that gap. Unmet demand is estimated from proxies rather than measured,
and the delay data covers only the large US carriers that report it, so a few airports (New Haven,
for one) have none.

If I continued, I would add an input classifier, persistent memory with summarization for long
conversations, voice input feeding the same chat, and optional weight sliders so analysts can test
their own assumptions.
