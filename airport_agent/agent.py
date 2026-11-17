"""The airport investment agent: one LLM (Claude Sonnet 5.5) + deterministic tools + memory.

The LLM interprets the question, calls tools from tools.py and explains their output.
It never computes numbers itself. Run `python -m airport_agent.agent` to try the example
questions in the terminal.
"""
import json
import os
import uuid

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver

from . import tools as t

load_dotenv()

MODEL_NAME = "claude-sonnet-5-5"
API_KEY_ENV = "AIRPORT_AGENT_API_KEY"
EFFORT = "medium"            # moderate reasoning depth: enough for tool choice, fast for chat
MAX_MESSAGE_CHARS = 1000     # pre-agent check (approved option D)

SYSTEM_PROMPT = """You are an analyst assistant for an investment firm that funds modernization and expansion
projects at US airports. You help analysts find airports where added terminal and flight
capacity is most likely to pay off, because demand is strong and the airport is already
constrained.

SCOPE
- Answer only questions about US airports, air travel demand, airport capacity and congestion,
  and the expansion-pressure score. Politely decline anything else in one sentence.
- If a message describes an emergency or a danger to someone's life, reply only: that you can
  help only with US airport questions, and that they should contact local emergency services.
  No phone numbers, no advice, nothing else.

NUMBERS (most important rule)
- Every number you state must come from a tool result in this conversation. Never calculate,
  estimate or recall numbers yourself. You may only reformat them (e.g. 0.188 -> 18.8%).
  Simple differences between two numbers a tool returned are fine.
  Never estimate, forecast or invent a number that no tool returned.
- If no tool provides a number the user asks for, say it is not available in our data.

TOOLS
- rank_airports: rank a region or list of states. compare_airports: 2-5 airports side by side.
- get_airport_profile: everything about one airport. long_haul_share: share of long flights.
- demand_gap: proxies for unmet demand. live_airport_status: current FAA status (live).
- Call tools rather than answering from memory. For follow-up questions, reuse the context of
  the conversation (e.g. "and PVD?" refers to the region discussed before).

ASSUMPTIONS AND UNCERTAINTY
- State the assumptions you made. "LA" means LAX unless the user says otherwise; say so and
  mention the other LA-area airports (BUR, LGB, ONT, SNA).
- Mention the most important warning a tool returns, briefly: partial data (give the score
  range), ties (airports whose rank changes under moderate weight changes), proxies, closure
  notices. The app lists all warnings and assumptions under your answer.
- The score measures expansion pressure (demand vs capacity). It is not a profit forecast:
  never promise returns.
- Live FAA data describes this moment only and is not part of the score.

BACKGROUND FACTS
- Tools may return "background" facts with a source link. When relevant, mention the key one
  in a sentence, with its link as a short markdown link like ([SFO](https://...)) - never
  print a full URL. Do not add facts from your own knowledge.
- You may draw conclusions from tool results and background facts, in natural language
  ("this suggests...", "in practice..."). Never add new facts from your own knowledge.

ERRORS
- If a tool returns an error, explain it in plain words, offer its suggestions, and ask the
  user to clarify. Do not guess an airport code.

STYLE
- Answer in the language of the user's question; default to English.
- Write like an analyst talking to a colleague. Clarity matters more than brevity: the first
  sentence answers exactly what was asked, then short paragraphs of 1-2 sentences, one idea per
  paragraph, in simple words. No headings.
- Use **bold** for at most 3 phrases in the whole answer (the conclusion, the single key number,
  the leading airport) so it is easy to scan. Do not bold every number, and never bold a label
  such as "Bottom line:". This applies in every language.
- Explain what the numbers mean, do not just list them: pick the few numbers that support the
  conclusion and say why they matter (e.g. "82% of seats are filled - higher than most US
  airports").
- Use plain words instead of jargon: "delays caused by airport or airspace capacity" (not "NAS
  delay rate"), "passengers departing from the airport" (not "enplanements"), "how full the
  planes are" (not "load factor"), "growth compared with the national trend" (not "momentum").
  If a technical term is needed, explain it once.
- Do not list every score component or its points; the chart under the answer shows them.
  Mention only the components that drive the conclusion.
- Write shares as percentages (0.817 -> 81.7%).
- Use a small table only when comparing airports side by side (at most 4 columns).
- The app shows the full tables, the score breakdown chart and all assumptions under your
  answer, so do not repeat them: do not list every airport (name the leaders and any important
  exception), and do not repeat general disclaimers - only caveats specific to this question.
- End with one plain sentence on what this means for an investor."""


# ---------- Tools exposed to the LLM (thin wrappers; weights are deliberately not exposed) ----------

def _json(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)


@tool
def rank_airports(region: str = None, states: list[str] = None, top_n: int = 10) -> str:
    """Rank US airports by expansion-pressure score (0-100) with a per-component breakdown.
    Give exactly one of: region (a US Census division such as 'New England', 'Pacific',
    'Middle Atlantic', or 'US' for the whole country) or states (2-letter codes, e.g. ['CA','NV']).
    top_n: how many airports to return (1-50)."""
    return _json(t.rank_airports(region=region, states=states, top_n=top_n))


@tool
def compare_airports(airports: list[str]) -> str:
    """Compare 2-5 US airports side by side: score, breakdown, congestion indicators
    (NAS delay rate, delay rate, cancellations) and background facts. Use 3-letter IATA codes."""
    return _json(t.compare_airports(airports))


@tool
def get_airport_profile(airport: str) -> str:
    """All metrics for one US airport (3-letter IATA code): score, breakdown, national rank,
    and background facts. Smaller airports get raw metrics without a score."""
    return _json(t.get_airport_profile(airport))


@tool
def long_haul_share(airport: str, threshold_miles: float = 3000, mode: str = "all") -> str:
    """Share of departing flights at or above a distance threshold (miles, 500-8000, default 3000)
    in 2025. mode: 'passenger', 'cargo' or 'all' (both, reported separately)."""
    return _json(t.long_haul_share(airport, threshold_miles, mode))


@tool
def demand_gap(airport: str) -> str:
    """Proxies for unmet demand at one US airport (load factor, NAS delays, delays,
    cancellations, momentum), each with its national percentile, plus background facts."""
    return _json(t.demand_gap(airport))


@tool
def live_airport_status(airport: str) -> str:
    """Current FAA status for one US airport from the live FAA NAS Status API: ground stops,
    ground delay programs, delays, closure notices. Live snapshot only; not part of the score."""
    return _json(t.live_airport_status(airport))


AGENT_TOOLS = [rank_airports, compare_airports, get_airport_profile, long_haul_share,
               demand_gap, live_airport_status]


# ---------- Agent ----------

def build_agent():
    """Create the single agent with its tools and in-memory conversation memory."""
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {API_KEY_ENV} in the .env file (see .env.example).")
    model = ChatAnthropic(model=MODEL_NAME, api_key=api_key, effort=EFFORT,
                          max_tokens=4000, timeout=90, max_retries=2)
    return create_agent(model=model, tools=AGENT_TOOLS, system_prompt=SYSTEM_PROMPT,
                        checkpointer=InMemorySaver())


def check_user_message(text: str):
    """Light deterministic checks before the agent (option D). Returns an error text or None."""
    if not text or not text.strip():
        return "Please type a question."
    if len(text) > MAX_MESSAGE_CHARS:
        return f"Please keep your question under {MAX_MESSAGE_CHARS:,} characters."
    return None


def _text(message) -> str:
    """Plain text of a message (content can be a string or a list of content blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def _parse_tool_output(content) -> dict:
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return {"error": "tool_failed", "message": str(content)}


def _turn_result(messages: list) -> dict:
    """Answer, tool calls and raw tool results of the latest turn (after the last user message)."""
    last_user = max(i for i, m in enumerate(messages) if m.type == "human")
    turn = messages[last_user + 1:]
    calls = [{"name": c["name"], "args": c["args"]}
             for m in turn for c in (getattr(m, "tool_calls", None) or [])]
    results = [{"name": m.name, "data": _parse_tool_output(m.content)}
               for m in turn if m.type == "tool"]
    return {"answer": _text(turn[-1]), "tool_calls": calls, "tool_results": results,
            "blocked": False}


def stream_ask(agent, thread_id: str, question: str):
    """One conversation turn as a stream of events, for the UI:
    {"type": "tool", "name": ...} each time the agent calls a tool, then
    {"type": "done", "answer": ..., "tool_calls": [...], "tool_results": [...], "blocked": bool}."""
    problem = check_user_message(question)
    if problem:
        yield {"type": "done", "answer": problem, "tool_calls": [], "tool_results": [],
               "blocked": True}
        return
    config = {"configurable": {"thread_id": thread_id}}
    for update in agent.stream({"messages": [{"role": "user", "content": question}]},
                               config=config, stream_mode="updates"):
        for node_output in update.values():
            if not isinstance(node_output, dict):
                continue
            for message in node_output.get("messages", []):
                for call in getattr(message, "tool_calls", None) or []:
                    yield {"type": "tool", "name": call["name"], "args": call["args"]}
    yield {"type": "done", **_turn_result(agent.get_state(config).values["messages"])}


def ask(agent, thread_id: str, question: str) -> dict:
    """One conversation turn (non-streaming). Returns answer, tool calls and tool results."""
    for event in stream_ask(agent, thread_id, question):
        if event["type"] == "done":
            return {k: v for k, v in event.items() if k != "type"}


def new_thread_id() -> str:
    return str(uuid.uuid4())


if __name__ == "__main__":
    agent = build_agent()
    thread = new_thread_id()
    for q in ["Which airports in New England are strong candidates for terminal expansion?",
              "Why is PWM ranked second?",
              "Compare LA and Santa Ana airport congestion levels.",
              "What is the percentage of long haul flights out of Anchorage airport?",
              "What is the unmet flight demand in SFO airport and why?"]:
        r = ask(agent, thread, q)
        print("\n" + "=" * 100 + f"\nQ: {q}\nTOOLS: {r['tool_calls']}\n\n{r['answer']}")
