"""Loads and cleans the BTS data files and builds per-airport metrics."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

# Data files ship with the repo, gzip-compressed (pandas reads .csv.gz directly).
# Sources: BTS T-100 Segment (All Carriers), BTS Airline On-Time Delay Causes, OurAirports.
T100_FILES = {
    2025: "t100_2025.csv.gz",
    2026: "t100_2026_partial.csv.gz",   # January-August 2026
}
DELAY_FILE = "delay_causes_2025_2026.csv.gz"
AIRPORTS_FILE = "airports.csv.gz"

# F = scheduled passenger service. We use only scheduled flights for passenger metrics.
PASSENGER_CLASSES = ["F"]
# G = scheduled all-cargo, P = non-scheduled all-cargo
CARGO_CLASSES = ["G", "P"]

NEW_ENGLAND = ["MA", "CT", "RI", "NH", "VT", "ME"]

# T-100 also contains foreign and Canadian origin airports (flights to the US).
# We rank only US airports: 50 states + DC + Puerto Rico (PR) + US Virgin Islands (VI).
# Pacific territories (T-100 state code TT: Guam, Saipan...) are deliberately excluded:
# they are outside the investment focus and none of the example questions involve them.
US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY", "DC", "PR", "VI",
]

# Airport codes that differ between sources (delay file code -> T-100 code).
# PBI appears as DJT in the most recent BTS delay data.
DELAY_CODE_ALIASES = {"DJT": "PBI"}

DELAY_COUNT_COLUMNS = ["arr_flights", "arr_del15", "arr_cancelled", "nas_ct", "weather_ct"]
DELAY_WINDOW_MONTHS = 12

# Airports below this many yearly enplanements are too small for a terminal investment
MIN_ENPLANEMENTS = 500_000


# ---------- Loading ----------

def load_t100() -> pd.DataFrame:
    """Load all T-100 years into one table and drop rows with no flights."""
    frames = []
    for filename in T100_FILES.values():
        df = pd.read_csv(DATA_DIR / filename)
        df.columns = df.columns.str.upper()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df[df["DEPARTURES_PERFORMED"] > 0]


def passenger_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only scheduled passenger flights."""
    return df[df["CLASS"].isin(PASSENGER_CLASSES)]


def us_origins(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only flights departing from US airports (see US_STATES)."""
    return df[df["ORIGIN_STATE_ABR"].isin(US_STATES)]


def load_delays() -> pd.DataFrame:
    """Load the BTS delay-causes file (one row per carrier, airport and month)."""
    df = pd.read_csv(DATA_DIR / DELAY_FILE)
    df.columns = df.columns.str.strip().str.lower()

    required = ["year", "month", "airport"] + DELAY_COUNT_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Columns in file: {list(df.columns)}")

    df["airport"] = df["airport"].replace(DELAY_CODE_ALIASES)
    # A few tiny rows (at most 1 flight each) have missing counts. Drop them instead of
    # treating missing as zero.
    return df.dropna(subset=DELAY_COUNT_COLUMNS)


# ---------- Metrics ----------

def momentum_months(pax_df: pd.DataFrame) -> list:
    """Months available in 2026; compared against the same months in 2025."""
    return sorted(pax_df.loc[pax_df["YEAR"] == 2026, "MONTH"].unique())


def passenger_metrics(t100: pd.DataFrame) -> pd.DataFrame:
    """Volume, recent momentum and load factor per US airport."""
    pax_df = us_origins(passenger_flights(t100))
    months = momentum_months(pax_df)

    pax = pax_df.pivot_table(index="ORIGIN", columns="YEAR", values="PASSENGERS", aggfunc="sum")
    seats = pax_df.pivot_table(index="ORIGIN", columns="YEAR", values="SEATS", aggfunc="sum")
    ytd = pax_df[pax_df["MONTH"].isin(months)].pivot_table(
        index="ORIGIN", columns="YEAR", values="PASSENGERS", aggfunc="sum")
    state = pax_df.groupby("ORIGIN")["ORIGIN_STATE_ABR"].first()
    national_momentum = ytd[2026].sum() / ytd[2025].sum() - 1

    return pd.DataFrame({
        "state": state,
        "enplanements_2025": pax[2025],
        "momentum_2026_ytd": ytd[2026] / ytd[2025] - 1,
        "load_factor_2025": pax[2025] / seats[2025],
        "momentum_vs_national": (ytd[2026] / ytd[2025] - 1) - national_momentum,
    })


def delay_window(delays: pd.DataFrame) -> list:
    """The latest DELAY_WINDOW_MONTHS periods (YYYYMM) in the delay file."""
    period = delays["year"] * 100 + delays["month"]
    return sorted(period.unique())[-DELAY_WINDOW_MONTHS:]


def delay_metrics(delays: pd.DataFrame) -> pd.DataFrame:
    """Congestion indicators per airport over the latest 12 months available."""
    period = delays["year"] * 100 + delays["month"]
    recent = delays[period.isin(delay_window(delays))]

    g = recent.groupby("airport")[DELAY_COUNT_COLUMNS].sum()
    return pd.DataFrame({
        # Share of arriving flights delayed by NAS (airport/airspace capacity) causes.
        # This is the congestion metric used in the score.
        "nas_delay_rate": g["nas_ct"] / g["arr_flights"],
        # Supporting indicators (not in the score; used to explain demand gaps)
        "delay_rate": g["arr_del15"] / g["arr_flights"],
        "cancel_rate": g["arr_cancelled"] / g["arr_flights"],
        "nas_share_of_delays": g["nas_ct"] / g["arr_del15"],
        "weather_share_of_delays": g["weather_ct"] / g["arr_del15"],
    })


def build_airport_metrics(min_enplanements: int = MIN_ENPLANEMENTS, t100: pd.DataFrame = None,
                          delays: pd.DataFrame = None) -> pd.DataFrame:
    """One row per US airport with all metrics, filtered by minimum size (0 = all airports).
    Already-loaded t100 / delays tables can be passed in to avoid reading the files again."""
    t100 = load_t100() if t100 is None else t100
    delays = load_delays() if delays is None else delays
    metrics = passenger_metrics(t100).join(delay_metrics(delays), how="left")
    return metrics[metrics["enplanements_2025"] >= min_enplanements]


def load_airport_names() -> pd.DataFrame:
    """Airport name and city by IATA code (OurAirports), for US airports incl. PR and VI."""
    df = pd.read_csv(DATA_DIR / AIRPORTS_FILE,
                     usecols=["iata_code", "name", "municipality", "iso_country", "type"])
    df = df[df["iso_country"].isin(["US", "PR", "VI"]) & df["iata_code"].notna()]
    # A code can appear on a closed/minor record too; prefer the largest airport type.
    size_order = {"large_airport": 0, "medium_airport": 1, "small_airport": 2}
    df = df.assign(order=df["type"].map(size_order).fillna(9)).sort_values("order")
    return df.drop_duplicates("iata_code").set_index("iata_code")[["name", "municipality"]]


def airports_without_delay_data(metrics: pd.DataFrame) -> list:
    """Airports in the ranking universe that have no delay data (shown, never set to zero)."""
    return sorted(metrics.index[metrics["nas_delay_rate"].isna()])


if __name__ == "__main__":
    t100 = passenger_flights(load_t100())
    print("Momentum months (2026 vs 2025):", [int(m) for m in momentum_months(t100)])
    window = delay_window(load_delays())
    print("Delay window:", window[0], "-", window[-1], f"({len(window)} months)")

    metrics = build_airport_metrics()
    print("US airports with >=", f"{MIN_ENPLANEMENTS:,}", "enplanements:", len(metrics))
    print("Without delay data:", airports_without_delay_data(metrics))
    ne = metrics[metrics["state"].isin(NEW_ENGLAND)].sort_values("enplanements_2025", ascending=False)

    pd.set_option("display.float_format", "{:,.2f}".format)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\nNew England airports (min", f"{MIN_ENPLANEMENTS:,}", "enplanements):")
    print(ne)