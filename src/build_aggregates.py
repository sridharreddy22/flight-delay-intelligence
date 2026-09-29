"""Step 4 - Build small summary tables for the live app.

Millions of raw rows won't fit on a free Streamlit server, so we pre-aggregate.
We store SUMS (flights, delayed, minutes...) rather than rates, so the app can
re-combine any filter (year, airline, airport) and still compute correct rates.

Usage:
    python src/build_aggregates.py
"""
import pandas as pd

from config import APP_DATA_DIR, CARRIER_NAMES, CAUSE_COLUMNS, PROCESSED_DIR

MEASURES = ["flights", "completed", "delayed", "cancelled", "diverted",
            "departed", "dep_delayed", "dep_late_minutes",
            "arr_delay_sum", "late_minutes"] + CAUSE_COLUMNS

TABLES = {
    "monthly": ["year", "month", "airline", "origin"],
    "hourly": ["year", "dep_hour", "airline", "origin"],
    "routes": ["year", "origin", "dest", "airline"],
    "daily": ["flight_date", "origin"],
}


def add_measures(df: pd.DataFrame) -> pd.DataFrame:
    completed = (df["cancelled"] == 0) & (df["diverted"] == 0)
    delayed = df["is_delayed"] == 1
    departed = (df["cancelled"] == 0) & df["dep_delay"].notna()
    dep_delayed = departed & (df["dep_delay"] >= 15)  # left gate 15+ min late
    return df.assign(
        flights=1,
        completed=completed.astype("int32"),
        delayed=delayed.astype("int32"),
        arr_delay_sum=df["arr_delay"].where(completed, 0).fillna(0),
        late_minutes=df["arr_delay"].where(delayed, 0).fillna(0),
        departed=departed.astype("int32"),
        dep_delayed=dep_delayed.astype("int32"),
        dep_late_minutes=df["dep_delay"].where(dep_delayed, 0).fillna(0),
    )


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.groupby(keys, observed=True, as_index=False)[MEASURES].sum()


def main() -> None:
    files = sorted(PROCESSED_DIR.glob("flights_*.parquet"))
    if not files:
        raise SystemExit("No processed files. Run src/clean_data.py first.")

    parts = {name: [] for name in TABLES}
    airports, distances = [], []

    for path in files:  # one month at a time keeps memory low
        df = add_measures(pd.read_parquet(path))
        for name, keys in TABLES.items():
            parts[name].append(summarize(df, keys))
        airports.append(df[["origin", "origin_city"]].drop_duplicates())
        distances.append(df.groupby(["origin", "dest"], as_index=False)["distance"].median())
        print(f"summarized {path.name}")

    for name, keys in TABLES.items():
        table = summarize(pd.concat(parts[name]), keys)
        table.to_parquet(APP_DATA_DIR / f"{name}.parquet", index=False)
        print(f"{name}: {len(table):,} rows")

    flights_by_airport = pd.concat(parts["monthly"]).groupby("origin")["flights"].sum()
    airport_table = (pd.concat(airports).drop_duplicates("origin")
                     .rename(columns={"origin": "airport", "origin_city": "city"}))
    airport_table["flights"] = airport_table["airport"].map(flights_by_airport).fillna(0).astype(int)
    airport_table.sort_values("flights", ascending=False).to_parquet(
        APP_DATA_DIR / "airports.parquet", index=False)

    (pd.concat(distances).groupby(["origin", "dest"], as_index=False)["distance"].median()
     .to_parquet(APP_DATA_DIR / "route_distance.parquet", index=False))

    codes = sorted(pd.concat(parts["monthly"])["airline"].unique())
    pd.DataFrame({"airline": codes,
                  "airline_name": [CARRIER_NAMES.get(c, c) for c in codes]}).to_parquet(
        APP_DATA_DIR / "airlines.parquet", index=False)

    print(f"\nApp tables written to {APP_DATA_DIR}")


if __name__ == "__main__":
    main()
