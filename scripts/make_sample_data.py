"""Generate FAKE data in the exact BTS file format, so you can test the whole
pipeline in minutes before downloading the real (large) files.

Usage:
    python scripts/make_sample_data.py            # 20,000 flights per month, 2023-2024
    python scripts/make_sample_data.py --rows 5000

Delete data/raw/ontime_*.csv before switching to the real data.
Never publish results computed from this fake data.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import RAW_DIR, YEARS  # noqa: E402

AIRPORTS = {
    "ATL": ("Atlanta, GA", 0.00), "DFW": ("Dallas/Fort Worth, TX", 0.02),
    "DEN": ("Denver, CO", 0.03), "ORD": ("Chicago, IL", 0.04),
    "LAX": ("Los Angeles, CA", -0.01), "JFK": ("New York, NY", 0.06),
    "LAS": ("Las Vegas, NV", 0.01), "MCO": ("Orlando, FL", 0.05),
    "SEA": ("Seattle, WA", -0.03), "PHX": ("Phoenix, AZ", -0.02),
    "STL": ("St. Louis, MO", 0.00), "MSP": ("Minneapolis, MN", -0.03),
    "BOS": ("Boston, MA", 0.03), "EWR": ("Newark, NJ", 0.07),
    "MDW": ("Chicago, IL", 0.02), "BNA": ("Nashville, TN", 0.01),
}
AIRLINES = {"WN": 0.02, "AA": 0.01, "DL": -0.05, "UA": 0.00, "AS": -0.03,
            "B6": 0.07, "NK": 0.05, "F9": 0.08, "OO": -0.01, "G4": 0.04}
MONTH_EFFECT = {1: 0.0, 2: -0.02, 3: 0.0, 4: -0.01, 5: 0.01, 6: 0.06,
                7: 0.07, 8: 0.03, 9: -0.04, 10: -0.03, 11: -0.03, 12: 0.04}


def route_distance(a: str, b: str) -> float:
    """Stable fake distance per airport pair."""
    key = "".join(sorted((a, b)))
    return float(200 + sum(ord(c) * (i + 1) for i, c in enumerate(key)) * 37 % 2400)


def make_month(year: int, month: int, n: int, rng: np.random.Generator) -> pd.DataFrame:
    codes = list(AIRPORTS)
    origin = rng.choice(codes, n)
    dest = rng.choice(codes, n)
    same = origin == dest
    dest[same] = rng.choice(codes, same.sum())
    dest[origin == dest] = np.where(origin[origin == dest] == "ATL", "DFW", "ATL")

    days = pd.Period(f"{year}-{month:02d}").days_in_month
    dates = pd.to_datetime([f"{year}-{month:02d}-{d:02d}" for d in rng.integers(1, days + 1, n)])
    hour_weights = np.r_[np.full(6, 1.5), np.full(13, 1.0)]  # hours 5-10 busier, then 11-23
    hour = rng.choice(np.arange(5, 24), n, p=hour_weights / hour_weights.sum())
    airline = rng.choice(list(AIRLINES), n)

    p = (0.12 + 0.012 * (hour - 5) + MONTH_EFFECT[month]
         + np.vectorize(AIRLINES.get)(airline) + np.vectorize(lambda a: AIRPORTS[a][1])(origin)
         + np.where(dates.dayofweek.isin([3, 4, 6]), 0.02, 0.0))
    delayed = rng.random(n) < np.clip(p, 0.02, 0.9)
    cancelled = rng.random(n) < 0.015
    diverted = (~cancelled) & (rng.random(n) < 0.002)
    completed = ~cancelled & ~diverted

    arr = np.where(delayed, 15 + rng.exponential(45, n), np.clip(rng.normal(-6, 8, n), -40, 14))
    arr = np.round(arr)
    shares = rng.dirichlet([3, 0.6, 2, 0.05, 4], n)  # carrier, weather, NAS, security, late aircraft
    cause = np.round(shares * arr[:, None])
    cause[:, 4] += arr - cause.sum(axis=1)  # make causes add up exactly

    df = pd.DataFrame({
        "Year": year, "Month": month,
        "FlightDate": dates.strftime("%Y-%m-%d"),
        "Reporting_Airline": airline,
        "Origin": origin, "OriginCityName": [AIRPORTS[a][0] for a in origin],
        "Dest": dest, "DestCityName": [AIRPORTS[a][0] for a in dest],
        "CRSDepTime": hour * 100 + rng.choice([0, 15, 30, 45], n),
        "DepDelay": np.where(completed, arr + rng.normal(3, 5, n).round(), np.nan),
        "ArrDelay": np.where(completed, arr, np.nan),
        "ArrDel15": np.where(completed, delayed.astype(float), np.nan),
        "Cancelled": cancelled.astype(float), "Diverted": diverted.astype(float),
        "Distance": [route_distance(o, d) for o, d in zip(origin, dest)],
    })
    for i, col in enumerate(["CarrierDelay", "WeatherDelay", "NASDelay",
                             "SecurityDelay", "LateAircraftDelay"]):
        df[col] = np.where(completed & delayed, cause[:, i], np.nan)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20_000)
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    for year in YEARS:
        for month in range(1, 13):
            path = RAW_DIR / f"ontime_{year}_{month:02d}.csv"
            make_month(year, month, args.rows, rng).to_csv(path, index=False)
            print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
