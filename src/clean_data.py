"""Step 2 - Clean and validate each raw monthly file.

Reads data/raw/ontime_YYYY_MM.csv, writes data/processed/flights_YYYY_MM.parquet,
and writes a data-quality report to reports/data_quality.csv.

Usage:
    python src/clean_data.py
"""
import numpy as np
import pandas as pd

from config import CARRIER_NAMES, CAUSE_COLUMNS, PROCESSED_DIR, RAW_COLUMNS, RAW_DIR, REPORTS_DIR


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=RAW_COLUMNS)

    df["flight_date"] = pd.to_datetime(df["flight_date"], format="mixed")
    df["year"] = df["flight_date"].dt.year.astype("int16")
    df["month"] = df["flight_date"].dt.month.astype("int8")
    df["day_of_week"] = (df["flight_date"].dt.dayofweek + 1).astype("int8")  # 1 = Monday

    # CRSDepTime is scheduled local time as HHMM (e.g. 1730). 2400 means midnight.
    dep = pd.to_numeric(df["crs_dep_time"], errors="coerce").fillna(0).astype(int)
    df["dep_hour"] = ((dep // 100) % 24).astype("int8")
    df = df.drop(columns="crs_dep_time")

    df["cancelled"] = df["cancelled"].fillna(0).astype("int8")
    df["diverted"] = df["diverted"].fillna(0).astype("int8")

    # is_delayed (arrived 15+ min late) is blank for cancelled/diverted flights - keep it blank.
    df["is_delayed"] = df["is_delayed"].astype("float32")

    # BTS only fills cause minutes for delayed flights; blanks there mean 0 minutes.
    df[CAUSE_COLUMNS] = df[CAUSE_COLUMNS].fillna(0).astype("float32")
    for col in ("dep_delay", "arr_delay", "distance"):
        df[col] = df[col].astype("float32")

    for col in ("airline", "origin", "dest", "origin_city", "dest_city"):
        df[col] = df[col].astype(str).str.strip()
    df["airline_name"] = df["airline"].map(CARRIER_NAMES).fillna(df["airline"])

    # A flight is uniquely identified by date + airline + flight number + route.
    return df.drop_duplicates(subset=["flight_date", "airline", "flight_number", "origin", "dest"])


def quality_report(raw: pd.DataFrame, df: pd.DataFrame, name: str) -> dict:
    """Checks a reviewer would ask about. Printed and saved so you can cite them."""
    completed = (df["cancelled"] == 0) & (df["diverted"] == 0)
    delayed = df["is_delayed"] == 1
    cause_total = df.loc[delayed, CAUSE_COLUMNS].sum(axis=1)
    causes_match = np.isclose(cause_total, df.loc[delayed, "arr_delay"], atol=1).mean()

    return {
        "file": name,
        "raw_rows": len(raw),
        "clean_rows": len(df),
        "duplicates_removed": len(raw) - len(df),
        "date_min": df["flight_date"].min().date(),
        "date_max": df["flight_date"].max().date(),
        "cancelled_pct": round(100 * float(df["cancelled"].mean()), 2),
        "diverted_pct": round(100 * float(df["diverted"].mean()), 2),
        "delayed_pct_of_completed": round(100 * float(df.loc[completed, "is_delayed"].mean()), 2),
        "completed_missing_arr_delay": int(df.loc[completed, "arr_delay"].isna().sum()),
        "delay_causes_sum_to_arr_delay_pct": round(100 * float(causes_match), 2),
        "unknown_airline_codes": sorted(set(df["airline"]) - set(CARRIER_NAMES)),
    }


def main() -> None:
    files = sorted(RAW_DIR.glob("ontime_*.csv"))
    if not files:
        raise SystemExit("No raw files found. Run src/download_data.py "
                         "(or scripts/make_sample_data.py to test with fake data).")

    reports = []
    for path in files:
        raw = pd.read_csv(path, usecols=list(RAW_COLUMNS), low_memory=False)
        df = clean(raw)
        out = PROCESSED_DIR / path.name.replace("ontime_", "flights_").replace(".csv", ".parquet")
        df.to_parquet(out, index=False)
        report = quality_report(raw, df, path.name)
        reports.append(report)
        print(f"{path.name}: {report['clean_rows']:,} rows, "
              f"{report['delayed_pct_of_completed']}% delayed, "
              f"{report['cancelled_pct']}% cancelled")

    pd.DataFrame(reports).to_csv(REPORTS_DIR / "data_quality.csv", index=False)
    print(f"\nData-quality report: {REPORTS_DIR / 'data_quality.csv'}")


if __name__ == "__main__":
    main()
