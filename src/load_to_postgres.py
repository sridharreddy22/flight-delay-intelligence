"""Step 3 - Bulk-load the cleaned data into PostgreSQL using COPY (fast for millions of rows).

Set your connection string first, then run:
    # Mac/Linux
    export DATABASE_URL="postgresql://postgres:yourpassword@localhost:5432/flights"
    # Windows PowerShell
    $env:DATABASE_URL="postgresql://postgres:yourpassword@localhost:5432/flights"

    python src/load_to_postgres.py
"""
import io
import os

import pandas as pd
import psycopg2

from config import PROCESSED_DIR, ROOT

COLUMNS = [
    "flight_date", "year", "month", "day_of_week", "dep_hour", "airline", "airline_name",
    "origin", "origin_city", "dest", "dest_city", "dep_delay", "arr_delay", "is_delayed",
    "cancelled", "diverted", "distance", "carrier_delay", "weather_delay", "nas_delay",
    "security_delay", "late_aircraft_delay",
]


def copy_frame(cur, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df[COLUMNS].to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    cur.copy_expert(
        f"COPY flights ({', '.join(COLUMNS)}) FROM STDIN WITH (FORMAT csv, NULL '')", buf
    )


def main() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("Set DATABASE_URL first (see the top of this file).")

    files = sorted(PROCESSED_DIR.glob("flights_*.parquet"))
    if not files:
        raise SystemExit("No processed files. Run src/clean_data.py first.")

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute((ROOT / "sql" / "01_schema.sql").read_text())
        total = 0
        for path in files:
            df = pd.read_parquet(path)
            df["flight_date"] = df["flight_date"].dt.date
            copy_frame(cur, df)
            conn.commit()
            total += len(df)
            print(f"loaded {path.name}: {len(df):,} rows (total {total:,})")

        print("building indexes...")
        cur.execute((ROOT / "sql" / "02_indexes.sql").read_text())
    print("done. Open sql/03_analysis.sql in pgAdmin, DBeaver, or VS Code and run the queries.")


if __name__ == "__main__":
    main()
