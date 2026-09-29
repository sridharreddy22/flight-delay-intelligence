"""Step 1 - Download monthly on-time performance files from BTS.

Usage:
    python src/download_data.py                  # all YEARS in config.py
    python src/download_data.py --years 2024 --months 1 2 3

Each month is ~30 MB zipped / ~250 MB unzipped. Already-downloaded months are skipped,
so you can stop and re-run safely.
"""
import argparse
import io
import shutil
import time
import zipfile

import requests

from config import BTS_URL, MONTHS, RAW_DIR, YEARS


def download_month(session: requests.Session, year: int, month: int) -> None:
    out = RAW_DIR / f"ontime_{year}_{month:02d}.csv"
    if out.exists():
        print(f"  skip {out.name} (already downloaded)")
        return

    url = BTS_URL.format(year=year, month=month)
    for attempt in range(1, 4):
        try:
            resp = session.get(url, timeout=180)
            resp.raise_for_status()
            break
        except requests.RequestException as err:
            print(f"  attempt {attempt} failed for {year}-{month:02d}: {err}")
            time.sleep(5 * attempt)
    else:
        print(f"  gave up on {year}-{month:02d}. Download it by hand (see README).")
        return

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
    print(f"  saved {out.name} ({out.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=YEARS)
    parser.add_argument("--months", type=int, nargs="+", default=MONTHS)
    args = parser.parse_args()

    with requests.Session() as session:
        session.headers["User-Agent"] = "flight-delay-intelligence (student portfolio project)"
        for year in args.years:
            for month in args.months:
                print(f"{year}-{month:02d}")
                download_month(session, year, month)


if __name__ == "__main__":
    main()
