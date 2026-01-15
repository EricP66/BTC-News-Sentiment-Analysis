import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# Base directory: same folder as this .py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "btc_usd_daily_price_cryptocompare.csv"
)

# Configuration
FSYM = "BTC"                 # From symbol
TSYM = "USD"                 # To symbol
START_DATE = "2011-01-01"    # Start date (inclusive)
LIMIT_PER_CALL = 2000        # Max days per API call
SLEEP_SECONDS = 0.3          # Delay between requests

HISTODAY_URL = "https://min-api.cryptocompare.com/data/v2/histoday"


def to_unix(date_str: str) -> int:
    """Convert YYYY-MM-DD string to Unix timestamp (UTC)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def call_histoday(to_ts: int) -> dict:
    """Call CryptoCompare histoday endpoint without API key."""
    params = {
        "fsym": FSYM,
        "tsym": TSYM,
        "limit": LIMIT_PER_CALL,
        "toTs": to_ts,
    }

    r = requests.get(HISTODAY_URL, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")

    data = r.json()
    if data.get("Response") == "Error":
        raise RuntimeError(f"API Error: {data.get('Message')}")

    return data


def main():
    start_ts = to_unix(START_DATE)
    to_ts = int(datetime.now(timezone.utc).timestamp())

    all_rows = []
    page = 0

    # Paginate backwards in time until start date
    while True:
        page += 1
        print(f"Fetching page {page} ...")

        payload = call_histoday(to_ts)
        rows = payload.get("Data", {}).get("Data", [])

        if not rows:
            break

        all_rows.extend(rows)

        min_ts = min(r["time"] for r in rows)
        if min_ts <= start_ts:
            break

        to_ts = min_ts - 1
        time.sleep(SLEEP_SECONDS)

    # Build DataFrame
    df = pd.DataFrame(all_rows)

    df["Date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.date
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volumefrom": "VolumeFrom",
        "volumeto": "VolumeTo",
    })

    df = df[["Date", "Open", "High", "Low", "Close", "VolumeFrom", "VolumeTo"]]

    df = (
        df.sort_values("Date")
          .drop_duplicates(subset=["Date"], keep="last")
    )

    df = df[df["Date"] >= datetime.strptime(START_DATE, "%Y-%m-%d").date()]

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Display first 5 rows (sanity check)
    print(df.head(5))


if __name__ == "__main__":
    main()
