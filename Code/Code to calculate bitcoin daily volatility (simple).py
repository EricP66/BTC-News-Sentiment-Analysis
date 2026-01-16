import pandas as pd
import os

# Base directory (this .py file location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
INPUT_FILE = os.path.join(BASE_DIR, "btc_usd_daily_price_cryptocompare.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "btc_usd_daily_volatility_cryptocompare_simple.csv")

# Load data
df = pd.read_csv(INPUT_FILE)

# Ensure Date is datetime and sorted
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Clean Close (force numeric)
# Convert to numeric; non-numeric becomes NaN
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

before = len(df)
df = df.dropna(subset=["Close"]).reset_index(drop=True)
after = len(df)

print(f"Close cleaned: kept {after}/{before} rows (dropped {before - after}).")

# Calculate daily price fluctuation metrics
# Directional daily price return (simple return)
df["Simple_return"] = df["Close"].pct_change()

# Absolute price change (volatility proxy)
df["Daily_volatility_simple"] = df["Simple_return"].abs()

# Drop first NaN (because no pct_change at the first day)
df = df.dropna(subset=["Daily_volatility_simple"]).reset_index(drop=True)

df_keep = df[["Date", "Close", "Simple_return", "Daily_volatility_simple"]].copy()

print(df_keep.head())
print(df_keep.tail())
print(df_keep.info())

# Save
df_keep.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


