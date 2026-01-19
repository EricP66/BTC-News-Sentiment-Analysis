import pandas as pd
import os

# Base directory (this .py file location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
INPUT_FILE = os.path.join(BASE_DIR, "btc_usd_daily_price_cryptocompare.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "btc_usd_daily_volatility_cryptocompare_log.csv")

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
# Directional daily price return (log return)
import numpy as np
df["Log_return"] = np.log(df["Close"] / df["Close"].shift(1))

# Absolute price return (volatility proxy)
df["Daily_volatility_log"] = df["Log_return"].abs()

# Drop first NaN (because no volatility at the first day)
df = df.dropna(subset=["Daily_volatility_log"]).reset_index(drop=True)

# Winsorize (clip) Daily_volatility_log at 1st and 99th percentiles
low, high = df["Daily_volatility_log"].quantile([0.01, 0.99])
df["Daily_volatility_log"] = df["Daily_volatility_log"].clip(lower=low, upper=high)

# Select columns needed
df_keep = df[["Date", "Close", "Log_return", "Daily_volatility_log"]].copy()
print(df_keep.head())
print(df_keep.tail())
print(df_keep.info())

# Save
df_keep.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


