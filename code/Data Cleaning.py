import os
import pandas as pd


# ============================================================
# File paths
# ============================================================
# This script is stored inside the code/ folder.
# BASE_DIR points to the main project folder.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw_file = os.path.join(
    BASE_DIR,
    "data",
    "Timeseries_23.400_54.171_SA3_2a_2019_2023.csv"
)

output_file = os.path.join(
    BASE_DIR,
    "data",
    "cleaned_timeseries_local_2019_2023.csv"
)


# ============================================================
# Load raw PVGIS file
# ============================================================

df = pd.read_csv(raw_file, skiprows=8)


# ============================================================
# Rename PVGIS columns
# ============================================================

df = df.rename(columns={
    "time": "time_raw",
    "Gb(i)": "DNI",
    "Gd(i)": "Diffuse",
    "Gr(i)": "Reflected",
    "H_sun": "SunHeight",
    "T2m": "Tamb",
    "WS10m": "WindSpeed",
    "Int": "Flag"
})


# ============================================================
# Parse timestamp
# ============================================================

df["time_raw"] = df["time_raw"].astype(str).str.strip()

df["time_local"] = pd.to_datetime(
    df["time_raw"],
    format="%Y%m%d:%H%M",
    errors="coerce"
)

bad_time_rows = df["time_local"].isna().sum()
print(f"Rows with invalid time format: {bad_time_rows}")

df = df.dropna(subset=["time_local"]).copy()


# ============================================================
# Convert numeric columns
# ============================================================

numeric_cols = [
    "DNI",
    "Diffuse",
    "Reflected",
    "SunHeight",
    "Tamb",
    "WindSpeed",
    "Flag"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# Keep usable rows
# ============================================================
# DNI and Tamb are the main variables used later in the prediction workflow.

df = df.dropna(subset=["DNI", "Tamb"]).copy()


# ============================================================
# Sort time series
# ============================================================

df = df.sort_values("time_local").reset_index(drop=True)


# ============================================================
# Basic checks
# ============================================================

print("Cleaned shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())


# ============================================================
# Save cleaned dataset
# ============================================================

df.to_csv(output_file, index=False)

print("\nCleaned file saved:")
print(output_file)
