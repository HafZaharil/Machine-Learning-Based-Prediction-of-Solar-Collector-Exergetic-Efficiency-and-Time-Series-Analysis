import os
import pandas as pd
import matplotlib

matplotlib.use("MacOSX")

import matplotlib.pyplot as plt


# ============================================================
# File paths
# ============================================================
# This script is stored inside the code/ folder.
# BASE_DIR points to the main project folder.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

classification_file = os.path.join(
    BASE_DIR,
    "results",
    "effex_2023_suitability_classification_K1.csv"
)

output_dir = os.path.join(BASE_DIR, "results", "time_series_analysis")
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# Load classification results
# ============================================================

df = pd.read_csv(classification_file)

df["time_local"] = pd.to_datetime(df["time_local"])
df = df.sort_values("time_local").reset_index(drop=True)

# Prevent physically impossible negative EffEX values
df["Max_Predicted_EffEX"] = df["Max_Predicted_EffEX"].clip(lower=0)

print("Loaded classification data:", df.shape)
print(df.head())


# ============================================================
# Keep only solar-active hours
# ============================================================

df_solar = df[df["DNI_W_per_m2"] > 0].copy()

print("\nRows with DNI > 0:", df_solar.shape[0])


# ============================================================
# Basic checks
# ============================================================

print("\nEffEX statistics, all hours:")
print(df["Max_Predicted_EffEX"].describe())

print("\nEffEX statistics, DNI > 0 hours only:")
print(df_solar["Max_Predicted_EffEX"].describe())

print("\nClassification counts, all hours:")
print(df["Classification"].value_counts())

print("\nClassification counts, DNI > 0 hours only:")
print(df_solar["Classification"].value_counts())


# ============================================================
# Daily average EffEX
# ============================================================
# Daily average is calculated using only hours with DNI > 0.
# This avoids mixing night-time hours into the daily performance metric.

df_solar["date"] = df_solar["time_local"].dt.date
df_solar["month"] = df_solar["time_local"].dt.to_period("M")

daily_df = (
    df_solar
    .groupby("date")
    .agg(
        Daily_Average_EffEX=("Max_Predicted_EffEX", "mean"),
        Nonzero_DNI_Hours=("DNI_W_per_m2", "count"),
        Daily_Mean_DNI=("DNI_W_per_m2", "mean"),
        Daily_Max_DNI=("DNI_W_per_m2", "max")
    )
    .reset_index()
)

daily_df["date"] = pd.to_datetime(daily_df["date"])


# ============================================================
# Monthly average EffEX
# ============================================================
# Monthly value is calculated from the daily averages.

daily_df["month"] = daily_df["date"].dt.to_period("M")

monthly_df = (
    daily_df
    .groupby("month")
    .agg(
        Monthly_Average_EffEX=("Daily_Average_EffEX", "mean"),
        Mean_Nonzero_DNI_Hours=("Nonzero_DNI_Hours", "mean"),
        Monthly_Mean_DNI=("Daily_Mean_DNI", "mean")
    )
    .reset_index()
)

monthly_df["month"] = monthly_df["month"].astype(str)


# ============================================================
# Daily average plot
# ============================================================

plt.figure(figsize=(14, 6))
plt.plot(
    daily_df["date"],
    daily_df["Daily_Average_EffEX"],
    linewidth=1.5,
    label="Daily Average EffEX"
)
plt.axhline(
    35,
    linestyle="--",
    linewidth=1.5,
    color="red",
    label="35% threshold"
)
plt.xlabel("Date")
plt.ylabel("Daily Average EffEX (%)")
plt.title("Daily Average Predicted EffEX Across 2023, DNI > 0 Hours Only")
plt.grid(False)
plt.legend()
plt.tight_layout()

daily_plot_path = os.path.join(
    output_dir,
    "daily_average_effex_2023_nonzero_dni_hours_only.png"
)

plt.savefig(daily_plot_path, dpi=600)
plt.show()


# ============================================================
# Daily average with 7-day rolling average
# ============================================================

daily_df["Daily_Average_EffEX_7day_Rolling"] = (
    daily_df["Daily_Average_EffEX"]
    .rolling(window=7, min_periods=1)
    .mean()
)

plt.figure(figsize=(14, 6))
plt.plot(
    daily_df["date"],
    daily_df["Daily_Average_EffEX"],
    linewidth=0.8,
    label="Daily Average EffEX"
)
plt.plot(
    daily_df["date"],
    daily_df["Daily_Average_EffEX_7day_Rolling"],
    linewidth=1.5,
    label="7-Day Rolling Average EffEX"
)
plt.axhline(
    35,
    linestyle="--",
    linewidth=1.5,
    color="red",
    label="35% threshold"
)
plt.xlabel("Date")
plt.ylabel("Daily Average EffEX (%)")
plt.title("Daily Average Predicted EffEX with 7-Day Rolling Average")
plt.grid(False)
plt.legend()
plt.tight_layout()

rolling_plot_path = os.path.join(
    output_dir,
    "daily_average_effex_2023_with_7day_rolling_average.png"
)

plt.savefig(rolling_plot_path, dpi=600)
plt.show()


# ============================================================
# Monthly average plot
# ============================================================

plt.figure(figsize=(12, 6))
plt.bar(
    monthly_df["month"],
    monthly_df["Monthly_Average_EffEX"],
    label="Monthly Average EffEX"
)
plt.axhline(
    35,
    linestyle="--",
    linewidth=1.5,
    color="red",
    label="35% threshold"
)
plt.xlabel("Month")
plt.ylabel("Monthly Average of Daily EffEX (%)")
plt.title("Monthly Average Predicted EffEX in 2023, Based on DNI > 0 Hours")
plt.xticks(rotation=45)
plt.grid(False)
plt.legend()
plt.tight_layout()

monthly_plot_path = os.path.join(
    output_dir,
    "monthly_average_effex_2023_nonzero_dni_hours_only.png"
)

plt.savefig(monthly_plot_path, dpi=600)
plt.show()


# ============================================================
# Save processed summaries
# ============================================================

daily_csv_path = os.path.join(
    output_dir,
    "daily_average_effex_2023_nonzero_dni_hours_only.csv"
)

monthly_csv_path = os.path.join(
    output_dir,
    "monthly_average_effex_2023_nonzero_dni_hours_only.csv"
)

daily_df.to_csv(daily_csv_path, index=False)
monthly_df.to_csv(monthly_csv_path, index=False)

print("\nSaved files:")
print(daily_plot_path)
print(rolling_plot_path)
print(monthly_plot_path)
print(daily_csv_path)
print(monthly_csv_path)
