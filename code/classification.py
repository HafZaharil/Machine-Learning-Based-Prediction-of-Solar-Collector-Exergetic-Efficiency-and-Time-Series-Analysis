import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# File paths
# ============================================================
# This script is stored inside the code/ folder.
# BASE_DIR points to the main project folder.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

weather_2023_file = os.path.join(
    BASE_DIR,
    "data",
    "cleaned_timeseries_local_2023.csv"
)

model_dir = os.path.join(BASE_DIR, "models")

scaler_path = os.path.join(model_dir, "effex_input_scaler.pkl")
model_path = os.path.join(model_dir, "effex_nn_weights_and_biases.pth")
feature_cols_path = os.path.join(model_dir, "effex_feature_cols.pkl")

output_dir = os.path.join(BASE_DIR, "results")
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# Load 2023 weather data
# ============================================================

df_2023 = pd.read_csv(weather_2023_file)

df_2023["time_local"] = pd.to_datetime(df_2023["time_local"])
df_2023 = df_2023.sort_values("time_local").reset_index(drop=True)

print("Loaded 2023 weather data:", df_2023.shape)
print(df_2023.head())


# ============================================================
# Load saved EffEX model files
# ============================================================

scaler = joblib.load(scaler_path)
feature_cols = joblib.load(feature_cols_path)

print("\nModel feature order:")
print(feature_cols)


# ============================================================
# Neural network structure
# ============================================================
# This must match the network used during training.
# No training is done here; the saved weights are loaded below.

class ThermalNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)


model = ThermalNN(input_dim=len(feature_cols))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

print("\nEffEX model loaded successfully.")


# ============================================================
# Prediction function
# ============================================================

def predict_effex(input_df):
    input_df = input_df[feature_cols].copy()

    x_scaled = scaler.transform(input_df.values)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(x_tensor).numpy().flatten()

    return prediction


# ============================================================
# Operating assumptions
# ============================================================

FIXED_PRESSUREHTF = 20000.0
FIXED_K = 1.0

TIN_MIN = 350.0
TIN_MAX = 850.0
TIN_STEP = 5.0

MHTF_MIN = 0.50
MHTF_MAX = 5.00
MHTF_STEP = 0.10

SUITABLE_THRESHOLD = 35.0

tin_values = np.arange(TIN_MIN, TIN_MAX + 1e-9, TIN_STEP)
mhtf_values = np.arange(MHTF_MIN, MHTF_MAX + 1e-9, MHTF_STEP)


# ============================================================
# Find the best EffEX for one weather condition
# ============================================================

def find_best_effex(dni, tamb_k):
    tin_grid, mhtf_grid = np.meshgrid(
        tin_values,
        mhtf_values,
        indexing="xy"
    )

    n_points = tin_grid.size

    candidate_points = pd.DataFrame({
        "Tin": tin_grid.ravel(),
        "DNI": np.full(n_points, float(dni)),
        "Mhtf": mhtf_grid.ravel(),
        "Tamb": np.full(n_points, float(tamb_k)),
        "Pressurehtf": np.full(n_points, FIXED_PRESSUREHTF),
        "K": np.full(n_points, FIXED_K),
    })

    candidate_points = candidate_points[feature_cols]

    predicted_effex = predict_effex(candidate_points)
    best_index = int(np.argmax(predicted_effex))

    best_tin_k = float(candidate_points.iloc[best_index]["Tin"])

    return {
        "max_effex": float(predicted_effex[best_index]),
        "best_tin_k": best_tin_k,
        "best_tin_c": best_tin_k - 273.15,
        "best_mhtf": float(candidate_points.iloc[best_index]["Mhtf"]),
    }


# ============================================================
# Run hourly classification for 2023
# ============================================================

results = []

for i, row in df_2023.iterrows():
    dni = float(row["DNI"])

    # The cleaned PVGIS file stores Tamb in °C.
    # The trained model expects Tamb in Kelvin.
    tamb_c = float(row["Tamb"])
    tamb_k = tamb_c + 273.15

    best = find_best_effex(dni=dni, tamb_k=tamb_k)

    if best["max_effex"] > SUITABLE_THRESHOLD:
        classification = "Suitable"
    else:
        classification = "Not suitable"

    results.append({
        "time_local": row["time_local"],
        "DNI_W_per_m2": dni,
        "Tamb_C": tamb_c,
        "Max_Predicted_EffEX": best["max_effex"],
        "Best_Tin_K": best["best_tin_k"],
        "Best_Tin_C": best["best_tin_c"],
        "Best_Mhtf": best["best_mhtf"],
        "Pressurehtf": FIXED_PRESSUREHTF,
        "Classification": classification
    })

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1} of {len(df_2023)} rows")


results_df = pd.DataFrame(results)


# ============================================================
# Summary
# ============================================================

print("\n2023 performance classification summary")
print("---------------------------------------")

print("\nClassification counts:")
print(results_df["Classification"].value_counts())

print("\nEffEX statistics:")
print(results_df["Max_Predicted_EffEX"].describe())

print("\nHighest predicted EffEX:")
print(results_df.loc[results_df["Max_Predicted_EffEX"].idxmax()])

print("\nLowest predicted EffEX:")
print(results_df.loc[results_df["Max_Predicted_EffEX"].idxmin()])


# ============================================================
# Save outputs
# ============================================================

classification_file = os.path.join(
    output_dir,
    "effex_2023_suitability_classification_K1.csv"
)

summary_file = os.path.join(
    output_dir,
    "effex_2023_suitability_summary_K1.csv"
)

results_df.to_csv(classification_file, index=False)

summary_df = pd.DataFrame([
    {
        "Metric": "Total hours",
        "Value": len(results_df)
    },
    {
        "Metric": "Suitable hours",
        "Value": int((results_df["Classification"] == "Suitable").sum())
    },
    {
        "Metric": "Not suitable hours",
        "Value": int((results_df["Classification"] == "Not suitable").sum())
    },
    {
        "Metric": "Maximum EffEX",
        "Value": results_df["Max_Predicted_EffEX"].max()
    },
    {
        "Metric": "Mean EffEX",
        "Value": results_df["Max_Predicted_EffEX"].mean()
    },
    {
        "Metric": "Minimum EffEX",
        "Value": results_df["Max_Predicted_EffEX"].min()
    },
    {
        "Metric": "Suitability threshold",
        "Value": SUITABLE_THRESHOLD
    },
])

summary_df.to_csv(summary_file, index=False)

print("\nSaved files:")
print(classification_file)
print(summary_file)
