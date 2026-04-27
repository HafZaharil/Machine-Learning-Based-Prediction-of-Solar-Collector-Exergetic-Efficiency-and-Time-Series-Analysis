# Machine-Learning-Based-Prediction-of-Solar-Collector-Exergetic-Efficiency-and-Time-Series-Analysis
Machine learning-based prediction of solar thermal system performance using operational inputs. This project develops a neural network model to estimate exergetic efficiency (EffEX) of a parabolic trough solar collector under varying operating and environmental conditions. Using cleaned 2023 meteorological data from PVGIS (Abu Dhabi), the model performs hourly optimisation to identify maximum achievable efficiency, followed by daily and monthly time-series analysis. Operating conditions are classified as suitable or not suitable based on a defined efficiency threshold of 35%, enabling full-year performance assessment without repeated thermodynamic simulations.

# Solar Thermal Performance Prediction and Time-Series Analysis

A machine learning-based project for predicting and analysing Parabolic Trough Solar Collector (PTSC) performance using a trained neural network and full-year 2023 meteorological data.

The workflow combines:
- trained ML model  
- real-world solar data  
- operating-point optimisation  
- time-series performance analysis  

---

## Project Overview

This project evaluates PTSC performance across an entire year (2023) by predicting the maximum achievable exergetic efficiency (EffEX) at each hour.

For each timestep:
- meteorological inputs are provided  
- the model searches for optimal operating conditions  
- the maximum EffEX is extracted  
- performance is classified as suitable or not suitable  

---

## Data Source and Preprocessing

The 2023 dataset was obtained from:

- **PVGIS (Photovoltaic Geographical Information System)**  
- Location: **Abu Dhabi region**

### Data Processing Steps

The raw PVGIS dataset was cleaned and processed before use:

- Invalid and inconsistent rows removed  
- Time series sorted and standardised  
- Variables converted into model-ready format  

### Final Inputs Used

The dataset was simplified for modelling:

- **DNI** — Direct Normal Irradiance  
- **Tamb** — Ambient Temperature  

Tamb values are stored in °C and converted to Kelvin (K) internally during model evaluation.

---

## Trained Model Files

The prediction pipeline uses pre-trained neural network components:

```text
effex_nn_weights_and_biases.pth
effex_input_scaler.pkl
effex_feature_cols.pkl
```
### Description

* effex_nn_weights_and_biases.pth — saved neural network weights and biases
* effex_input_scaler.pkl — saved input scaler used during model training
* effex_feature_cols.pkl — saved input feature order used by the model

These files allow the trained EffEX model to be loaded and used directly for prediction without retraining.

## Python Code File

The main prediction and classification workflow is implemented in:
Classification.py

##Code Workflow

The script performs the following steps:

1. Load cleaned 2023 PVGIS meteorological data
2. Load the saved EffEX neural network files
3. Reconstruct the neural network architecture
4. Load the saved weights and biases
5. Convert Tamb from °C to K for model calculation
6. Use DNI and Tamb for each hourly timestep
7. Search across Tin and Mhtf to find the maximum predicted EffEX
8. Classify each hour as suitable or not suitable
9. Export the hourly classification result
10. Calculate daily average EffEX using DNI > 0 hours only
11. Calculate monthly average EffEX
12. Generate daily and monthly time-series plots

Model Input Features

The trained surrogate EffEX model uses the following input features:

* Tin — Inlet temperature
* DNI — Direct normal irradiance
* Mhtf — Heat transfer fluid mass flow rate
* Tamb — Ambient temperature
* Pressurehtf — Heat transfer fluid pressure
* K — Incident angle factor


```
The input structure is:
(Tin, DNI, Mhtf, Tamb, Pressurehtf, K)
                ↓
          Predicted EffEX
```

# Neural Network Architecture

The trained neural network uses a fully connected feedforward structure:
```
Input → 128 → 64 → 32 → 16 → Output
```

## Model Settings

* Activation function: ReLU
* Output: EffEX
* Input scaling: StandardScaler
* Saved model format: PyTorch .pth

The neural network is not retrained in the classification script. The same architecture is rebuilt only so that the saved weights and biases can be loaded correctly.

Operating Assumptions

For the 2023 time-series prediction workflow, the following assumptions are used:

* Pressurehtf = 20000 kPa
* K = 1

For each hourly timestep, the model searches across:

* Tin (K)
* Mhtf (Mhtf)

to find the highest predicted EffEX for that specific DNI and Tamb condition.

** Suitability Classification

Each hourly timestep is classified based on the maximum predicted EffEX.

The classification rule is:
```
EffEX > 35%  → Suitable
EffEX ≤ 35%  → Not suitable
```
The values above are based on the academic litereature that exergetic efficiency of >35% is generally suitable for power generation.

This threshold is used to identify whether the operating condition provides acceptable exergetic performance.

## Performance Summary

| Metric | Value |
|---|---:|
| Total hours | 8760 |
| Suitable hours | 3368 |
| Not suitable hours | 5392 |
| Maximum EffEX (%) | 45.9518 |
| Mean EffEX (%) | 26.8797 |
| Minimum EffEX (%) | 11.0318 |
| Suitability threshold (%) | 35.0 |

## Time-Series Analysis Method

Daily average EffEX is calculated using only hours where:
```
DNI > 0
```
This avoids distorting the daily average with night-time or zero-solar-input periods.


### Daily Average Formula

```text
Daily average EffEX =
sum of EffEX during DNI > 0 hours
/
number of DNI > 0 hours
```
This produces a clearer time-series trend because only solar-active hours are included in the daily calculation.

### Daily EffEX Time-Series Plot

Plot Description

This figure shows the daily average predicted EffEX across 2023.

The plot includes:

* Daily average EffEX
* 7-day rolling average
* 35% suitability threshold

The 7-day rolling average smooths short-term fluctuations and makes the annual trend easier to interpret.

### Monthly EffEX Time-Series Plot

Plot Description

This figure shows the monthly average predicted EffEX for 2023.

Monthly values are calculated from daily average EffEX values based on DNI > 0 hours only.

The plot provides a clearer view of longer-term performance variation across the year.


# Results and Discussion
The full-year hourly classification produced:

```
Suitable hours = 3368
Not suitable hours = 5392
```

The not suitable hours include low-irradiance and night-time conditions.

When the analysis focuses on DNI > 0 hours, the system performance becomes more representative of active solar operation.

The daily and monthly time-series results show that the system generally maintains EffEX values close to or above the suitability threshold during meaningful solar availability periods.

This indicates that low hourly suitability is largely influenced by solar availability rather than poor collector performance during active operation.

<img width="7200" height="3600" alt="monthly_average_effex_2023_nonzero_dni_hours_only" src="https://github.com/user-attachments/assets/a51a2abd-a409-499d-8277-95a629a7c3fc" />
<img width="8400" height="3600" alt="daily_average_effex_2023_with_7day_rolling_average" src="https://github.com/user-attachments/assets/3fa31620-7062-40fc-b85a-ee52b3a2f144" />

The daily and monthly results show that the PTSC system maintains generally strong predicted EffEX performance across 2023 when only DNI > 0 hours are considered.

The daily time-series plot shows short-term fluctuations caused by changes in solar conditions, but the 7-day rolling average remains mostly between 38% and 43%. This indicates that the system performs consistently above the 35% suitability threshold for most solar-active periods. A few sharp daily drops appear around March, August, and late October, likely linked to low-DNI or poor-weather days.

The monthly plot gives a clearer overall trend. All months remain above the 35% threshold, meaning the system is suitable on a monthly-average basis throughout the year. February shows the highest average EffEX, while July is the lowest month, although it still remains above the suitability line.

Overall, the results suggest that the system is not weak across 2023. Performance dips occur on specific days, but the broader trend remains stable and suitable during meaningful solar-operating hours.

# Key Contribution

This project demonstrates a complete machine learning workflow for solar thermal performance prediction:

```
Cleaned PVGIS data
        ↓
Trained EffEX neural network
        ↓
Hourly operating-point search
        ↓
Maximum EffEX prediction
        ↓
Suitability classification
        ↓
Daily and monthly time-series analysis
```

# Applications

This project can support:

* Solar thermal performance prediction
* Full-year operating-condition analysis
* Suitability classification
* Time-series performance assessment
* ML-assisted thermodynamic system evaluation
* Rapid screening of solar collector operating conditions

# Repository Structure
```
├── README.md
├── LICENSE
├── .gitignore
│
├── code/
│   └── Classification.py
│
├── models/
│   ├── effex_nn_weights_and_biases.pth
│   ├── effex_input_scaler.pkl
│   └── effex_feature_cols.pkl
│
├── data/
│   ├── effex_2023_suitability_classification_K1.csv
│   └── effex_2023_suitability_summary_K1.csv
│
├── figures/
│   ├── daily_average_effex_2023_with_7day_rolling_average.jpeg
│   └── monthly_average_effex_2023_nonzero_dni_hours_only.png
│
└── results/
    ├── daily_average_effex_2023_nonzero_dni_hours_only.csv
    └── monthly_average_effex_2023_nonzero_dni_hours_only.csv
```

# Intended Use

This project is intended as a machine learning-based performance prediction and time-series analysis workflow for PTSC systems.

It is designed for:

* Fast annual analysis
* Operational screening
* Data-driven performance evaluation
* Research demonstration
* GitHub portfolio presentation

  

