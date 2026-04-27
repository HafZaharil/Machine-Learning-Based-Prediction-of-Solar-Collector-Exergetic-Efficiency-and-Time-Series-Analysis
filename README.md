# Machine-Learning-Based-Prediction-of-Solar-Collector-Exergetic-Efficiency-and-Time-Series-Analysis
Machine learning-based prediction of solar thermal system performance using operational inputs. This project develops a neural network model to estimate exergetic efficiency (EffEX) of a parabolic trough solar collector under varying operating and environmental conditions. Using cleaned 2023 meteorological data from PVGIS (Abu Dhabi), the model performs hourly optimisation to identify maximum achievable efficiency, followed by daily and monthly time-series analysis. Operating conditions are classified as suitable or not suitable based on a defined efficiency threshold of 35%, enabling full-year performance assessment without repeated thermodynamic simulations.

# Solar Thermal Performance Prediction and Time-Series Analysis (2023)

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
This produces a clearer time-series trend because only solar-active hours are included in the daily calculation.
