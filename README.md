
# Bike Sharing Demand Regression (Numerical Optimization)

This repository presents a **complete numerical optimization–based regression study**
on the **Kaggle Bike Sharing Demand dataset**.

The objective is to model and predict **hourly bike rental demand** using different
feature mappings and to analyze how model expressiveness affects generalization
performance.

Dataset source:
https://www.kaggle.com/competitions/bike-sharing-demand

---

## Problem Statement

Given historical hourly data describing temporal and environmental conditions,
predict the total number of bike rentals (`count`) for each hour.

Formally, this is a **supervised regression problem** solved via
**least-squares optimization**:

min_θ (1/n) Σ (yᵢ − Xᵢθ)²

All models in this project estimate parameters using the **Normal Equation**:

θ = (XᵀX)⁻¹Xᵀy

---

## Feature Engineering

From the `datetime` column:
- year, month, day, hour
- day of week, day of year

Cyclical encoding:
- sin/cos of hour
- sin/cos of month

Binary indicators:
- is_weekend
- is_rush_hour

Leakage prevention:
- Removed `casual` and `registered` columns

All features are standardized using **z-score normalization** based only on
training data.

---

## Models Implemented

1. Linear Regression (baseline)
2. Polynomial Regression (degrees 2, 3, 4 — no interaction terms)
3. Quadratic Regression with Interaction Terms (xᵢ² and xᵢxⱼ)

Each model is trained using a closed-form solution via the Normal Equation.

---

## Experimental Setup

- Train–test split: 80 / 20
- Fixed random seed for reproducibility
- Evaluation metrics:
  - MSE
  - RMSE
  - MAE
  - R² score

---

## Results (Test Set)

| Model | Test MSE | Test RMSE | Test R² |
|------|---------|-----------|---------|
| Linear Regression | 11627.96 | 107.83 | 0.6479 |
| Polynomial (d=4) | 10515.90 | 102.55 | 0.6815 |
| Quadratic + Interactions | **4564.13** | **67.56** | **0.8618** |

The **Quadratic + Interactions** model significantly outperforms simpler models
while maintaining close train–test R², indicating good generalization.

---

## Visual Analysis

- Model comparison across MSE, RMSE, and R²
- Train vs Test R² to assess overfitting
- Predicted vs Actual plots with y = x reference

All figures are available in the `figures/` directory.

---

## How to Run

```bash
pip install numpy pandas matplotlib
python src/bike_sharing_regression.py
```

---

## Key Takeaways

- Bike-sharing demand exhibits strong non-linear and interaction effects
- Polynomial features alone are insufficient
- Explicit interaction terms dramatically improve performance
- Feature engineering can outperform more complex models

---

## Author

Parth Malhotra
