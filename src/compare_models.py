
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "train.csv"
PREDICTIONS_PATH = ROOT / "reports" / "model_predictions.csv"
OUTPUT_PATH = ROOT / "reports" / "model_comparison.csv"


# Load historical sales data and model predictions
raw_data = pd.read_csv(DATA_PATH, parse_dates=["date"])
predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])

# Aggregate sales by date and product family
data = (
    raw_data.groupby(["date", "family"], as_index=False)
    .agg(
        sales=("sales", "sum"),
        onpromotion=("onpromotion", "sum"),
    )
)

# Baseline: use the previous day's sales as the prediction
data = data.sort_values(["family", "date"]).copy()

data["baseline_predicted_sales"] = (
    data.groupby("family")["sales"].shift(1)
)

# Match baseline predictions with the Random Forest test predictions
baseline = data[
    ["date", "family", "baseline_predicted_sales"]
]

comparison = predictions.merge(
    baseline,
    on=["date", "family"],
    how="left",
)

comparison = comparison.dropna(
    subset=["sales", "predicted_sales", "baseline_predicted_sales"]
)

# Calculate evaluation metrics
actual = comparison["sales"]
random_forest = comparison["predicted_sales"]
baseline_predictions = comparison["baseline_predicted_sales"]

results = pd.DataFrame(
    {
        "Model": ["Baseline (Previous Day)", "Random Forest"],
        "MAE": [
            mean_absolute_error(actual, baseline_predictions),
            mean_absolute_error(actual, random_forest),
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(actual, baseline_predictions)),
            np.sqrt(mean_squared_error(actual, random_forest)),
        ],
    }
)

results["MAE"] = results["MAE"].round(2)
results["RMSE"] = results["RMSE"].round(2)

print("\n========== MODEL COMPARISON ==========")
print(results.to_string(index=False))

results.to_csv(OUTPUT_PATH, index=False)
print(f"\nComparison saved to: {OUTPUT_PATH}")
