
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

print("Original dataset:", df.shape)

# --------------------------------------------------
# 3. Aggregate sales by date and product family
# --------------------------------------------------
data = (
    df.groupby(["date", "family"], as_index=False)
    .agg(
        sales=("sales", "sum"),
        onpromotion=("onpromotion", "sum")
    )
    .sort_values(["family", "date"])
    .reset_index(drop=True)
)

# --------------------------------------------------
# 4. Feature engineering
# --------------------------------------------------
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day
data["day_of_week"] = data["date"].dt.dayofweek
data["day_of_year"] = data["date"].dt.dayofyear

# Previous sales for each product family
for lag in [1, 7, 14]:
    data[f"sales_lag_{lag}"] = (
        data.groupby("family")["sales"].shift(lag)
    )

# Rolling average using only previous days
data["sales_rolling_7"] = (
    data.groupby("family")["sales"]
    .transform(lambda s: s.shift(1).rolling(7).mean())
)

# Remove rows without enough historical sales
data = data.dropna().reset_index(drop=True)

# --------------------------------------------------
# 5. Chronological train/test split
# --------------------------------------------------
data = data.sort_values("date").reset_index(drop=True)

split_date = data["date"].quantile(0.8)

train = data[data["date"] < split_date].copy()
test = data[data["date"] >= split_date].copy()

features = [
    "family",
    "onpromotion",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_rolling_7",
]

X_train = train[features]
y_train = train["sales"]

X_test = test[features]
y_test = test["sales"]

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Training cutoff date:", split_date)

# --------------------------------------------------
# 6. Build and train model
# --------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "family_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            ["family"]
        )
    ],
    remainder="passthrough"
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=18,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("\nTraining model...")
model.fit(X_train, y_train)

# --------------------------------------------------
# 7. Evaluate model
# --------------------------------------------------
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n--- MODEL PERFORMANCE ---")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# --------------------------------------------------
# 8. Save model and feature information
# --------------------------------------------------
model_path = MODEL_DIR / "sales_prediction_model.joblib"
joblib.dump(
    {
        "model": model,
        "features": features,
    },
    model_path
)

print("\nModel saved to:", model_path)

# Save sample predictions for inspection
results = test[["date", "family", "sales"]].copy()
results["predicted_sales"] = predictions
results.to_csv(ROOT / "reports" / "model_predictions.csv", index=False)

print("Sample predictions saved in reports/model_predictions.csv")
print("\nTraining and evaluation completed!")
