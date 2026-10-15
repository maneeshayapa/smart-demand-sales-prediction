
import pandas as pd
import joblib

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "models" / "sales_prediction_model.joblib"


def forecast_next_7_days(product_family):
    # Load historical data and trained model
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Dataset not found at expected path '{DATA_PATH}': {error}"
        ) from error

    try:
        saved = joblib.load(MODEL_PATH)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Saved model not found at expected path '{MODEL_PATH}': {error}"
        ) from error
    model = saved["model"]
    features = saved["features"]

    # Aggregate historical sales by date and product family
    data = (
        df.groupby(["date", "family"], as_index=False)
        .agg(
            sales=("sales", "sum"),
            onpromotion=("onpromotion", "sum")
        )
        .sort_values("date")
    )

    family_data = (
        data[data["family"] == product_family]
        .sort_values("date")
        .copy()
    )

    if family_data.empty:
        raise ValueError("Product family not found.")

    history = family_data["sales"].tolist()
    last_date = family_data["date"].max()

    forecasts = []

    # Recursive 7-day forecast
    for step in range(1, 8):
        forecast_date = last_date + pd.Timedelta(days=step)

        # Use the latest available promotion value as an assumption
        promotion = int(family_data["onpromotion"].iloc[-1])

        row = {
            "family": product_family,
            "onpromotion": promotion,
            "year": forecast_date.year,
            "month": forecast_date.month,
            "day": forecast_date.day,
            "day_of_week": forecast_date.dayofweek,
            "day_of_year": forecast_date.dayofyear,
            "sales_lag_1": history[-1],
            "sales_lag_7": history[-7],
            "sales_lag_14": history[-14],
            "sales_rolling_7": sum(history[-7:]) / 7,
        }

        X = pd.DataFrame([row])[features]
        predicted_sales = max(0.0, float(model.predict(X)[0]))

        forecasts.append({
            "date": forecast_date,
            "family": product_family,
            "predicted_sales": round(predicted_sales, 2),
        })

        # Feed prediction into the next forecast step
        history.append(predicted_sales)

    return pd.DataFrame(forecasts)


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    families = sorted(df["family"].unique())
    selected_family = families[0]

    result = forecast_next_7_days(selected_family)

    print(f"\n7-Day Forecast: {selected_family}")
    print(result.to_string(index=False))
