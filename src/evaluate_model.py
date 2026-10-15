
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "reports" / "model_predictions.csv"
REPORTS_PATH = ROOT / "reports"

df = pd.read_csv(RESULTS_PATH, parse_dates=["date"])

# Overall evaluation
mae = mean_absolute_error(
    df["sales"],
    df["predicted_sales"]
)

print(f"Mean Absolute Error: {mae:.2f}")

# Plot daily actual vs predicted sales for the top-selling family
top_family = df.groupby("family")["sales"].sum().idxmax()
family_df = (
    df[df["family"] == top_family]
    .sort_values("date")
    .tail(90)
)

plt.figure(figsize=(13, 5))
plt.plot(
    family_df["date"],
    family_df["sales"],
    label="Actual Sales"
)
plt.plot(
    family_df["date"],
    family_df["predicted_sales"],
    label="Predicted Sales"
)

plt.title(f"Actual vs Predicted Sales: {top_family}")
plt.xlabel("Date")
plt.ylabel("Daily Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(REPORTS_PATH / "actual_vs_predicted.png")
plt.show()

print("Chart saved: reports/actual_vs_predicted.png")
