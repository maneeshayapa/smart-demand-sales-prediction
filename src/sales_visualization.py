
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
REPORTS_PATH = PROJECT_ROOT / "reports"

REPORTS_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

# Total sales by date
daily_sales = df.groupby("date")["sales"].sum()

plt.figure(figsize=(12, 5))
plt.plot(daily_sales.index, daily_sales.values)
plt.title("Daily Total Sales")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(REPORTS_PATH / "daily_sales.png")
plt.show()

# Top 10 product families by sales
family_sales = (
    df.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
family_sales.sort_values().plot(kind="barh")
plt.title("Top 10 Product Families by Total Sales")
plt.xlabel("Total Sales")
plt.tight_layout()

plt.savefig(REPORTS_PATH / "top_product_families.png")
plt.show()

print("Charts saved successfully in the reports folder.")