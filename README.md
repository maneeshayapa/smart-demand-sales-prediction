# Smart Demand & Sales Prediction System

An educational machine-learning project for exploring historical retail sales by product family. It prepares date, promotion, and historical-sales features, trains a Random Forest regression model, and presents historical results and an experimental forecast in a Streamlit dashboard.

> **Scope:** This project demonstrates analysis of the supplied historical dataset. It does not predict current real-world sales in 2026.

## Key features

- Historical retail sales analysis by product family.
- Random Forest sales prediction trained on historical data.
- Actual-versus-predicted sales visualization and recent prediction records.
- Baseline-versus-Random Forest comparison using a previous-day-sales baseline.
- Feature importance analysis for the trained Random Forest model.
- Experimental, recursive seven-day forecast based on the last available historical record.
- Interactive Streamlit dashboard with CSV downloads for prediction records and the seven-day forecast.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib and Seaborn (listed project dependencies)

## Project structure

```text
smart-demand-sales-prediction/
├── app.py
├── requirements.txt
├── data/
│   └── train.csv                 # Local training input; ignored by Git
├── models/
│   └── sales_prediction_model.joblib
├── notebooks/
├── reports/
│   ├── actual_vs_predicted.png
│   ├── daily_sales.png
│   ├── deployment_history.csv
│   ├── model_comparison.csv
│   ├── model_predictions.csv
│   └── top_product_families.png
└── src/
    ├── compare_models.py
    ├── data_analysis.py
    ├── evaluate_model.py
    ├── forecast.py
    ├── model_training.py
    └── sales_visualization.py
```

The raw Kaggle CSV is a local training input and is excluded from Git. The dashboard uses the compact `reports/deployment_history.csv` artifact, which contains complete daily sales and promotion totals by product family. Prediction records are required; model comparison results and feature importance are displayed when their corresponding artifacts are available.

## Installation and setup (Windows PowerShell)

Run these commands from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

If PowerShell blocks activation scripts, allow script activation for the current PowerShell process and activate the environment again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
```

## Generate artifacts and run locally

From the project root, with the virtual environment active:

1. Generate the compact dashboard history without retraining the model:

   ```powershell
   python .\src\model_training.py --history-only
   ```

   This reads `data/train.csv`, applies the same daily aggregation used for model training (sum `sales` and `onpromotion` by `date` and `family`), and writes `reports/deployment_history.csv`. The history is saved before lag processing removes early rows. Keep the raw Kaggle CSV local; it is intentionally ignored by Git.

2. To train or regenerate the model and test-period prediction records, run:

   ```powershell
   python .\src\model_training.py
   ```

   This also refreshes `reports/deployment_history.csv`. It retrains and saves the model, so do not run it just to prepare the deployment history.

3. Compare the Random Forest predictions with the previous-day baseline:

   ```powershell
   python .\src\compare_models.py
   ```

4. Start the Streamlit dashboard:

   ```powershell
   streamlit run .\app.py
   ```

The comparison script reads the test-period prediction records and writes `reports/model_comparison.csv`. The dashboard reads the compact history and prediction records; it displays model comparison and feature importance when their corresponding files are available. The dashboard does not treat the test-period prediction report as a substitute for complete historical data.

## Deploy to Streamlit Community Cloud

1. On your local checkout, create `reports/deployment_history.csv` using the `--history-only` command above. No model retraining is needed if the existing model and prediction artifacts are still valid.
2. Ensure the repository contains `app.py`, `src/`, `requirements.txt`, `reports/deployment_history.csv`, `reports/model_predictions.csv`, and `models/sales_prediction_model.joblib`. Include `reports/model_comparison.csv` to retain the model-comparison view. The raw `data/train.csv` is excluded by `.gitignore` and is not needed by the deployed dashboard or forecast.
3. Push the required project files to a GitHub repository. In Streamlit Community Cloud, create an app from that repository, select the intended branch, and set the main file path to `app.py`.
4. Deploy the app. If a required artifact is missing, the dashboard reports its expected path and the command needed to generate it.

## Evaluation metrics

The project reports two error metrics on historical test data:

- **Mean Absolute Error (MAE):** the average absolute difference between actual and predicted sales. Lower is better.
- **Root Mean Squared Error (RMSE):** the square root of the average squared prediction error, which gives larger errors more weight. Lower is better.

The current `reports/model_comparison.csv` records these results:

| Model | MAE | RMSE |
|---|---:|---:|
| Baseline (Previous Day) | 5275.53 | 16429.15 |
| Random Forest | 2642.11 | 9258.43 |

These are results recorded in the project's current report, not a guarantee of future performance. Evaluation uses a chronological historical test split. The lag features are created from the full time-ordered series before splitting, so lag-based test predictions may use actual previous-day sales from within the test period. This is not a fully recursive, real-world multi-day evaluation.

## Dataset

The project uses the **Store Sales - Time Series Forecasting** dataset from Kaggle:

[Kaggle competition data](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

The provided `data/train.csv` ends on **2017-08-15**. The dashboard's seven-day forecast therefore begins after the last date in this historical file; it is a demonstration using historical data, not a forecast of current 2026 sales.

## Limitations and future improvements

- The source data is historical and ends on 2017-08-15; it cannot support claims about current market conditions or 2026 sales.
- The forecast reuses the last observed promotion value for each of its seven steps, then feeds each predicted sales value into the following step. Future promotions and other external factors are not truly known by the current forecasting logic.
- The historical test evaluation can use actual preceding sales during the test period for lag features, so it does not represent a fully recursive forecast evaluation.
- The results are based on the supplied dataset and the current feature set and model configuration; they may not generalize to other stores, time periods, or conditions.
- Possible next steps include evaluating with a fully recursive backtest, incorporating known future calendar and promotion data, and testing additional forecasting approaches.

## Disclaimer

This project is provided for educational and portfolio demonstration purposes only. Its historical evaluation and experimental forecasts are not business, financial, or operational advice, and should not be relied on as real-world sales predictions.
