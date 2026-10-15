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
│   └── train.csv
├── models/
│   └── sales_prediction_model.joblib
├── notebooks/
├── reports/
│   ├── actual_vs_predicted.png
│   ├── daily_sales.png
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

Generated model and report files are produced or updated by the project scripts. The dashboard expects the training data and prediction records to be present; model comparison results and the saved model are used when available.

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

## Train, compare, and run

From the project root, with the virtual environment active:

1. Train the model and create the saved model and test-period prediction records:

   ```powershell
   python .\src\model_training.py
   ```

2. Compare the Random Forest predictions with the previous-day baseline:

   ```powershell
   python .\src\compare_models.py
   ```

3. Start the Streamlit dashboard:

   ```powershell
   streamlit run .\app.py
   ```

The training script reads `data/train.csv` and writes `models/sales_prediction_model.joblib` and `reports/model_predictions.csv`. The comparison script reads those prediction records and writes `reports/model_comparison.csv`. The dashboard reads the training data and prediction records; it displays comparison results and feature importance when their corresponding files are available.

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
