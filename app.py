
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from src.forecast import forecast_next_7_days

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
RESULTS_PATH = ROOT / "reports" / "model_predictions.csv"
MODEL_COMPARISON_PATH = ROOT / "reports" / "model_comparison.csv"
MODEL_PATH = ROOT / "models" / "sales_prediction_model.joblib"

st.set_page_config(
    page_title="Smart Sales Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Smart Demand & Sales Prediction System")
st.write(
    "Explore historical retail sales and review "
    "Machine Learning predictions."
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df

@st.cache_data
def load_predictions():
    return pd.read_csv(RESULTS_PATH, parse_dates=["date"])

@st.cache_data
def load_model_comparison():
    return pd.read_csv(MODEL_COMPARISON_PATH)

@st.cache_resource
def load_saved_model():
    return joblib.load(MODEL_PATH)

try:
    df = load_data()
    predictions = load_predictions()

    # Sidebar filters
    st.sidebar.header("Filters")

    families = sorted(df["family"].dropna().unique())
    selected_family = st.sidebar.selectbox(
        "Product family",
        families
    )

    # Historical sales for selected family
    family_data = df[df["family"] == selected_family]

    daily_sales = (
        family_data.groupby("date")["sales"]
        .sum()
        .sort_index()
    )

    total_sales = daily_sales.sum()
    average_sales = daily_sales.mean()
    best_day = daily_sales.idxmax()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"{total_sales:,.0f}")
    col2.metric("Average Daily Sales", f"{average_sales:,.2f}")
    col3.metric("Best Sales Day", best_day.strftime("%Y-%m-%d"))

    st.subheader("Historical Daily Sales")
    st.line_chart(daily_sales)

    st.subheader("Actual vs Predicted Sales")

    selected_predictions = (
        predictions[predictions["family"] == selected_family]
        .sort_values("date")
        .tail(90)
        .set_index("date")
    )

    if not selected_predictions.empty:
        chart_data = selected_predictions[
            ["sales", "predicted_sales"]
        ].rename(
            columns={
                "sales": "Actual Sales",
                "predicted_sales": "Predicted Sales"
            }
        )

        st.line_chart(chart_data)
    else:
        st.info("No predictions available for this product family.")

    st.subheader("Recent Prediction Records")
    st.dataframe(
        selected_predictions.reset_index().tail(20),
        use_container_width=True
    )

    st.subheader("Model Performance Comparison")
    try:
        model_comparison = load_model_comparison()
    except FileNotFoundError:
        st.info("Model comparison results are not available.")
    else:
        if model_comparison.empty:
            st.info("No model comparison results are available.")
        else:
            st.dataframe(model_comparison, use_container_width=True)

            mae_model = model_comparison.loc[
                model_comparison["MAE"].idxmin(), "Model"
            ]
            rmse_model = model_comparison.loc[
                model_comparison["RMSE"].idxmin(), "Model"
            ]

            mae_col, rmse_col = st.columns(2)
            mae_col.metric(
                "Lowest MAE",
                f"{mae_model} ({model_comparison['MAE'].min():,.2f})"
            )
            mae_col.bar_chart(
                model_comparison.set_index("Model")[["MAE"]]
            )
            rmse_col.metric(
                "Lowest RMSE",
                f"{rmse_model} ({model_comparison['RMSE'].min():,.2f})"
            )
            rmse_col.bar_chart(
                model_comparison.set_index("Model")[["RMSE"]]
            )

            st.caption(
                "Lower MAE and RMSE indicate better predictions on the "
                "evaluated test data."
            )

    st.subheader("Feature Importance Analysis")
    st.caption(
        "Feature importance shows how useful features were to the trained "
        "model, but it does not prove that they cause changes in sales."
    )

    try:
        saved_model = load_saved_model()
    except FileNotFoundError:
        st.info("The trained model is not available for feature importance analysis.")
    else:
        if (
            not isinstance(saved_model, dict)
            or "model" not in saved_model
            or "features" not in saved_model
        ):
            st.info("The saved model file has an unsupported format.")
        else:
            pipeline = saved_model["model"]
            if not isinstance(pipeline, Pipeline):
                st.info("The saved model uses an unsupported pipeline structure.")
            elif (
                "preprocessor" not in pipeline.named_steps
                or "regressor" not in pipeline.named_steps
                or not isinstance(
                    pipeline.named_steps["preprocessor"], ColumnTransformer
                )
                or not isinstance(
                    pipeline.named_steps["regressor"], RandomForestRegressor
                )
            ):
                st.info("The saved model uses an unsupported pipeline structure.")
            else:
                preprocessor = pipeline.named_steps["preprocessor"]
                regressor = pipeline.named_steps["regressor"]
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    importances = regressor.feature_importances_
                    if len(feature_names) != len(importances):
                        raise ValueError(
                            "Transformed feature names do not match the "
                            "model's feature importances."
                        )
                except (AttributeError, TypeError, ValueError) as error:
                    st.info(
                        "Feature importance is unavailable for this saved "
                        f"model: {error}"
                    )
                else:
                    importance_data = (
                        pd.DataFrame(
                            {
                                "Feature": feature_names,
                                "Importance": importances,
                            }
                        )
                        .nlargest(15, "Importance")
                        .sort_values("Importance", ascending=False)
                    )
                    st.bar_chart(
                        importance_data.set_index("Feature")["Importance"]
                    )
                    st.dataframe(importance_data, use_container_width=True)

    # 7-Day Future Forecast
    st.subheader("🔮 Next 7 Days Sales Forecast")

    try:
        forecast_df = forecast_next_7_days(selected_family)

        st.caption(
            "Experimental forecast based on historical data. "
            "Future promotions and other external factors are assumed."
        )

        st.line_chart(
            forecast_df.set_index("date")["predicted_sales"]
        )

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        st.download_button(
            "Download 7-Day Forecast",
            data=forecast_df.to_csv(index=False),
            file_name="seven_day_sales_forecast.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Could not generate forecast: {error}")


    st.download_button(
        "Download Prediction CSV",
        data=selected_predictions.reset_index().to_csv(index=False),
        file_name="sales_predictions.csv",
        mime="text/csv"
    )

except FileNotFoundError:
    st.error(
        "Required files were not found. Please run "
        "python src/model_training.py first."
    )
