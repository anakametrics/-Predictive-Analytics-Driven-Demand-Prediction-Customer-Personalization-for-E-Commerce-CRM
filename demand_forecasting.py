import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


def run_forecast(path="data/sample_data.csv", product_name=None):

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    df = pd.read_csv(path)

    # Expected columns:
    # date, product, sales

    # -------------------------------
    # FILTER PRODUCT (optional)
    # -------------------------------
    if product_name:
        df = df[df["product"] == product_name]

    # -------------------------------
    # PREPARE DATA FOR PROPHET
    # -------------------------------
    df_prophet = df[["date", "sales"]].copy()
    df_prophet.columns = ["ds", "y"]

    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    # -------------------------------
    # MODEL
    # -------------------------------
    model = Prophet()
    model.fit(df_prophet)

    # -------------------------------
    # FUTURE PREDICTION
    # -------------------------------
    future = model.make_future_dataframe(periods=30)  # next 30 days
    forecast = model.predict(future)

    # -------------------------------
    # PLOT
    # -------------------------------
    fig = model.plot(forecast)

    # Save plot
    plt.savefig("outputs/forecast_plot.png")

    # -------------------------------
    # RETURN OUTPUT
    # -------------------------------
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
