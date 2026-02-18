import streamlit as st
import pandas as pd

st.title("Data Overview")
st.subheader("Amazon (AMZN) Stock Dataset")

st.markdown("""
This page describes the **Amazon (AMZN) stock dataset** used in the AI Stock Trend Predictor.

**Data sources**
- Kaggle dataset (historical AMZN prices)
- Yahoo Finance API (validation and completeness checks)

**Time range**
- 1997 to 2025

**Goal**
- Provide transparent dataset understanding before running predictions and model evaluation.
""")

@st.cache_data
def load_data():
    return pd.read_csv(
        "/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/Data/AMZN_data_with_features.csv"
    )

df = load_data()

st.divider()

st.markdown("### Dataset Preview")
st.dataframe(df.head())

st.markdown("### Dataset Size")
c1, c2 = st.columns(2)
with c1:
    st.metric("Rows (trading days)", df.shape[0])
with c2:
    st.metric("Columns (features + targets)", df.shape[1])

st.divider()

st.markdown("## Feature List and Explanation")

feature_groups = {
    "Raw Market Columns": {
        "Open": "Opening price at the start of the trading day.",
        "High": "Highest price reached during the trading day.",
        "Low": "Lowest price reached during the trading day.",
        "Close": "Closing price at the end of the trading day.",
        "Volume": "Number of shares traded that day.",
        "Dividends": "Dividend payout value (mostly zero for AMZN historically).",
        "Stock Splits": "Split ratio information for share adjustments."
    },
    "Return and Price Movement": {
        "Daily_Return": "Percent change of Close compared to previous Close.",
        "Price_Change": "Absolute intraday move: Close minus Open.",
        "Price_Change_Pct": "Percent intraday move: (Close minus Open) relative to Open."
    },
    "Trend and Moving Averages": {
        "MA_5": "5 day simple moving average of Close, short trend smoothing.",
        "MA_10": "10 day moving average of Close, medium trend smoothing.",
        "MA_20": "20 day moving average of Close, common baseline trend indicator."
    },
    "Volatility and Range": {
        "Volatility": "Daily range: High minus Low, measures price fluctuation.",
        "ATR": "Average True Range (14 day), measures volatility including gaps."
    },
    "Volume Indicators": {
        "Volume_Change": "Percent change in Volume compared to previous day.",
        "Volume_MA_5": "5 day moving average of volume, baseline activity level.",
        "Volume_Ratio": "Current volume divided by Volume_MA_5, shows unusual activity."
    },
    "Momentum Indicators": {
        "Momentum": "Short momentum: Close today minus Close 5 days ago.",
        "ROC": "Rate of Change (10 day), percent price change over last 10 days."
    },
    "Oscillators and Advanced Technical Indicators": {
        "RSI": "Relative Strength Index (14 day), overbought or oversold signal.",
        "MACD": "Trend momentum signal from EMA(12) minus EMA(26).",
        "MACD_Signal": "EMA(9) of MACD, used as confirmation line.",
        "MACD_Diff": "MACD minus MACD_Signal, histogram like momentum strength."
    },
    "Bollinger Bands": {
        "BB_Middle": "20 day moving average (center band).",
        "BB_Std": "20 day standard deviation of Close, volatility base.",
        "BB_Upper": "Upper band: BB_Middle plus 2 times BB_Std.",
        "BB_Lower": "Lower band: BB_Middle minus 2 times BB_Std.",
        "BB_Position": "Normalized band position: 0 near lower band, 1 near upper band."
    },
    "Targets (Supervised Learning Labels)": {
        "Target": "Classification label: 1 if next day Close is higher, else 0."
    }
}

for group, items in feature_groups.items():
    with st.expander(group, expanded=False):
        for k, v in items.items():
            st.markdown(f"- **{k}**: {v}")

st.divider()

st.markdown("## Visualizations")

viz_options = {
    "Amazon Stock Price Over Time": "/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/visualizations/chart1_stock_price.png",
    "Trading Volume Over Time": "/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/visualizations/chart2_volume.png",
    "Daily Returns Distribution": "/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/visualizations/chart3_returns.png",
    "RSI Indicator Over Time": "/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/visualizations/chart4_rsi.png",
    "Target Distribution (Up vs Down)": "/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/visualizations/chart5_target.png",
    "Feature Correlation Matrix": "/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/visualizations/chart6_correlation.png"
}

selected_viz = st.selectbox("Select a visualization to display", list(viz_options.keys()))
st.image(viz_options[selected_viz], caption=selected_viz, use_container_width=True)

st.info(
    "All engineered features follow strict time-series ordering. "
    "Targets are created using shift operations to avoid data leakage."
)