import streamlit as st
import numpy as np
import joblib

st.title("Regression Prediction")
st.subheader("Future Return Estimation")

st.markdown("""
This page predicts **future stock returns** using trained regression models.
You can choose between **1-day** and **5-day** return forecasts.
""")

# Load models
model_1d = joblib.load("/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/models/rf_1day_regressor.joblib")
model_5d = joblib.load("/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/models/rf_5day_regressor.joblib")

st.markdown("### Select Prediction Horizon")
horizon = st.radio("Prediction Type", ["1-Day Return", "5-Day Return"])

st.markdown("### Input Technical Indicators")

features = {
    "Daily_Return": st.slider("Daily Return", -0.1, 0.1, 0.0),
    "Price_Change_Pct": st.slider("Price Change %", -5.0, 5.0, 0.0),
    "Volatility": st.slider("Volatility", 0.0, 10.0, 2.0),
    "Volume_Change": st.slider("Volume Change", -1.0, 1.0, 0.0),
    "Momentum": st.slider("Momentum", -10.0, 10.0, 0.0),
    "RSI": st.slider("RSI", 0.0, 100.0, 50.0),
    "MACD": st.slider("MACD", -5.0, 5.0, 0.0),
    "MACD_Diff": st.slider("MACD Diff", -2.0, 2.0, 0.0),
    "BB_Position": st.slider("Bollinger Band Position", 0.0, 1.0, 0.5),
    "Volume_Ratio": st.slider("Volume Ratio", 0.0, 3.0, 1.0),
    "ROC": st.slider("Rate of Change", -10.0, 10.0, 0.0),
    "ATR": st.slider("ATR", 0.0, 10.0, 1.0),
}

X = np.array(list(features.values())).reshape(1, -1)

if st.button("Predict Return"):
    if horizon == "1-Day Return":
        pred = model_1d.predict(X)[0]
        st.success(f"Predicted 1-Day Return: {pred*100:.2f}%")
    else:
        pred = model_5d.predict(X)[0]
        st.success(f"Predicted 5-Day Return: {pred*100:.2f}%")

st.markdown("### How to Interpret This Prediction")

st.markdown("""
The regression model predicts **expected percentage return**, not exact future prices.

**Return scale guide:**

- **Below −1.0 percent**  
  Strong negative expectation  
  Indicates unfavorable short-term conditions

- **Between −1.0 percent and −0.2 percent**  
  Mild negative expectation  
  Small potential downside

- **Between −0.2 percent and +0.2 percent**  
  Neutral zone  
  No meaningful directional signal

- **Between +0.2 percent and +1.0 percent**  
  Mild positive expectation  
  Weak upward momentum

- **Above +1.0 percent**  
  Strong positive expectation  
  Rare in short-term stock prediction and often unstable

**Important notes:**
- Predicted returns are averages learned from historical patterns
- Real markets often behave differently than historical data
- Larger absolute values usually come with higher uncertainty
""")

st.info(
    "Regression outputs represent expected returns, not guaranteed outcomes. "
    "Negative values indicate potential losses."
)