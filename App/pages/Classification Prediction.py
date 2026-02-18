import streamlit as st
import numpy as np
import joblib

st.title("Classification Prediction")
st.subheader("Next-Day Stock Direction Prediction")

st.markdown("""
This page allows you to **predict whether Amazon stock will go UP or DOWN**
on the next trading day using the trained Random Forest classifier.
""")

# Load model
model = joblib.load("/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/models/rf_classifier.joblib")

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

if st.button("Predict Direction"):
    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    if prediction == 1:
        st.success(f"Prediction: UP 📈 (Confidence: {prob[1]*100:.2f}%)")
    else:
        st.error(f"Prediction: DOWN 📉 (Confidence: {prob[0]*100:.2f}%)")

st.markdown("### How to Interpret This Prediction")

st.markdown("""
This classification model predicts **direction only**, not price magnitude.

**Prediction meanings:**
- **UP**  
  The model detects a short-term upward signal in the technical indicators.  
  This does **not** mean the price will rise significantly, only that upward movement is slightly more likely than downward.

- **DOWN**  
  The model detects a short-term downward signal.  
  This does **not** guarantee a loss, only that downward movement is slightly more likely.

**Confidence score:**
- Around **50 percent** means the model is uncertain  
- **55–60 percent** indicates a weak directional signal  
- Above **60 percent** is rare in financial markets and still not reliable on its own

**Important notes:**
- Stock direction prediction is close to random by nature
- Even small changes in input features can flip the prediction
- This model should be used for **educational analysis**, not trading decisions
""")

st.info(
    "Note: This prediction is for educational purposes only. "
    "Short-term stock direction is highly unpredictable."
)