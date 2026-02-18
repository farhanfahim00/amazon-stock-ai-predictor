import streamlit as st

st.set_page_config(
    page_title="AI Stock Trend Predictor",
    layout="wide"
)

st.title("AI Stock Trend Predictor")
st.subheader("AMZN Stock Analysis and Assistance System")

st.markdown("""
Welcome to the AI Stock Trend Predictor.

This Streamlit application is a data-driven assistance system built for the Assistance Systems course.
It demonstrates how classical machine learning models behave on real financial time series data.

The system supports:
- Dataset exploration and validation
- Feature engineering inspection
- Classification and regression predictions
- Comparison of real vs augmented data training
- A chatbot assistant for guidance and questions

Important: This app is designed for learning and analysis, not for financial decision making.
""")

st.info(
    "Disclaimer: Stock markets are highly noisy and efficient. "
    "Predictions shown here are not guaranteed and should not be used for trading."
)

st.markdown("## What you can do in this app")

st.markdown("""
### 1) Explore Datasets
- View the real AMZN dataset with engineered features
- Explore the augmented dataset created by controlled noise injection
- Compare feature behavior between real and synthetic data

### 2) Run Machine Learning Models
This project includes supervised learning models for:

- Classification: Predict next day direction (Up or Down)
- Regression: Predict 1-day return
- Regression: Predict 5-day return
- Baseline: Ridge regression for comparison

Each model page explains the purpose, evaluation metrics, and limitations.

### 3) Make Interactive Predictions
On the prediction pages you can manually set technical indicators using sliders and:
- Predict direction using classification
- Predict future returns using regression

The app includes interpretation guides to help understand model outputs.

### 4) Compare Real vs Augmented Training
You can compare model behavior when trained on:
- Real market data
- Augmented synthetic data

This helps analyze robustness and sensitivity of models.

### 5) Use the Chatbot Assistant
A chatbot page is included for:
- Basic help navigating the app
- Explaining features and model outputs
- Answering domain questions about the dataset and indicators
""")

st.markdown("## Navigation")

st.write("""
Use the left sidebar to move between pages such as:
- Real Dataset Exploration  
- Augmented Dataset Exploration  
- Real Data Predictions (Classification and Regression)  
- Augmented Data Predictions  
- Chatbot Assistant  
""")

st.success("You can start by opening the dataset pages from the sidebar.")