import streamlit as st
import pandas as pd
import plotly.express as px
import importlib.util
import sys
import os

# Page configuration
st.set_page_config(page_title="Amazon Stock Predictor", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    [
        "Home",
        "Upload Your CSV",
        "Predict Up/Down (Classification)",
        "Predict Return (Regression)",
        "5-Day Forecast",
        "Data Visualization Dashboard",
        "Chatbot"
    ]
)

# Home page
if page == "Home":
    st.title("Amazon Stock Prediction Project")
    st.subheader("Machine Learning Pipeline for Time-Series Forecasting")

    st.write(
        """
        This application demonstrates the full workflow for building a 
        time-series stock predictor.

        - Data cleaning and feature engineering  
        - Leakage-free target creation  
        - Time-series aware model training  
        - Performance evaluation of Random Forest  
        """
    )

    st.markdown("---")

    data = pd.read_csv("Data/AMZN_with_features.csv", index_col="Date", parse_dates=True)

    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(data))
    col2.metric("Date Range", f"{data.index.min().date()} → {data.index.max().date()}")
    col3.metric("Total Features", len(data.columns))

    st.write("Preview of the engineered dataset:")
    st.dataframe(data.head())

    st.markdown("---")
    st.header("Stock Price Visualization")
    fig = px.line(data, x=data.index, y="Close", title="Amazon Close Price Over Time")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("5-Day Moving Average (MA5)")
    fig2 = px.line(data, x=data.index, y="MA_5")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.header("Model Performance Summary")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Classification Model")
        st.write(
            """
            - Accuracy: 51.43%  
            - Precision: 52.05%  
            - Recall: 78.95%  
            - Cross-Validation Accuracy: 50.28%  
            """
        )
    with col_b:
        st.subheader("Regression Model")
        st.write(
            """
            - R²: -0.0057  
            - MAE: 3.50%  
            - Directional Accuracy: 53.98%  
            - CV R²: -0.2126  
            """
        )

    st.markdown("---")
    st.header("Project Features")
    st.write(
        """
        - Advanced feature engineering (RSI, MACD, Bollinger Bands, ATR)
        - Classification model for trend direction prediction  
        - Regression model for 5-day return prediction  
        - Interactive data visualization dashboard  
        - Voice-enabled chatbot for stock insights
        """
    )

# Upload Your CSV Page
elif page == "Upload Your CSV":
    st.title("Upload Your CSV")
    st.write("Content coming soon.")

# Classification Prediction Page
elif page == "Predict Up/Down (Classification)":
    st.title("Predict Up/Down (Classification Model)")
    st.write("Content coming soon.")

# Regression Prediction Page
elif page == "Predict Return (Regression)":
    st.title("Predict Next-Day Return (Regression Model)")
    st.write("Content coming soon.")

# 5-Day Forecast Page
elif page == "5-Day Forecast":
    st.title("5-Day Stock Forecasting")
    st.write("Content coming soon.")

# Data Visualization Dashboard
elif page == "Data Visualization Dashboard":
    st.title("Data Visualization Dashboard")
    st.write("Content coming soon.")

# Chatbot page
elif page == "Chatbot":
    chatbot_path = os.path.join(os.path.dirname(__file__), "chatbot_page.py")
    if os.path.exists(chatbot_path):
        spec = importlib.util.spec_from_file_location("chatbot_page", chatbot_path)
        chatbot_module = importlib.util.module_from_spec(spec)
        sys.modules["chatbot_page"] = chatbot_module
        spec.loader.exec_module(chatbot_module)
        # Call the chatbot function
        chatbot_module.run_chatbot()
    else:
        st.error("Chatbot module not found. Make sure 'chatbot_page.py' exists in the same folder as app.py.")
