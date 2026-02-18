Farhan Fahim, Taimoor, 12400208

The AI Stock Trend Predictor

https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor

https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/-/wikis/Home

# Project Description

**WS 25/26 – Assistance Systems**

The **AI Stock Trend Predictor** is a machine learning–based assistance system designed to analyze historical Amazon (AMZN) stock data and provide short-term market insights.  
The system combines **feature engineering, ensemble learning models, and an interactive chatbot** to predict stock movement direction and short-term returns while emphasizing explainability and realistic expectations.

Rather than attempting to outperform financial markets, the project focuses on:
- Understanding machine learning behavior on noisy financial data
- Demonstrating the limitations of stock prediction
- Providing an interactive, voice-enabled assistant for educational analysis

The project implements a complete ML pipeline:
- Data preprocessing and feature engineering  
- Model training and evaluation  
- Real vs augmented data comparison  
- Interactive Streamlit dashboard  
- Context-aware chatbot with optional voice interaction


# Installation

## Prerequisites

- Python **3.11 or higher**
- pip (Python package manager)
- Microphone (optional, for voice input feature)

Clone the repo and install dependencies:

```bash
git clone https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor.git
cd ai-assistant-stock-trend-predictor-farhan-fahim-taimoor
pip install -r requirements.txt
```


### Running the Application

### Start the Streamlit App

```bash
streamlit run "App/App Home.py"
```
The application will open in your default web browser.


###  Data Cleaning & Feature Engineering

```bash
cd scripts
python data_cleaning.py
```

This script:
- Cleans raw Amazon stock data
- Engineers 18+ technical indicators
- Creates classification and regression targets
- Saves the processed dataset to the /Data/ directory


###  Model Training

```bash
python model_training.py
python model_training_augmented_data.py
```
This generates the following trained models:
- rf_classifier.joblib – Next-day direction classification
- rf_1day_regressor.joblib – 1-day return regression
- rf_5day_regressor.joblib – 5-day return regression
- ridge_regressor.joblib – Linear baseline model

Note: Pre-trained models are included in the repository for immediate use.


# Data

Kaggle Link to Amazon Stocks Dataset [Link](https://www.kaggle.com/datasets/meharshanali/amazon-stocks-2025)

The descriptions about the data can be found in the [Data Wiki](https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/-/wikis/Data)


# Basic Usage

Within the Streamlit application, users can:
- Explore historical stock data and engineered features
- Evaluate classification and regression model performance
- Compare real and augmented datasets
- Predict:
- Next-day stock direction (Up / Down)
- 1-day return
- 5-day return
- Interact with StockBot, a context-aware chatbot supporting:
	- Text-based queries
	- Optional voice input
	- On-demand text-to-speech responses
	- Date-based price queries followed by predictions

**Watch the screencast here:**  
[Streamlit App Screencast](App/screencast/streamlit-App Home-2026-01-23-19-01-20.webm)

Additional Notes
- Daily stock direction prediction is inherently difficult; the system achieves ~54–58% directional accuracy, which is realistic for financial time series.
- Regression results highlight the challenge of predicting exact returns, supporting the Efficient Market Hypothesis.
- All models are trained using leakage-free, time-series–aware validation.
- The chatbot is designed as an assistance system, focusing on explanation, interaction, and user guidance rather than automated trading decisions.



# Additional Notes

- The project demonstrates realistic expectations in stock prediction, showing the difficulty of exact returns but achieving ~54% directional accuracy.
- The ML pipeline is fully leakage-free, using proper time-series validation and feature engineering with advanced indicators like RSI, MACD, Bollinger Bands, ATR, ROC.
