Farhan Fahim Taimoor, 12400208

The AI Stock Trend Predictor

https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor

https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/-/wikis/Home

## Project Description

The AI Stock Trend Predictor is a machine learning–based assistance system that analyzes historical Amazon (AMZN) stock data to forecast short-term market movements. The system uses advanced technical indicators, ensemble methods, and a chatbot interface to provide trend predictions, feature insights, and voice-based queries. The project demonstrates a complete ML pipeline, including data preprocessing, feature engineering, model training, evaluation, and user interaction through a Streamlit app.


## Installation

Ensure Python 3.12+ is installed. Then run:
```bash
pip install -r requirements.txt
```

## Key dependencies:

- scikit-learn 1.3+
- pandas 2.0+
- numpy 1.25+
- scipy 1.12+
- matplotlib 3.8+
- seaborn 0.12+
- streamlit 1.28+
- yfinance 0.2+


## Data

- [Data Wiki](https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/-/wikis/Data)
- [Amazon Stocks 2025 Dataset](https://www.kaggle.com/datasets/meharshanali/amazon-stocks-2025)

## Model Evaluation

- [Model Wiki](https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/-/wikis/Model-Evaluation)

## Basic Usage

1) Start the Streamlit app:

```bash
streamlit run app.py
```
2) Navigate the app to:
- Explore historical data and feature trends
- Predict next-day or 5-day returns using the ML models
- Interact with the chatbot for voice-based stock queries


## Wiki Pages

- [Data](https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/-/wikis/Data)
- [Data Preprocessing](https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/-/wikis/Data-Preprocessing)
- [Data Transformation](https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/-/wikis/Data-Transformation)
- [Data Storage](https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/-/wikis/Data-Storage)
- [Machine Learning Models](https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/-/wikis/Machine-Learning-Models)
- [Model Evaluation](https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/-/wikis/Model-Evaluation)
- [ChatBot Development](https://mygit.th-deg.de/ft02208/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/-/wikis/Chatbot-Development)

## Additional Notes

- The project demonstrates realistic expectations in stock prediction, showing the difficulty of exact returns but achieving ~54% directional accuracy.
- The ML pipeline is fully leakage-free, using proper time-series validation and feature engineering with advanced indicators like RSI, MACD, Bollinger Bands, ATR, ROC.
- The chatbot enables voice-based interaction with the system, providing an additional user interface for querying predictions and features.