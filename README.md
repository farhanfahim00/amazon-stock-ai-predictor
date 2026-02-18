# Farhan Fahim Taimoor 
# AI Stock Trend Predictor

---

## Project Description

The AI Stock Trend Predictor is a machine learning–based assistance system that analyzes historical Amazon (AMZN) stock data to forecast short-term market movement. This project demonstrates a complete ML pipeline including:

- Data preprocessing and feature engineering
- Leakage-free time series modeling
- Model training and evaluation
- Augmented (synthetic) data generation for robustness testing
- Interactive Streamlit application
- Voice-enabled chatbot with on-demand text-to-speech

Course: Assistance Systems (WS 25/26)
Stock: Amazon (AMZN)
Time Range: 1997 – 2025

---

## Key Features

- End-to-end ML pipeline: raw data → features → models → app
- Classification model (Next-Day UP/DOWN)
- Regression models (1-Day and 5-Day returns)
- Augmented synthetic dataset for robustness analysis
- Real vs Augmented model comparison
- Interactive Streamlit dashboard
- Voice-enabled chatbot
- Date-based price lookup + prediction
- On-demand text-to-speech responses

---

## Repository Structure

``` text
.
├── App/
├── Data/
├── models/
├── scripts/
├── visualizations/
├── model_training.py
├── model_training_augmented_data.py
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11+
- pip
- Microphone (optional for voice features)

### Setup

```bash
git clone git@github.com:farhanfahim00/amazon-stock-ai-predictor.git
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit app from the project root:

```bash
streamlit run "App/'App Home'.py"
```

The application will open in your default browser.

---

## Data Pipeline

**1) Create Engineered Features**

```bash
python scripts/'data cleaning'.py
```

**2) Generate Augmented Dataset**

```bash
python scripts/'augment data'.py
```

---

## Model Training

**Train Real Data Models**

```bash
python model_training.py
```

**Train Augmented Data Models**

```bash
python model_training_augmented_data.py
```

Models will be saved inside the `models/` folder as `.joblib` files.

---

## Model Philosophy

- Stock prediction at daily level is extremely difficult
- R² near zero is realistic for short-term returns
- Directional accuracy slightly above 50% can indicate weak but real signal
- This project focuses on methodological correctness, not inflated metrics

---

## Screencast

A full walkthrough of:

- Repository structure
- Data engineering
- Model training
- Streamlit app
- Chatbot (voice + text)

Link: *(Add your public video link here)*

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas | Data manipulation |
| NumPy | Numerical operations |
| scikit-learn | ML models |
| Streamlit | Web application |
| matplotlib | Static plots |
| Plotly | Interactive charts |
| SpeechRecognition | Voice input |
| gTTS | Text-to-speech |
| joblib | Model serialization |

---

## Disclaimer

This project is for educational purposes only and does not provide financial advice. Financial markets are highly efficient and unpredictable.

---
---

# Streamlit Application – AI Stock Trend Predictor

This folder contains the full Streamlit interface for the AI Stock Trend Predictor. The app provides interactive access to:

- Historical stock data exploration
- Technical indicator explanations
- Real data model predictions
- Augmented data model predictions
- Real vs Augmented comparison
- Chatbot interaction (text + voice)
- Date-based price lookup + prediction
- On-demand text-to-speech responses

---

## How to Run

From the project root directory:

```bash
streamlit run "App/'App Home'.py"
```

---

## App Pages Overview

**Home**
Project overview and navigation instructions.

**Data Overview**
- Dataset preview
- Feature explanations
- Selectable visualizations
- Target distribution
- Correlation matrix

**Real Data Predictions**
- Next-day direction classification
- 1-day return regression
- 5-day return regression

**Augmented Data Predictions**
- Same prediction types using augmented models
- Used to test robustness

**Real vs Augmented Comparison**
- Compare predictions using identical feature inputs
- Evaluate stability of models

**Chatbot**
Supports:
- Predict tomorrow
- Predict 1-day return
- Predict 5-day return
- Explain RSI, MACD, ATR, BB_Position
- Show top features
- Price on specific date
- Repeat last prediction
- Voice input
- Playable text-to-speech output

---

## Important Path Note

All paths inside the app use relative references such as:

```
Data/AMZN_data_with_features.csv
models/rf_classifier.joblib
visualizations/chart1_stock_price.png
```

This ensures portability across systems.

---

## Chatbot Architecture

Located in: `App/chatbot/`

| File | Role |
|---|---|
| `controller.py` | Routes intent |
| `intents.py` | Detects user intent |
| `responses.py` | Explainable response logic |
| `dialogs.py` | Example conversation patterns |

---

## Educational Purpose

This application demonstrates:

- Time-series safe ML design
- Realistic financial modeling
- Explainable AI concepts
- Robustness testing using synthetic data

It is not intended for trading decisions.
```