"""
Data Cleaning and Feature Engineering for Stock Prediction

"""

import pandas as pd
import numpy as np
import os


INPUT_FILE = '/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/Data/AMZN_stock_data.csv' 

OUTPUT_FILE = '/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/Data/AMZN_with_features.csv'

# Loading Data
data = pd.read_csv(INPUT_FILE)
print(f"✓ Loaded {len(data)} rows")
print(f"✓ Columns: {list(data.columns)}")


# Cleaning Data 

# Converting Date to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Setting Date as index
data.set_index('Date', inplace=True)


# Sort by date
data.sort_index(inplace=True)

# Checking for missing values
missing_before = data.isnull().sum().sum()


# Features

print("\n Creating Features")

# Feature 1: Daily Return (percentage change from previous day)
data['Daily_Return'] = data['Close'].pct_change()

# Feature 2: Price Change (absolute change from Open to Close)
data['Price_Change'] = data['Close'] - data['Open']

# Feature 3: Price Change Percentage
data['Price_Change_Pct'] = ((data['Close'] - data['Open']) / data['Open']) * 100

# Feature 4: Moving Average - 5 days
data['MA_5'] = data['Close'].rolling(window=5).mean()

# Feature 5: Volatility (daily trading range)
data['Volatility'] = data['High'] - data['Low']

# Feature 6: Volume Change
data['Volume_Change'] = data['Volume'].pct_change()

# Feature 7: TARGET VARIABLE - Trend Direction (1 = Up, 0 = Down)
# This predicts if tomorrow's close will be higher than today's close
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)


# Feature 8: 10-day and 20-day Moving Averages
data['MA_10'] = data['Close'].rolling(window=10).mean()

data['MA_20'] = data['Close'].rolling(window=20).mean()


# Feature 9: Price momentum (over 5 days)
data['Momentum'] = data['Close'] - data['Close'].shift(4)

# Feature 10: RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data)

# Feature 11: MACD (Moving Average Convergence Divergence)
exp1 = data['Close'].ewm(span=12, adjust=False).mean()
exp2 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = exp1 - exp2
data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD_Diff'] = data['MACD'] - data['MACD_Signal']

# Feature 12: Bollinger Bands
data['BB_Middle'] = data['Close'].rolling(window=20).mean()
data['BB_Std'] = data['Close'].rolling(window=20).std()
data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

# Feature 13: Volume indicators
data['Volume_MA_5'] = data['Volume'].rolling(window=5).mean()
data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_5']

# Feature 14: Price Rate of Change
data['ROC'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100

# Feature 15: Average True Range (ATR) - volatility measure
data['TR1'] = data['High'] - data['Low']
data['TR2'] = abs(data['High'] - data['Close'].shift(1))
data['TR3'] = abs(data['Low'] - data['Close'].shift(1))
data['TR'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)
data['ATR'] = data['TR'].rolling(window=14).mean()
data.drop(['TR1', 'TR2', 'TR3', 'TR'], axis=1, inplace=True)


# Cleaning NaN Vales

rows_before = len(data)
data.dropna(inplace=True)
rows_after = len(data)
rows_removed = rows_before - rows_after

print(f"✓ Final dataset: {rows_after} rows")


# Saving Clean Data

data.to_csv(OUTPUT_FILE)
print(f"Saved to: {OUTPUT_FILE}")


# Summary Report of Processing and Cleaning

print("\n" + "="*80)
print("PROCESSING COMPLETE - SUMMARY")
print("="*80)

print(f"\nDataset Information:")
print(f"  • Date range: {data.index.min()} to {data.index.max()}")
print(f"  • Total trading days: {len(data)}")
print(f"  • Number of features: {len(data.columns)}")

print(f"\nFeatures created:")
feature_list = ['Daily_Return', 'Price_Change', 'Price_Change_Pct', 'MA_5', 
                'MA_10', 'MA_20', 'Volatility', 'Volume_Change', 'Momentum', 
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'BB_Position', 
                'Volume_Ratio', 'ROC', 'ATR', 'Target']
for i, feat in enumerate(feature_list, 1):
    print(f"  {i}. {feat}")

print(f"\nTarget Distribution:")
target_counts = data['Target'].value_counts()
print(f"  • UP days (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(data)*100:.2f}%)")
print(f"  • DOWN days (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(data)*100:.2f}%)")

print(f"\nOriginal columns: {list(data.columns[:8])}")
print(f"\nNew advanced features added:")
print(f"  • RSI (Relative Strength Index)")
print(f"  • MACD indicators")
print(f"  • Bollinger Bands position")
print(f"  • Volume ratio")
print(f"  • Rate of Change (ROC)")
print(f"  • Average True Range (ATR)")


# Displaying first and last few rows of the processed data
print("\nFirst 5 rows of processed data:")
print(data.head())

print("\nLast 5 rows of processed data:")
print(data.tail())