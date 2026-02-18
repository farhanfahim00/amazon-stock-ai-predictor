"""
AMZN Stock Prediction
Model Training Script

Models:
1. Random Forest Classification (Up / Down)
2. Random Forest Regression (5-day return)
3. Random Forest Regression (1-day return)
4. Ridge Regression (Linear baseline)
"""


# All Imports
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)


# Configuration
INPUT_FILE = "Data/AMZN_data_with_features.csv"
MODEL_DIR = "models/"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "Daily_Return",
    "Price_Change_Pct",
    "Volatility",
    "Volume_Change",
    "Momentum",
    "RSI",
    "MACD",
    "MACD_Diff",
    "BB_Position",
    "Volume_Ratio",
    "ROC",
    "ATR"
]


# Loading Data
data = pd.read_csv(INPUT_FILE, index_col="Date", parse_dates=True)

print(f"Loaded {len(data)} rows")
print(f"Using features: {FEATURE_COLUMNS}")

tscv = TimeSeriesSplit(n_splits=5)


# MODEL 1: Classification
# Predict UP or DOWN tomorrow


print("\nMODEL 1: Classification (Up / Down)")

X_cls = data[FEATURE_COLUMNS]
y_cls = data["Target"]

split = int(len(X_cls) * 0.8)
X_train_cls, X_test_cls = X_cls.iloc[:split], X_cls.iloc[split:]
y_train_cls, y_test_cls = y_cls.iloc[:split], y_cls.iloc[split:]

clf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_split=40,
    min_samples_leaf=20,
    max_samples=0.8,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

clf_model.fit(X_train_cls, y_train_cls)
y_pred_cls = clf_model.predict(X_test_cls)

# Training performance
y_pred_train_cls = clf_model.predict(X_train_cls)
train_acc = accuracy_score(y_train_cls, y_pred_train_cls)

# Test performance
acc = accuracy_score(y_test_cls, y_pred_cls)
prec = precision_score(y_test_cls, y_pred_cls, zero_division=0)
rec = recall_score(y_test_cls, y_pred_cls, zero_division=0)
f1 = f1_score(y_test_cls, y_pred_cls, zero_division=0)

cv_acc = np.mean([
    clf_model.fit(X_cls.iloc[tr], y_cls.iloc[tr]).score(
        X_cls.iloc[val], y_cls.iloc[val]
    )
    for tr, val in tscv.split(X_cls)
])

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Accuracy Gap: {train_acc - acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")
print(f"CV Accuracy: {cv_acc:.4f}")

# Overfitting/Underfitting Detection
if train_acc - acc > 0.10:
    print("WARNING: Possible OVERFITTING (train accuracy >> test accuracy)")
elif train_acc < 0.55 and acc < 0.55:
    print("WARNING: Possible UNDERFITTING (both train and test accuracy low)")
else:
    print("Model appears well-fitted")




# MODEL 2: Regression (5-day return)

print("\nMODEL 2: Regression (5-day return)")

data_reg5 = data.copy()
data_reg5["Next_5_Close"] = data_reg5["Close"].shift(-5)
data_reg5.dropna(inplace=True)
data_reg5["Target_5d_Return"] = (
    data_reg5["Next_5_Close"] / data_reg5["Close"] - 1
)

X_reg5 = data_reg5[FEATURE_COLUMNS]
y_reg5 = data_reg5["Target_5d_Return"]

split = int(len(X_reg5) * 0.8)
X_train_5, X_test_5 = X_reg5.iloc[:split], X_reg5.iloc[split:]
y_train_5, y_test_5 = y_reg5.iloc[:split], y_reg5.iloc[split:]

rf_5day = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=25,
    min_samples_leaf=12,
    max_samples=0.8,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_5day.fit(X_train_5, y_train_5)
y_pred_5 = rf_5day.predict(X_test_5)

# Training performance
y_pred_train_5 = rf_5day.predict(X_train_5)
train_rmse_5 = np.sqrt(mean_squared_error(y_train_5, y_pred_train_5))
train_r2_5 = r2_score(y_train_5, y_pred_train_5)

# Test performance
rmse_5 = np.sqrt(mean_squared_error(y_test_5, y_pred_5))
mae_5 = mean_absolute_error(y_test_5, y_pred_5)
r2_5 = r2_score(y_test_5, y_pred_5)

print(f"Train RMSE: {train_rmse_5:.4f}, Train R2: {train_r2_5:.4f}")
print(f"Test RMSE: {rmse_5:.4f}, Test R2: {r2_5:.4f}")
print(f"RMSE Gap: {rmse_5 - train_rmse_5:.4f}")
print(f"R2 Gap: {train_r2_5 - r2_5:.4f}")
print(f"MAE: {mae_5:.4f}")

# Overfitting/Underfitting Detection
if train_r2_5 - r2_5 > 0.15:
    print("WARNING: Possible OVERFITTING (train R2 >> test R2)")
elif train_r2_5 < 0.10 and r2_5 < 0.10:
    print("WARNING: Possible UNDERFITTING (both train and test R2 very low)")
else:
    print("Model appears well-fitted")





# MODEL 3: Regression (1-day return)



print("\nMODEL 3: Regression (1-day return)")

data_reg1 = data.copy()
data_reg1["Next_Close"] = data_reg1["Close"].shift(-1)
data_reg1.dropna(inplace=True)
data_reg1["Target_1d_Return"] = (
    data_reg1["Next_Close"] / data_reg1["Close"] - 1
)

X_reg1 = data_reg1[FEATURE_COLUMNS]
y_reg1 = data_reg1["Target_1d_Return"]

split = int(len(X_reg1) * 0.8)
X_train_1, X_test_1 = X_reg1.iloc[:split], X_reg1.iloc[split:]
y_train_1, y_test_1 = y_reg1.iloc[:split], y_reg1.iloc[split:]

rf_1day = RandomForestRegressor(
    n_estimators=150,
    max_depth=6,
    min_samples_split=50,
    min_samples_leaf=20,
    max_samples=0.7,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_1day.fit(X_train_1, y_train_1)
y_pred_1 = rf_1day.predict(X_test_1)

# Training performance
y_pred_train_1 = rf_1day.predict(X_train_1)
train_rmse_1 = np.sqrt(mean_squared_error(y_train_1, y_pred_train_1))
train_r2_1 = r2_score(y_train_1, y_pred_train_1)

# Test performance
rmse_1 = np.sqrt(mean_squared_error(y_test_1, y_pred_1))
mae_1 = mean_absolute_error(y_test_1, y_pred_1)
r2_1 = r2_score(y_test_1, y_pred_1)

print(f"Train RMSE: {train_rmse_1:.4f}, Train R2: {train_r2_1:.4f}")
print(f"Test RMSE: {rmse_1:.4f}, Test R2: {r2_1:.4f}")
print(f"RMSE Gap: {rmse_1 - train_rmse_1:.4f}")
print(f"R2 Gap: {train_r2_1 - r2_1:.4f}")
print(f"MAE: {mae_1:.4f}")

# Overfitting/Underfitting Detection
if train_r2_1 - r2_1 > 0.15:
    print("WARNING: Possible OVERFITTING (train R2 >> test R2)")
elif train_r2_1 < 0.05 and r2_1 < 0.05:
    print("WARNING: Possible UNDERFITTING (both train and test R2 very low)")
else:
    print("Model appears well-fitted")




# MODEL 4: Ridge Regression (baseline)


print("\nMODEL 4: Ridge Regression (Baseline)")

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_1, y_train_1)
y_pred_ridge = ridge_model.predict(X_test_1)

# Training performance
y_pred_train_ridge = ridge_model.predict(X_train_1)
train_rmse_r = np.sqrt(mean_squared_error(y_train_1, y_pred_train_ridge))
train_r2_r = r2_score(y_train_1, y_pred_train_ridge)

# Test performance
rmse_r = np.sqrt(mean_squared_error(y_test_1, y_pred_ridge))
mae_r = mean_absolute_error(y_test_1, y_pred_ridge)
r2_r = r2_score(y_test_1, y_pred_ridge)

print(f"Train RMSE: {train_rmse_r:.4f}, Train R2: {train_r2_r:.4f}")
print(f"Test RMSE: {rmse_r:.4f}, Test R2: {r2_r:.4f}")
print(f"RMSE Gap: {rmse_r - train_rmse_r:.4f}")
print(f"R2 Gap: {train_r2_r - r2_r:.4f}")
print(f"MAE: {mae_r:.4f}")

# Overfitting/Underfitting Detection
if train_r2_r - r2_r > 0.15:
    print("WARNING: Possible OVERFITTING (train R2 >> test R2)")
elif train_r2_r < 0.05 and r2_r < 0.05:
    print("WARNING: Possible UNDERFITTING (both train and test R2 very low)")
else:
    print("Model appears well-fitted")




# FINAL SUMMARY


print("\nMODEL COMPARISON SUMMARY")

summary = pd.DataFrame({
    "Model": [
        "RF Classification",
        "RF Regression 5-day",
        "RF Regression 1-day",
        "Ridge Regression"
    ],
    "Primary Metric": [
        acc,
        r2_5,
        r2_1,
        r2_r
    ]
})

print(summary)



# Saving Models

joblib.dump(clf_model, MODEL_DIR + "rf_classifier.joblib")
joblib.dump(rf_5day, MODEL_DIR + "rf_5day_regressor.joblib")
joblib.dump(rf_1day, MODEL_DIR + "rf_1day_regressor.joblib")
joblib.dump(ridge_model, MODEL_DIR + "ridge_regressor.joblib")

print("\nAll models saved successfully")