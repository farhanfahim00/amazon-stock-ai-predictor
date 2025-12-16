"""
Stock Prediction Models 
"""

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)



# Input file with features
INPUT_FILE = 'Data/AMZN_with_features.csv'

# Features to be used for modeling
FEATURE_COLUMNS = [
    'Daily_Return',
    'Price_Change_Pct',
    'Volatility',
    'Volume_Change',
    'Momentum'
]

RANDOM_STATE = 42

print("="*80)
print("STOCK PREDICTION - CLASSIFICATION vs REGRESSION")
print("="*80)


# Loading Data
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
print(f"Loaded {len(data)} rows from {data.index.min()} to {data.index.max()}")
print(f"Using {len(FEATURE_COLUMNS)} features: {FEATURE_COLUMNS}")



#------------------------------------------------------------------------------
# Classification Prediction: Will the stock go UP or DOWN tomorrow
#------------------------------------------------------------------------------

print("CLASSIFICATION MODEL (UP/DOWN PREDICTION)")

# Data Preparation
X_cls = data[FEATURE_COLUMNS].copy()
y_cls = data['Target'].copy()

# Time based split
split_index = int(len(X_cls) * 0.8)

X_train_cls = X_cls.iloc[:split_index]
X_test_cls = X_cls.iloc[split_index:]
y_train_cls = y_cls.iloc[:split_index]
y_test_cls = y_cls.iloc[split_index:]


# Training Classification Model
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=40,
    min_samples_leaf=20,
    bootstrap=True,
    max_samples=0.7,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_classifier.fit(X_train_cls, y_train_cls)
print("Classification Training complete")

# Making Classification Predictions
y_pred_cls = rf_classifier.predict(X_test_cls)

# Cross Validation for Classification
tscv = TimeSeriesSplit(n_splits=5)
cv_scores_cls = []

for train_idx, val_idx in tscv.split(X_cls):
    X_tr, X_val = X_cls.iloc[train_idx], X_cls.iloc[val_idx]
    y_tr, y_val = y_cls.iloc[train_idx], y_cls.iloc[val_idx]
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=40,
        min_samples_leaf=20,
        bootstrap=True,
        max_samples=0.7,
        max_features='sqrt',
        random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_tr)
    cv_scores_cls.append(model.score(X_val, y_val))


# Evaluating Classification Model
accuracy_cls = accuracy_score(y_test_cls, y_pred_cls)
precision_cls = precision_score(y_test_cls, y_pred_cls, zero_division=0)
recall_cls = recall_score(y_test_cls, y_pred_cls, zero_division=0)
f1_cls = f1_score(y_test_cls, y_pred_cls, zero_division=0)

print(f"\nPerformance Metrics:")
print(f" Accuracy:  {accuracy_cls:.4f} ({accuracy_cls*100:.2f} percent)")
print(f" Precision: {precision_cls:.4f} ({precision_cls*100:.2f} percent)")
print(f" Recall:    {recall_cls:.4f} ({recall_cls*100:.2f} percent)")
print(f" F1 Score:  {f1_cls:.4f} ({f1_cls*100:.2f} percent)")
print(f" CV Avg:    {np.mean(cv_scores_cls):.4f}")

# Overfitting check
train_score_cls = rf_classifier.score(X_train_cls, y_train_cls)
print(f"\nOverfitting Check:")
print(f"  Train Accuracy: {train_score_cls:.4f}")
print(f"  Test Accuracy:  {accuracy_cls:.4f}")
print(f"  Gap: {abs(train_score_cls - accuracy_cls):.4f}")
if abs(train_score_cls - accuracy_cls) > 0.10:
    print(" Large gap suggests overfitting")
else:
    print(" Good generalization")



#------------------------------------------------------------------------------
# Regression Prediction: What will be the stock price tomorrow
#------------------------------------------------------------------------------

print("REGRESSION MODEL (PRICE PREDICTION)")

# Avoid leakage by predicting next day return only
data_reg = data.copy()
data_reg['Next_Close'] = data_reg['Close'].shift(-1)
data_reg = data_reg.dropna()

# Stationary target
data_reg['Next_Return'] = data_reg['Next_Close'] / data_reg['Close'] - 1.0

X_reg = data_reg[FEATURE_COLUMNS]
y_reg = data_reg['Next_Return']
current_prices = data_reg['Close']

split_index_reg = int(len(X_reg) * 0.8)

X_train_reg = X_reg.iloc[:split_index_reg]
X_test_reg = X_reg.iloc[split_index_reg:]
y_train_reg = y_reg.iloc[:split_index_reg]
y_test_reg = y_reg.iloc[split_index_reg:]
test_current_prices = current_prices.iloc[split_index_reg:]


# Training Regression Model
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=20,
    bootstrap=True,
    max_samples=0.7,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_regressor.fit(X_train_reg, y_train_reg)
print("Regression Model Training complete")

# Predict returns
y_pred_return = rf_regressor.predict(X_test_reg)

# Convert to price prediction
y_pred_reg = test_current_prices.values * (1.0 + y_pred_return)
true_next_price = test_current_prices.values * (1.0 + y_test_reg.values)

# Cross validation for regression
cv_scores_reg = []
for train_idx, val_idx in tscv.split(X_reg):
    X_tr, X_val = X_reg.iloc[train_idx], X_reg.iloc[val_idx]
    y_tr, y_val = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=50,
        min_samples_leaf=20,
        bootstrap=True,
        max_samples=0.7,
        max_features='sqrt',
        random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_tr)
    cv_scores_reg.append(model.score(X_val, y_val))


# Evaluating Regression Model (on RETURNS, not prices)
rmse_return = np.sqrt(mean_squared_error(y_test_reg, y_pred_return))
mae_return = mean_absolute_error(y_test_reg, y_pred_return)
mape_return = np.mean(np.abs((y_test_reg - y_pred_return) / (np.abs(y_test_reg) + 1e-8))) * 100
r2_return = r2_score(y_test_reg, y_pred_return)

# Directional accuracy (for returns)
actual_direction = (y_test_reg > 0).astype(int)
predicted_direction = (y_pred_return > 0).astype(int)
directional_accuracy = accuracy_score(actual_direction, predicted_direction)

# Convert to price for reporting (but don't evaluate on it)
y_pred_price = test_current_prices.values * (1.0 + y_pred_return)
true_next_price = test_current_prices.values * (1.0 + y_test_reg.values)
mae_price = mean_absolute_error(true_next_price, y_pred_price)

print(f"\nPerformance Metrics (on Returns):")
print(f"  RMSE (Return): {rmse_return:.4f}")
print(f"  MAE (Return):  {mae_return:.4f}")
print(f"  MAPE (Return): {mape_return:.4f} percent")
print(f"  R2 (Return):   {r2_return:.4f}")
print(f"  CV Avg R2:     {np.mean(cv_scores_reg):.4f}")
print(f"  Directional Accuracy: {directional_accuracy:.4f} ({directional_accuracy*100:.2f} percent)")
print(f"\nPrice-based (for reference):")
print(f"  MAE (Price):   ${mae_price:.2f}")

# Overfitting check
train_score_reg = rf_regressor.score(X_train_reg, y_train_reg)
print(f"\nOverfitting Check:")
print(f"  Train R2: {train_score_reg:.4f}")
print(f"  Test R2:  {r2_return:.4f}")
print(f"  Gap: {abs(train_score_reg - r2_return):.4f}")
if abs(train_score_reg - r2_return) > 0.15:
    print(" Large gap suggests overfitting")
else:
    print(" Reasonable performance")



#------------------------------------------------------------------------------
# Comparison
#------------------------------------------------------------------------------

print("MODEL COMPARISON SUMMARY")

print("\n CLASSIFICATION MODEL:")
print(f"Accuracy:  {accuracy_cls*100:.2f} percent")
print(f"Precision: {precision_cls*100:.2f} percent")
print(f"Recall:    {recall_cls*100:.2f} percent")
print(f"CV Average: {np.mean(cv_scores_cls)*100:.2f} percent")

print("\nREGRESSION MODEL:")
print(f"   R2 (Return): {r2_return:.4f}")
print(f"   MAE (Price): ${mae_price:.2f}")
print(f"   MAPE (Return): {mape_return:.2f} percent")
print(f"   Directional: {directional_accuracy*100:.2f} percent")
print(f"   CV Average: {np.mean(cv_scores_reg):.4f}")
