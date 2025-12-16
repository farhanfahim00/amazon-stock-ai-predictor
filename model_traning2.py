"""
Improved Regression Model - 5-Day Return Prediction
Predicts the cumulative 5-day return percentage
"""

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
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
    'Momentum',
    'RSI',
    'MACD',
    'MACD_Diff',
    'BB_Position',
    'Volume_Ratio',
    'ROC',
    'ATR'
]

RANDOM_STATE = 42

print("="*80)
print("IMPROVED REGRESSION MODEL - 5-DAY RETURN PREDICTION")
print("="*80)


# Loading Data
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
print(f"\nLoaded {len(data)} rows from {data.index.min()} to {data.index.max()}")
print(f"Using {len(FEATURE_COLUMNS)} features")


#------------------------------------------------------------------------------
# Creating 5-Day Return Target
#------------------------------------------------------------------------------

print("\n" + "="*80)
print("CREATING 5-DAY RETURN TARGET")
print("="*80)

# Calculate 5-day future return (percentage)
data['Close_5day'] = data['Close'].shift(-5)
data['Return_5day'] = ((data['Close_5day'] - data['Close']) / data['Close']) * 100

# Remove rows with NaN
data_reg = data.dropna(subset=['Return_5day']).copy()

print(f"\n5-Day Return Statistics:")
print(f"  Mean return: {data_reg['Return_5day'].mean():.2f}%")
print(f"  Std dev: {data_reg['Return_5day'].std():.2f}%")
print(f"  Min return: {data_reg['Return_5day'].min():.2f}%")
print(f"  Max return: {data_reg['Return_5day'].max():.2f}%")
print(f"  Median return: {data_reg['Return_5day'].median():.2f}%")

# Show distribution
positive_returns = (data_reg['Return_5day'] > 0).sum()
negative_returns = (data_reg['Return_5day'] < 0).sum()
print(f"\n  Positive returns: {positive_returns} ({positive_returns/len(data_reg)*100:.2f}%)")
print(f"  Negative returns: {negative_returns} ({negative_returns/len(data_reg)*100:.2f}%)")


#------------------------------------------------------------------------------
# Data Preparation
#------------------------------------------------------------------------------

print("\n" + "="*80)
print("PREPARING DATA")
print("="*80)

X_reg = data_reg[FEATURE_COLUMNS].copy()
y_reg = data_reg['Return_5day'].copy()

# Time-based split (80% train, 20% test)
split_index = int(len(X_reg) * 0.8)

X_train = X_reg.iloc[:split_index]
X_test = X_reg.iloc[split_index:]
y_train = y_reg.iloc[:split_index]
y_test = y_reg.iloc[split_index:]

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Training period: {X_train.index.min()} to {X_train.index.max()}")
print(f"Test period: {X_test.index.min()} to {X_test.index.max()}")


#------------------------------------------------------------------------------
# Model 1: Random Forest Regressor
#------------------------------------------------------------------------------

print("\n" + "="*80)
print("MODEL 1: RANDOM FOREST REGRESSOR")
print("="*80)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=25,
    min_samples_leaf=12,
    bootstrap=True,
    max_samples=0.8,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Metrics
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Directional accuracy
actual_direction = (y_test > 0).astype(int)
predicted_direction = (y_pred_rf > 0).astype(int)
directional_acc_rf = (actual_direction == predicted_direction).mean()

print(f"\nRandom Forest Performance:")
print(f"  RMSE: {rmse_rf:.4f}%")
print(f"  MAE: {mae_rf:.4f}%")
print(f"  R²: {r2_rf:.4f}")
print(f"  Directional Accuracy: {directional_acc_rf:.4f} ({directional_acc_rf*100:.2f}%)")

train_r2_rf = rf_model.score(X_train, y_train)
print(f"\nOverfitting Check:")
print(f"  Train R²: {train_r2_rf:.4f}")
print(f"  Test R²: {r2_rf:.4f}")
print(f"  Gap: {abs(train_r2_rf - r2_rf):.4f}")


#------------------------------------------------------------------------------
# Model 2: Gradient Boosting Regressor
#------------------------------------------------------------------------------

print("\n" + "="*80)
print("MODEL 2: GRADIENT BOOSTING REGRESSOR")
print("="*80)

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=25,
    min_samples_leaf=12,
    subsample=0.8,
    random_state=RANDOM_STATE
)

gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# Metrics
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

predicted_direction_gb = (y_pred_gb > 0).astype(int)
directional_acc_gb = (actual_direction == predicted_direction_gb).mean()

print(f"\nGradient Boosting Performance:")
print(f"  RMSE: {rmse_gb:.4f}%")
print(f"  MAE: {mae_gb:.4f}%")
print(f"  R²: {r2_gb:.4f}")
print(f"  Directional Accuracy: {directional_acc_gb:.4f} ({directional_acc_gb*100:.2f}%)")

train_r2_gb = gb_model.score(X_train, y_train)
print(f"\nOverfitting Check:")
print(f"  Train R²: {train_r2_gb:.4f}")
print(f"  Test R²: {r2_gb:.4f}")
print(f"  Gap: {abs(train_r2_gb - r2_gb):.4f}")


#------------------------------------------------------------------------------
# Model 3: Ridge Regression (Linear Baseline)
#------------------------------------------------------------------------------

print("\n" + "="*80)
print("MODEL 3: RIDGE REGRESSION (LINEAR BASELINE)")
print("="*80)

ridge_model = Ridge(alpha=1.0, random_state=RANDOM_STATE)

ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Metrics
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

predicted_direction_ridge = (y_pred_ridge > 0).astype(int)
directional_acc_ridge = (actual_direction == predicted_direction_ridge).mean()

print(f"\nRidge Regression Performance:")
print(f"  RMSE: {rmse_ridge:.4f}%")
print(f"  MAE: {mae_ridge:.4f}%")
print(f"  R²: {r2_ridge:.4f}")
print(f"  Directional Accuracy: {directional_acc_ridge:.4f} ({directional_acc_ridge*100:.2f}%)")


#------------------------------------------------------------------------------
# Cross-Validation
#------------------------------------------------------------------------------

print("\n" + "="*80)
print("CROSS-VALIDATION (Time Series)")
print("="*80)

tscv = TimeSeriesSplit(n_splits=5)

# Cross-validate Random Forest
cv_scores_rf = []
for train_idx, val_idx in tscv.split(X_reg):
    X_tr, X_val = X_reg.iloc[train_idx], X_reg.iloc[val_idx]
    y_tr, y_val = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
    
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=25,
        min_samples_leaf=12,
        bootstrap=True,
        max_samples=0.8,
        max_features='sqrt',
        random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_tr)
    cv_scores_rf.append(model.score(X_val, y_val))

print(f"\nRandom Forest CV R² Scores: {[f'{s:.4f}' for s in cv_scores_rf]}")
print(f"Random Forest CV Average: {np.mean(cv_scores_rf):.4f}")
print(f"Random Forest CV Std Dev: {np.std(cv_scores_rf):.4f}")


#------------------------------------------------------------------------------
# Feature Importance
#------------------------------------------------------------------------------

print("\n" + "="*80)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': FEATURE_COLUMNS,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:20s}: {row['importance']:.4f}")


#------------------------------------------------------------------------------
# Prediction Analysis
#------------------------------------------------------------------------------

print("\n" + "="*80)
print("PREDICTION ANALYSIS (Best Model)")
print("="*80)

# Select best model based on R²
best_r2 = max(r2_rf, r2_gb, r2_ridge)
if best_r2 == r2_rf:
    best_model_name = "Random Forest"
    best_pred = y_pred_rf
    best_model = rf_model
    best_r2_val = r2_rf
    best_mae = mae_rf
    best_dir_acc = directional_acc_rf
elif best_r2 == r2_gb:
    best_model_name = "Gradient Boosting"
    best_pred = y_pred_gb
    best_model = gb_model
    best_r2_val = r2_gb
    best_mae = mae_gb
    best_dir_acc = directional_acc_gb
else:
    best_model_name = "Ridge Regression"
    best_pred = y_pred_ridge
    best_model = ridge_model
    best_r2_val = r2_ridge
    best_mae = mae_ridge
    best_dir_acc = directional_acc_ridge

print(f"\nBest Model: {best_model_name}")

# Prediction error analysis
errors = y_test - best_pred
print(f"\nPrediction Errors:")
print(f"  Mean error: {errors.mean():.4f}%")
print(f"  Error std dev: {errors.std():.4f}%")
print(f"  Max overestimate: {errors.min():.4f}%")
print(f"  Max underestimate: {errors.max():.4f}%")

# Show some example predictions
print(f"\nSample Predictions (first 10 test samples):")
print(f"{'Actual':>10s} {'Predicted':>10s} {'Error':>10s}")
print("-" * 35)
for i in range(min(10, len(y_test))):
    print(f"{y_test.iloc[i]:>9.2f}% {best_pred[i]:>9.2f}% {errors.iloc[i]:>9.2f}%")


#------------------------------------------------------------------------------
# Model Comparison Summary
#------------------------------------------------------------------------------

print("\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting', 'Ridge Regression'],
    'R²': [r2_rf, r2_gb, r2_ridge],
    'MAE (%)': [mae_rf, mae_gb, mae_ridge],
    'RMSE (%)': [rmse_rf, rmse_gb, rmse_ridge],
    'Directional Acc (%)': [directional_acc_rf*100, directional_acc_gb*100, directional_acc_ridge*100]
})

print("\n" + comparison_df.to_string(index=False))

print(f"\n{'='*80}")
print("RECOMMENDATION")
print("="*80)
print(f"\nBest Model: {best_model_name}")
print(f"R² Score: {best_r2_val:.4f}")
print(f"Mean Absolute Error: {best_mae:.2f}%")
print(f"Directional Accuracy: {best_dir_acc*100:.2f}%")

if best_r2_val > 0:
    print(f"\n✓ Positive R² indicates the model explains {best_r2_val*100:.2f}% of variance")
    print(f"  in 5-day returns, which is better than just predicting the mean.")
else:
    print(f"\n⚠ Negative R² indicates the model performs worse than baseline.")
    print(f"  However, directional accuracy of {best_dir_acc*100:.2f}% shows some predictive power.")

print(f"\n  Average prediction error is ±{best_mae:.2f}% for 5-day returns.")


print("\n" + "="*80)
print("REGRESSION MODEL COMPLETE")
print("="*80)
