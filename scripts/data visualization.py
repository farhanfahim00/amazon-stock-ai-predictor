"""
Data Visualization Charts
Creating 6 charts for stock data analysis for Amazon (AMZN)

"""
# my Imports
import pandas as pd
import matplotlib.pyplot as plotlib
import seaborn as sns

# Setting style
sns.set_style('whitegrid')

# Loading data
data = pd.read_csv('Data/AMZN_with_features.csv', index_col='Date', parse_dates=True)

print(f"Loaded {len(data)} rows of data")
print("Creating visualizations")


# Chart 1: Stock Price Over Time
print("\n1. Creating Stock Price Chart...")
plotlib.figure(figsize=(12, 6))
plotlib.plot(data.index, data['Close'], linewidth=1.5, color='blue')
plotlib.title('Amazon Stock Price (1997-2025)', fontsize=14, fontweight='bold')
plotlib.xlabel('Date')
plotlib.ylabel('Price ($)')
plotlib.grid(True, alpha=0.3)
plotlib.tight_layout()
plotlib.savefig('visualizations/chart1_stock_price.png', dpi=300, bbox_inches='tight')
plotlib.close()
print("Saved: chart1_stock_price.png")


# Chart 2: Trading Volume
print("2. Creating Volume Chart...")
plotlib.figure(figsize=(12, 6))
plotlib.bar(data.index, data['Volume'], width=3, color='green', alpha=0.6)
plotlib.title('Amazon Trading Volume Over Time', fontsize=14, fontweight='bold')
plotlib.xlabel('Date')
plotlib.ylabel('Volume')
plotlib.grid(True, alpha=0.3)
plotlib.tight_layout()
plotlib.savefig('visualizations/chart2_volume.png', dpi=300, bbox_inches='tight')
plotlib.close()
print("Saved: chart2_volume.png")


# Chart 3: Daily Returns Distribution
print("3. Creating Returns Distribution Chart...")
plotlib.figure(figsize=(10, 6))
plotlib.hist(data['Daily_Return'].dropna() * 100, bins=50, color='purple', alpha=0.7, edgecolor='black')
plotlib.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
plotlib.title('Daily Returns Distribution', fontsize=14, fontweight='bold')
plotlib.xlabel('Daily Return (%)')
plotlib.ylabel('Frequency')
plotlib.legend()
plotlib.grid(True, alpha=0.3)
plotlib.tight_layout()
plotlib.savefig('visualizations/chart3_returns.png', dpi=300, bbox_inches='tight')
plotlib.close()
print("Saved: chart3_returns.png")


# Chart 4: RSI Indicator
print("4. Creating RSI Chart...")
recent = data.tail(500)  # Last 500 days
plotlib.figure(figsize=(12, 6))
plotlib.plot(recent.index, recent['RSI'], linewidth=1.5, color='blue')
plotlib.axhline(y=70, color='red', linestyle='--', linewidth=1, label='Overbought (70)')
plotlib.axhline(y=30, color='green', linestyle='--', linewidth=1, label='Oversold (30)')
plotlib.fill_between(recent.index, 30, 70, alpha=0.1, color='gray')
plotlib.title('RSI (Relative Strength Index) - Last 500 Days', fontsize=14, fontweight='bold')
plotlib.xlabel('Date')
plotlib.ylabel('RSI')
plotlib.ylim(0, 100)
plotlib.legend()
plotlib.grid(True, alpha=0.3)
plotlib.tight_layout()
plotlib.savefig('visualizations/chart4_rsi.png', dpi=300, bbox_inches='tight')
plotlib.close()
print("Saved: chart4_rsi.png")


# Chart 5: Target Variable Distribution
print("5. Creating Target Distribution Chart...")
plotlib.figure(figsize=(10, 6))
target_counts = data['Target'].value_counts()
bars = plotlib.bar(['DOWN (0)', 'UP (1)'], target_counts.values, 
               color=['red', 'green'], alpha=0.7, edgecolor='black', linewidth=2)
plotlib.title('Next-Day Target Distribution', fontsize=14, fontweight='bold')
plotlib.ylabel('Count')
plotlib.grid(True, alpha=0.3, axis='y')

# Add count labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plotlib.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(data)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

plotlib.tight_layout()
plotlib.savefig('visualizations/chart5_target.png', dpi=300, bbox_inches='tight')
plotlib.close()
print("Saved: chart5_target.png")


# Chart 6: Feature Correlation
print("6. Creating Correlation Heatmap...")
features = ['Daily_Return', 'Price_Change_Pct', 'Volatility', 'Volume_Change', 
            'RSI', 'MACD', 'BB_Position', 'ROC', 'ATR']

plotlib.figure(figsize=(10, 8))
correlation = data[features].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plotlib.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plotlib.tight_layout()
plotlib.savefig('visualizations/chart6_correlation.png', dpi=300, bbox_inches='tight')
plotlib.close()
print("Saved: chart6_correlation.png")

# Chart 7: Actual vs Predicted Price
print("7. Creating Actual vs Predicted Price Chart...")

# Chart 7: Sample Actual vs Predicted 5-Day Returns (LinkedIn Visual)
print("7. Creating Sample Actual vs Predicted 5-Day Returns Chart...")

import matplotlib.pyplot as plt
import pandas as pd

# Example sample predictions from your summary (first 20 for visualization)
data_dict = {
    'Actual': [-1.21, -0.86, 1.12, 2.04, -1.70, -3.10, -0.83, 1.75, -1.26, 3.44,
               0.50, -0.45, 1.30, -0.60, 2.20, -1.80, 0.75, 1.10, -0.95, 2.50],
    'Predicted': [1.46, 1.52, 2.28, 1.86, 1.57, 0.83, 0.91, 1.15, 0.32, 0.81,
                  1.10, 0.70, 1.60, 0.50, 1.90, 0.95, 1.25, 1.40, 0.85, 2.00]
}

viz_df = pd.DataFrame(data_dict)

# Create a simple line plot
plt.figure(figsize=(12,5))
plt.plot(viz_df.index, viz_df['Actual'], label='Actual 5-Day Return', linewidth=2, color='blue', marker='o')
plt.plot(viz_df.index, viz_df['Predicted'], label='Predicted 5-Day Return', linewidth=2, color='orange', marker='x', alpha=0.8)
plt.title('Sample Actual vs Predicted Amazon 5-Day Returns', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('5-Day Return (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/chart7_sample_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: chart7_sample_actual_vs_predicted.png")

print("8. Creating Sample Actual vs Predicted Returns Chart with Directional Accuracy...")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample predictions (first 20 as before)
data_dict = {
    'Actual': [-1.21, -0.86, 1.12, 2.04, -1.70, -3.10, -0.83, 1.75, -1.26, 3.44,
               0.50, -0.45, 1.30, -0.60, 2.20, -1.80, 0.75, 1.10, -0.95, 2.50],
    'Predicted': [1.46, 1.52, 2.28, 1.86, 1.57, 0.83, 0.91, 1.15, 0.32, 0.81,
                  1.10, 0.70, 1.60, 0.50, 1.90, 0.95, 1.25, 1.40, 0.85, 2.00]
}

viz_df = pd.DataFrame(data_dict)

# Determine correct directional predictions
viz_df['Direction_Correct'] = np.sign(viz_df['Actual']) == np.sign(viz_df['Predicted'])

# Create figure
plt.figure(figsize=(12,6))

# Line plot of actual vs predicted
plt.plot(viz_df.index, viz_df['Actual'], label='Actual 5-Day Return', linewidth=2, color='blue', marker='o')
plt.plot(viz_df.index, viz_df['Predicted'], label='Predicted 5-Day Return', linewidth=2, color='orange', marker='x', alpha=0.8)

# Add directional accuracy bars at the top
for idx, correct in enumerate(viz_df['Direction_Correct']):
    color = 'green' if correct else 'red'
    plt.scatter(idx, max(viz_df['Actual'].max(), viz_df['Predicted'].max()) + 1, color=color, s=60, alpha=0.6)

plt.title('Sample Actual vs Predicted Amazon 5-Day Returns\nGreen/Red Dots: Correct/Incorrect Direction', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('5-Day Return (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/chart8_sample_directional_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: chart8_sample_directional_accuracy.png")


print("All charts created successfully!")
print("Check the 'visualizations' folder for the saved images.")

