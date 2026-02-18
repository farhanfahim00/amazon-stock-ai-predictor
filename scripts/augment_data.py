"""
AMZN Stock Prediction
Augmented Data Generator

Creates an augmented version of the existing feature dataset by:
- Sampling rows with replacement
- Adding small, controlled Gaussian noise to selected numeric features
- Clipping features to realistic ranges
- Saving the augmented dataset to the Data folder
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


# Input and output paths (exact path you provided)
INPUT_FILE = Path("/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/Data/AMZN_data_with_features.csv")
OUTPUT_FILE = INPUT_FILE.parent / "AMZN_augmented_data_with_features.csv"


# Configuration
RANDOM_STATE = 42

# Create extra rows as a fraction of original (0.5 means add 50 percent more rows)
AUGMENT_FRACTION = 0.50

# Noise strength relative to each column standard deviation
NOISE_STD_FRACTION = 0.10

# Feature columns to augment (based on your training script)
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
    "ATR",
]


def clip_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    # Clip to realistic ranges based on indicator meaning
    if "RSI" in df.columns:
        df["RSI"] = df["RSI"].clip(0, 100)

    if "BB_Position" in df.columns:
        df["BB_Position"] = df["BB_Position"].clip(0, 1)

    for col in ["Volatility", "ATR"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    for col in ["Volume_Ratio"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    np.random.seed(RANDOM_STATE)

    df = pd.read_csv(INPUT_FILE)

    # Basic validation
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns in CSV: {missing_cols}")

    original_len = len(df)
    n_new = int(original_len * AUGMENT_FRACTION)

    # Sample rows with replacement
    sampled = df.sample(n=n_new, replace=True, random_state=RANDOM_STATE).copy()

    # Add controlled Gaussian noise to numeric feature columns
    for col in FEATURE_COLUMNS:
        col_std = df[col].std()
        if pd.isna(col_std) or col_std == 0:
            continue

        noise = np.random.normal(loc=0.0, scale=col_std * NOISE_STD_FRACTION, size=len(sampled))
        sampled[col] = sampled[col].astype(float) + noise

    sampled = clip_feature_values(sampled)

    # Combine original + augmented
    augmented_df = pd.concat([df, sampled], ignore_index=True)

    # Save
    augmented_df.to_csv(OUTPUT_FILE, index=False)

    print("Augmentation complete")
    print(f"Original rows: {original_len}")
    print(f"Added rows: {n_new}")
    print(f"Total rows: {len(augmented_df)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()