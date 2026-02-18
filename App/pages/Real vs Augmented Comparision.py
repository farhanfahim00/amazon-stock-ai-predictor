import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Real vs Augmented Comparison", layout="wide")
st.title("Real vs Augmented Model Comparison")
st.subheader("Same input indicators, compared across real-data models and augmented-data models")

st.markdown("""
This page takes **one set of indicator inputs** (sliders) and runs predictions on:
- **Real-data models**
- **Augmented-data models**

Then it compares the outputs side by side so you can see how much predictions change when training data changes.

Note: These predictions are for learning and comparison only, not trading.
""")

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

MODELS_DIR = Path("models")

REAL_MODELS = {
    "clf": MODELS_DIR / "/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/models/rf_classifier.joblib",
    "reg_1d": MODELS_DIR / "/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/models/rf_1day_regressor.joblib",
    "reg_5d": MODELS_DIR / "/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/models/rf_5day_regressor.joblib",
}

AUG_MODELS = {
    "clf": MODELS_DIR / "/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/models/rf_classifier_augmented.joblib",
    "reg_1d": MODELS_DIR / "/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/models/rf_1day_regressor_augmented.joblib",
    "reg_5d": MODELS_DIR / "/Users/farhanfahim/Documents/Assistance Systems (WS25:26)/ai-assistant-stock-trend-predictor-farhan-fahim-taimoor/models/rf_5day_regressor_augmented.joblib",
}


@st.cache_resource
def load_models():
    def safe_load(p: Path):
        if not p.exists():
            return None, f"Missing file: {p}"
        try:
            return joblib.load(p), None
        except Exception as e:
            return None, f"Failed to load {p}: {e}"

    real = {}
    aug = {}
    errors = []

    for k, p in REAL_MODELS.items():
        m, err = safe_load(p)
        real[k] = m
        if err:
            errors.append(err)

    for k, p in AUG_MODELS.items():
        m, err = safe_load(p)
        aug[k] = m
        if err:
            errors.append(err)

    return real, aug, errors


real_models, aug_models, model_errors = load_models()

if model_errors:
    st.error("Some models could not be loaded:")
    for e in model_errors:
        st.write(f"- {e}")
    st.stop()


st.markdown("## Input Technical Indicators")
st.caption("Use the same inputs for both model sets, then compare outputs.")

col_a, col_b, col_c = st.columns(3)

with col_a:
    daily_return = st.slider("Daily_Return", -0.15, 0.15, 0.0, 0.001)
    price_change_pct = st.slider("Price_Change_Pct", -8.0, 8.0, 0.0, 0.01)
    volatility = st.slider("Volatility", 0.0, 15.0, 2.0, 0.01)
    volume_change = st.slider("Volume_Change", -1.0, 1.0, 0.0, 0.01)

with col_b:
    momentum = st.slider("Momentum", -25.0, 25.0, 0.0, 0.01)
    rsi = st.slider("RSI", 0.0, 100.0, 50.0, 0.1)
    macd = st.slider("MACD", -10.0, 10.0, 0.0, 0.01)
    macd_diff = st.slider("MACD_Diff", -5.0, 5.0, 0.0, 0.01)

with col_c:
    bb_position = st.slider("BB_Position", 0.0, 1.0, 0.5, 0.001)
    volume_ratio = st.slider("Volume_Ratio", 0.0, 5.0, 1.0, 0.01)
    roc = st.slider("ROC", -50.0, 50.0, 0.0, 0.01)
    atr = st.slider("ATR", 0.0, 15.0, 1.0, 0.01)

features = {
    "Daily_Return": daily_return,
    "Price_Change_Pct": price_change_pct,
    "Volatility": volatility,
    "Volume_Change": volume_change,
    "Momentum": momentum,
    "RSI": rsi,
    "MACD": macd,
    "MACD_Diff": macd_diff,
    "BB_Position": bb_position,
    "Volume_Ratio": volume_ratio,
    "ROC": roc,
    "ATR": atr,
}

X_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)


def predict_all(model_pack, X: pd.DataFrame):
    clf = model_pack["clf"]
    reg_1d = model_pack["reg_1d"]
    reg_5d = model_pack["reg_5d"]

    out = {}

    try:
        pred = int(clf.predict(X)[0])
        proba = clf.predict_proba(X)[0]
        out["direction_label"] = "UP" if pred == 1 else "DOWN"
        out["p_up"] = float(proba[1])
        out["p_down"] = float(proba[0])
    except Exception as e:
        out["direction_label"] = "ERROR"
        out["p_up"] = None
        out["p_down"] = None
        out["clf_error"] = str(e)

    try:
        out["ret_1d"] = float(reg_1d.predict(X)[0])
    except Exception as e:
        out["ret_1d"] = None
        out["reg1_error"] = str(e)

    try:
        out["ret_5d"] = float(reg_5d.predict(X)[0])
    except Exception as e:
        out["ret_5d"] = None
        out["reg5_error"] = str(e)

    return out


def pct(x):
    if x is None:
        return "N/A"
    return f"{x * 100:.2f}%"


def conf(p):
    if p is None:
        return "N/A"
    return f"{p * 100:.2f}%"


if st.button("Run Comparison"):
    real_out = predict_all(real_models, X_df)
    aug_out = predict_all(aug_models, X_df)

    st.markdown("## Results")

    left, right = st.columns(2)

    with left:
        st.markdown("### Real-data models")
        if real_out["direction_label"] != "ERROR":
            st.write(f"Next-day direction: **{real_out['direction_label']}**")
            st.write(f"Confidence UP: **{conf(real_out['p_up'])}**")
            st.write(f"Confidence DOWN: **{conf(real_out['p_down'])}**")
        else:
            st.error(f"Classification error: {real_out.get('clf_error', 'unknown')}")

        st.write(f"Predicted 1-day return: **{pct(real_out['ret_1d'])}**")
        st.write(f"Predicted 5-day return: **{pct(real_out['ret_5d'])}**")

    with right:
        st.markdown("### Augmented-data models")
        if aug_out["direction_label"] != "ERROR":
            st.write(f"Next-day direction: **{aug_out['direction_label']}**")
            st.write(f"Confidence UP: **{conf(aug_out['p_up'])}**")
            st.write(f"Confidence DOWN: **{conf(aug_out['p_down'])}**")
        else:
            st.error(f"Classification error: {aug_out.get('clf_error', 'unknown')}")

        st.write(f"Predicted 1-day return: **{pct(aug_out['ret_1d'])}**")
        st.write(f"Predicted 5-day return: **{pct(aug_out['ret_5d'])}**")

    st.markdown("## Difference Summary")

    def diff(a, b):
        if a is None or b is None:
            return None
        return b - a

    delta_p_up = diff(real_out["p_up"], aug_out["p_up"])
    delta_1d = diff(real_out["ret_1d"], aug_out["ret_1d"])
    delta_5d = diff(real_out["ret_5d"], aug_out["ret_5d"])

    summary = pd.DataFrame(
        {
            "Metric": [
                "Direction (Real)",
                "Direction (Augmented)",
                "Confidence UP (Real)",
                "Confidence UP (Augmented)",
                "Delta Confidence UP (Augmented - Real)",
                "1-day return (Real)",
                "1-day return (Augmented)",
                "Delta 1-day return (Augmented - Real)",
                "5-day return (Real)",
                "5-day return (Augmented)",
                "Delta 5-day return (Augmented - Real)",
            ],
            "Value": [
                real_out["direction_label"],
                aug_out["direction_label"],
                conf(real_out["p_up"]),
                conf(aug_out["p_up"]),
                "N/A" if delta_p_up is None else f"{delta_p_up * 100:.2f}%",
                pct(real_out["ret_1d"]),
                pct(aug_out["ret_1d"]),
                "N/A" if delta_1d is None else f"{delta_1d * 100:.2f}%",
                pct(real_out["ret_5d"]),
                pct(aug_out["ret_5d"]),
                "N/A" if delta_5d is None else f"{delta_5d * 100:.2f}%",
            ],
        }
    )

    st.dataframe(summary, use_container_width=True)

    st.markdown("## How to interpret differences")
    st.markdown("""
- If the **direction flips** between real and augmented, it means predictions are sensitive to training data and weak signals.
- If confidence changes by only a small amount, both models behave similarly.
- If regression outputs differ strongly, it suggests return magnitude prediction is unstable and highly noisy.
- A confidence near 50 percent means uncertainty is high, even if the label says UP or DOWN.
""")

    st.info("Tip: Try extreme and normal indicator values to see where the two model sets agree or disagree.")