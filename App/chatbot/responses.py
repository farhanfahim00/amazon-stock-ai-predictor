"""
Response handlers for StockBot.
All responses are explainable and aligned with the assistance system goal.
"""

from __future__ import annotations

from typing import Any, Dict, List
import re

import numpy as np
import pandas as pd


FEATURE_COLUMNS: List[str] = [
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


def _format_date(idx: Any) -> str:
    try:
        return idx.strftime("%Y-%m-%d")
    except Exception:
        return str(idx)


def _latest_row_features(data: pd.DataFrame) -> pd.DataFrame:
    return data[FEATURE_COLUMNS].iloc[-1:].copy()


def extract_date_from_text(text: str):
    """
    Accepts formats like:
    - 2018-01-05
    - 2018/01/05
    - 05-01-2018 (day first)
    """
    t = (text or "").strip()

    m = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", t)
    if m:
        try:
            return pd.to_datetime(m.group(1).replace("/", "-"))
        except Exception:
            return None

    m2 = re.search(r"(\d{2}[-/]\d{2}[-/]\d{4})", t)
    if m2:
        try:
            return pd.to_datetime(m2.group(1).replace("/", "-"), dayfirst=True)
        except Exception:
            return None

    return None


def get_row_for_date(data: pd.DataFrame, requested_date: pd.Timestamp):
    """
    Returns the row for the requested trading date.
    If exact date is missing, returns the nearest previous trading day.
    """
    if data is None or data.empty:
        return None, None, "No data loaded. Please check the dataset path."

    if requested_date is None:
        return None, None, "Please provide a date like 2018-01-05."

    # Force index into a clean DatetimeIndex (handles tz-aware objects safely)
    try:
        idx_dt_utc = pd.to_datetime(data.index, utc=True, errors="coerce")
    except Exception:
        return None, None, "Date index could not be parsed. Please check the dataset Date format."

    if idx_dt_utc.isna().any():
        return None, None, "Some dates in the dataset index are invalid. Please check the Date column formatting."

    # tz-naive version for safe comparisons with user input
    idx_naive = pd.DatetimeIndex(idx_dt_utc).tz_convert(None)

    # User provided date should also be tz-naive
    try:
        requested_naive = pd.to_datetime(requested_date).tz_localize(None)
    except Exception:
        requested_naive = pd.to_datetime(requested_date)

    # Exact match
    if requested_naive in idx_naive:
        pos = idx_naive.get_loc(requested_naive)
        used_date = data.index[pos]
        return data.iloc[pos], pd.to_datetime(used_date), None

    # Nearest previous trading day
    prev_dates = idx_naive[idx_naive <= requested_naive]
    if len(prev_dates) == 0:
        first_date = idx_naive.min()
        return None, None, f"No earlier trading data exists. Earliest available date is {_format_date(first_date)}."

    nearest_naive = prev_dates.max()
    pos = idx_naive.get_loc(nearest_naive)

    used_date = data.index[pos]
    note = (
        f"No trading data on {_format_date(requested_naive)}. "
        f"Using nearest previous trading day {_format_date(nearest_naive)}."
    )

    return data.iloc[pos], pd.to_datetime(used_date), note


def features_for_context_date(data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
    """
    If the user previously asked for a specific date, use that row's features.
    Otherwise use the latest row.
    """
    if context is None:
        return _latest_row_features(data)

    d = context.get("selected_date")
    if d is None:
        return _latest_row_features(data)

    if d in data.index:
        return data.loc[[d], FEATURE_COLUMNS].copy()

    return _latest_row_features(data)


def handle_help() -> str:
    return (
        "I can help with:\n"
        "- Price on a specific date (OHLCV)\n"
        "- Next day direction prediction (UP or DOWN)\n"
        "- 1-day and 5-day return predictions\n"
        "- Explaining indicators like RSI, MACD, ATR, Bollinger Bands\n"
        "- Feature importance\n"
        "- Historical return and volatility analysis\n\n"
        "Try:\n"
        "- 'price on 2018-01-05'\n"
        "- 'predict tomorrow'\n"
        "- 'predict 5-day return'\n"
        "- 'explain RSI'\n"
        "- 'top features'"
    )


def handle_greeting() -> str:
    return "Hello! I am StockBot. Ask me for AMZN prices by date, predictions, indicator explanations, or model metrics."


def handle_small_talk(user_input: str) -> str:
    t = (user_input or "").lower()
    if "how are you" in t:
        return "Doing well. Ready to help with AMZN predictions and explanations."
    if "who are you" in t or "name" in t:
        return "I am StockBot, the assistant for the AI Stock Trend Predictor."
    if "thank" in t:
        return "You are welcome."
    if "bye" in t:
        return "Goodbye. Come back anytime."
    return "I am here to help with AMZN predictions and explanations."


def handle_price_on_date(user_input: str, data: pd.DataFrame, context: Dict[str, Any]) -> str:
    requested = extract_date_from_text(user_input)
    row, used_date, note = get_row_for_date(data, requested)

    if row is None:
        return note or "Could not find data for that date."

    context["selected_date"] = used_date

    open_p = float(row["Open"]) if "Open" in row else None
    high_p = float(row["High"]) if "High" in row else None
    low_p = float(row["Low"]) if "Low" in row else None
    close_p = float(row["Close"]) if "Close" in row else None
    volume = int(row["Volume"]) if "Volume" in row else None

    lines = []
    if note:
        lines.append(note)

    lines.append(f"AMZN on {_format_date(used_date)}:")
    if open_p is not None:
        lines.append(f"- Open: ${open_p:.2f}")
    if high_p is not None:
        lines.append(f"- High: ${high_p:.2f}")
    if low_p is not None:
        lines.append(f"- Low: ${low_p:.2f}")
    if close_p is not None:
        lines.append(f"- Close: ${close_p:.2f}")
    if volume is not None:
        lines.append(f"- Volume: {volume:,}")

    lines.append("Now you can ask: predict tomorrow (it will be relative to this date).")
    return "\n".join(lines)


def handle_next_day_prediction(data: pd.DataFrame, clf_model, context: Dict[str, Any]) -> str:
    if clf_model is None:
        return "Model not loaded. Please check that rf_classifier.joblib is available."

    if data is None or data.empty:
        return "No data loaded. Please check the dataset path."

    missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
    if missing:
        return f"Missing feature columns in data: {', '.join(missing)}"

    X = features_for_context_date(data, context)
    base_date = X.index[0] if hasattr(X, "index") else data.index[-1]
    base_date_str = _format_date(base_date)

    pred = int(clf_model.predict(X)[0])
    direction = "UP" if pred == 1 else "DOWN"

    confidence = None
    try:
        prob = clf_model.predict_proba(X)[0]
        confidence = float(prob[1] if pred == 1 else prob[0])
    except Exception:
        pass

    if confidence is None:
        msg = f"Next day direction prediction (relative to {base_date_str}): {direction}."
    else:
        msg = f"Next day direction prediction (relative to {base_date_str}): {direction} (confidence {confidence*100:.2f}%)."

    context["last_prediction"] = msg
    return msg


def handle_one_day_return(data: pd.DataFrame, reg_1d_model, context: Dict[str, Any]) -> str:
    if reg_1d_model is None:
        return "1-day regression model not loaded. Please check the joblib file."

    if data is None or data.empty:
        return "No data loaded. Please check the dataset path."

    X = features_for_context_date(data, context)
    base_date = X.index[0] if hasattr(X, "index") else data.index[-1]

    pred = float(reg_1d_model.predict(X)[0])
    msg = f"Predicted 1-day return (relative to {_format_date(base_date)}): {pred*100:.2f}%."

    context["last_prediction"] = msg
    return msg


def handle_five_day_return(data: pd.DataFrame, reg_5d_model, context: Dict[str, Any]) -> str:
    if reg_5d_model is None:
        return "5-day regression model not loaded. Please check the joblib file."

    if data is None or data.empty:
        return "No data loaded. Please check the dataset path."

    X = features_for_context_date(data, context)
    base_date = X.index[0] if hasattr(X, "index") else data.index[-1]

    pred = float(reg_5d_model.predict(X)[0])
    msg = f"Predicted 5-day return (relative to {_format_date(base_date)}): {pred*100:.2f}%."

    context["last_prediction"] = msg
    return msg


def handle_feature_importance(clf_model) -> str:
    if clf_model is None:
        return "Model not loaded, cannot compute feature importance."

    try:
        importances = clf_model.feature_importances_
    except Exception:
        return "This model does not expose feature importance."

    pairs = list(zip(FEATURE_COLUMNS, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:5]

    lines = ["Top 5 important features:"]
    for name, val in top:
        lines.append(f"- {name}: {val*100:.2f}%")
    return "\n".join(lines)


def handle_explain_indicator(user_input: str) -> str:
    t = (user_input or "").lower()

    if "rsi" in t:
        return (
            "RSI (Relative Strength Index) is a momentum indicator from 0 to 100.\n"
            "- Above 70: often interpreted as overbought\n"
            "- Below 30: often interpreted as oversold\n"
            "It helps summarize recent gains versus losses."
        )

    if "macd" in t:
        return (
            "MACD (Moving Average Convergence Divergence) measures trend momentum.\n"
            "MACD is the difference between two exponential moving averages.\n"
            "MACD_Diff shows the gap between MACD and its signal line."
        )

    if "atr" in t:
        return (
            "ATR (Average True Range) measures volatility.\n"
            "High ATR means larger daily price ranges. Low ATR means calmer price movement."
        )

    if "bollinger" in t or "bb" in t or "bb_position" in t:
        return (
            "BB_Position tells where the price sits between Bollinger Bands.\n"
            "0 means near the lower band, 1 means near the upper band.\n"
            "It can indicate relative overbought or oversold conditions."
        )

    return "Which indicator do you want? Try RSI, MACD, ATR, or Bollinger Bands."


def handle_historical_return(data: pd.DataFrame) -> str:
    if data is None or data.empty:
        return "No data loaded."

    if "Close" not in data.columns:
        return "Close column is missing."

    if len(data) < 6:
        return "Not enough data for a 5-trading-day return."

    weekly_return = ((data["Close"].iloc[-1] / data["Close"].iloc[-6]) - 1) * 100
    last_close = float(data["Close"].iloc[-1])
    last_date = _format_date(data.index[-1])

    return (
        f"Last close ({last_date}): ${last_close:.2f}\n"
        f"Last 5 trading days return: {weekly_return:.2f}%."
    )


def handle_volatility_analysis(data: pd.DataFrame) -> str:
    if data is None or data.empty:
        return "No data loaded."

    if "ATR" not in data.columns or "Close" not in data.columns:
        return "ATR or Close column is missing."

    top = data.nlargest(3, "ATR")[["Close", "ATR"]]
    lines = ["Highest volatility days (by ATR):"]
    for i in range(min(3, len(top))):
        date = _format_date(top.index[i])
        lines.append(f"- {date}: ATR {float(top['ATR'].iloc[i]):.2f}, Close ${float(top['Close'].iloc[i]):.2f}")
    return "\n".join(lines)


def handle_model_confidence() -> str:
    return (
        "Model confidence guidance:\n"
        "- Classification accuracy slightly above 50% is common for daily stock direction.\n"
        "- Regression R2 near 0 means exact returns are hard to predict.\n"
        "Use predictions as educational signals, not trading advice."
    )


def handle_repeat_last(context: Dict[str, Any]) -> str:
    last = context.get("last_prediction") if context else None
    if not last:
        return "No previous prediction to repeat."
    return f"Last result: {last}"


def handle_unknown() -> str:
    return (
        "I am not sure I understood.\n"
        "Try:\n"
        "- price on 2018-01-05\n"
        "- predict tomorrow\n"
        "- predict 1-day return\n"
        "- predict 5-day return\n"
        "- explain RSI\n"
        "- top features"
    )