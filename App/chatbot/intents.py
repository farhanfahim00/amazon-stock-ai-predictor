"""
Intent detection for StockBot.

Simple keyword based intent detection.
This is enough for the course and is explainable in the Wiki.
"""

from __future__ import annotations

INTENT_KEYWORDS = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening", "greetings"],
    "small_talk": ["how are you", "who are you", "what is your name", "thanks", "thank you", "bye", "goodbye"],
    "help": ["help", "what can you do", "commands", "how to use", "capabilities"],

    "price_on_date": ["price on", "price for", "close on", "close for", "open on", "high on", "low on", "volume on", "on date", "on the date"],

    "next_day_prediction": ["tomorrow", "next day", "next-day", "direction", "up or down"],
    "five_day_prediction": ["5 day", "5-day", "five day", "five-day", "5d", "weekly", "next week"],
    "one_day_return": ["1 day", "1-day", "one day", "1d return", "next day return"],

    "feature_importance": ["feature importance", "important features", "top features", "influential features"],
    "explain_indicator": ["rsi", "macd", "atr", "bollinger", "bb_position", "bb position", "indicator"],

    "historical_return": ["historical return", "last week", "last month", "performance", "what happened"],
    "volatility_analysis": ["volatility", "volatile", "highest volatility", "extreme", "atr spikes"],

    "model_confidence": ["confidence", "trust", "accurate", "reliable", "metrics", "score", "r2", "accuracy"],
    "repeat_last": ["repeat", "again", "last prediction", "say again"],
}


def detect_intent(user_input: str) -> str:
    text = (user_input or "").lower().strip()

    if not text:
        return "unknown"

    priority = [
        "price_on_date",
        "five_day_prediction",
        "one_day_return",
        "next_day_prediction",
        "feature_importance",
        "explain_indicator",
        "historical_return",
        "volatility_analysis",
        "model_confidence",
        "repeat_last",
        "help",
        "small_talk",
        "greeting",
    ]

    for intent in priority:
        keywords = INTENT_KEYWORDS.get(intent, [])
        if any(k in text for k in keywords):
            return intent

    return "unknown"