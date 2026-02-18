"""
Example dialogs for StockBot.
These can be copied into the Wiki as the required example dialogs.
"""

from __future__ import annotations

EXAMPLE_DIALOGS = [
    {
        "user": "What is the predicted trend for Amazon tomorrow?",
        "bot": "Next day direction prediction: UP (confidence 52.10%).",
    },
    {
        "user": "Predict 5-day return for Amazon.",
        "bot": "Predicted 5-day return: 0.62%.",
    },
    {
        "user": "Predict 1-day return.",
        "bot": "Predicted 1-day return: -0.15%.",
    },
    {
        "user": "Show me the top features affecting predictions.",
        "bot": "Top 5 important features: ATR, Price_Change_Pct, BB_Position, Volume_Change, MACD.",
    },
    {
        "user": "Explain RSI and how it affects the prediction.",
        "bot": "RSI measures momentum from 0 to 100. Above 70 can mean overbought, below 30 oversold.",
    },
    {
        "user": "What is MACD?",
        "bot": "MACD measures trend momentum using exponential moving averages. MACD_Diff compares it to its signal.",
    },
    {
        "user": "Which days had the highest volatility?",
        "bot": "Highest volatility days (by ATR) were during major market stress events.",
    },
    {
        "user": "Can I trust the regression prediction?",
        "bot": "Exact returns are difficult to predict. Treat regression as an educational estimate, not a guarantee.",
    },
    {
        "user": "How confident are you about the next-day trend?",
        "bot": "Confidence near 50% means uncertainty. 55 to 60% is a weak signal. Above 60% is rare.",
    },
    {
        "user": "Repeat the last prediction.",
        "bot": "Last result: Next day direction prediction: UP (confidence 52.10%).",
    },
]