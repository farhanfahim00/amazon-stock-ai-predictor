"""
Main chatbot controller.
Routes a user message to an intent and response handler.
"""

from __future__ import annotations

from typing import Any, Dict

from chatbot.intents import detect_intent
from chatbot.responses import (
    handle_explain_indicator,
    handle_feature_importance,
    handle_five_day_return,
    handle_greeting,
    handle_help,
    handle_historical_return,
    handle_model_confidence,
    handle_next_day_prediction,
    handle_one_day_return,
    handle_price_on_date,
    handle_repeat_last,
    handle_small_talk,
    handle_unknown,
    handle_volatility_analysis,
)


def chatbot_response(
    user_input: str,
    data,
    clf_model,
    reg_1d_model,
    reg_5d_model,
    context: Dict[str, Any],
) -> str:
    intent = detect_intent(user_input)

    if context is None:
        context = {}

    context["last_intent"] = intent
    context["last_user_input"] = user_input

    if intent == "greeting":
        return handle_greeting()

    if intent == "small_talk":
        return handle_small_talk(user_input)

    if intent == "help":
        return handle_help()

    if intent == "price_on_date":
        return handle_price_on_date(user_input, data, context)

    if intent == "next_day_prediction":
        return handle_next_day_prediction(data, clf_model, context)

    if intent == "one_day_return":
        return handle_one_day_return(data, reg_1d_model, context)

    if intent == "five_day_prediction":
        return handle_five_day_return(data, reg_5d_model, context)

    if intent == "feature_importance":
        return handle_feature_importance(clf_model)

    if intent == "explain_indicator":
        return handle_explain_indicator(user_input)

    if intent == "historical_return":
        return handle_historical_return(data)

    if intent == "volatility_analysis":
        return handle_volatility_analysis(data)

    if intent == "model_confidence":
        return handle_model_confidence()

    if intent == "repeat_last":
        return handle_repeat_last(context)

    return handle_unknown()