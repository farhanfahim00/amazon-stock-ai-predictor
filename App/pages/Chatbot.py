import os
from pathlib import Path

import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import pandas as pd
import joblib

from chatbot.controller import chatbot_response


st.set_page_config(page_title="StockBot Voice Assistant", layout="wide")
st.title("StockBot Voice Assistant")
st.subheader("AI Assistant for the AMZN Stock Trend Predictor")

st.markdown("""
StockBot helps you understand predictions, indicators, and model behavior.
You can type or speak your questions and get answers based on the trained models and dataset.
""")

st.info(
    "Educational use only. This is not financial advice and should not be used for trading decisions."
)

BASE_DIR = Path(__file__).resolve().parents[2]


@st.cache_data
def load_data():
    return pd.read_csv(
        BASE_DIR / "Data" / "AMZN_data_with_features.csv",
        index_col="Date",
        parse_dates=True
    )


@st.cache_resource
def load_models():
    clf = joblib.load(BASE_DIR / "models" / "rf_classifier.joblib")
    reg_1d = joblib.load(BASE_DIR / "models" / "rf_1day_regressor.joblib")
    reg_5d = joblib.load(BASE_DIR / "models" / "rf_5day_regressor.joblib")
    return clf, reg_1d, reg_5d


data = load_data()
clf_model, reg_1d_model, reg_5d_model = load_models()


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = {}
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = False


# Voice functions

SPEECH_CORRECTIONS = {
    "amazon": "amzn",
    "a m z n": "amzn",
    "a m z": "amzn",
    "am z n": "amzn",
    "trend": "predict",
    "prediction": "predict",
    "predicted": "predict",
    "forecast": "predict",
    "tomorrow": "tomorrow",
    "next day": "tomorrow",
    "next-day": "tomorrow",
    "five day": "5-day",
    "5 day": "5-day",
    "one day": "1-day",
    "1 day": "1-day",
    "return": "return",
    "confidence": "confidence",
    "accuracy": "accuracy",
    "r squared": "r2",
    "r two": "r2",
    "feature importance": "top features",
    "important features": "top features",
    "top feature": "top features",
    "volatility": "volatility",
    "volatile": "volatility",
    "indicator": "indicator",
    "r s i": "rsi",
    "rsi": "rsi",
    "m a c d": "macd",
    "macd": "macd",
    "a t r": "atr",
    "atr": "atr",
    "bollinger": "bollinger",
    "bb position": "bb_position",
    "b b position": "bb_position",
    "repeat": "repeat",
}


def correct_speech_text(text: str) -> str:
    corrected = (text or "").lower().strip()
    for wrong, right in SPEECH_CORRECTIONS.items():
        corrected = corrected.replace(wrong, right)
    return corrected


def recognize_speech():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening, please speak now.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            recognizer.energy_threshold = 4000
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)

            text = recognizer.recognize_google(audio)
            corrected_text = correct_speech_text(text)

            if text.lower().strip() != corrected_text:
                st.info(f"Heard: '{text}' -> Corrected to: '{corrected_text}'")

            return {"success": True, "text": corrected_text}

    except sr.WaitTimeoutError:
        return {"success": False, "error": "No speech detected. Please try again."}
    except sr.UnknownValueError:
        return {"success": False, "error": "Could not understand audio. Please speak clearly."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def speak_response(text: str):
    try:
        clean_text = (
            (text or "")
            .replace("**", "")
            .replace("*", "")
            .replace("#", "")
            .replace("-", " ")
            .replace("\n", ". ")
        )

        tts = gTTS(text=clean_text, lang="en", slow=False)
        tts.save("response.mp3")

        with open("response.mp3", "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")

        os.remove("response.mp3")
    except Exception as e:
        st.error(f"Voice error: {e}")


# UI

st.session_state.voice_enabled = st.checkbox(
    "Enable voice responses",
    value=st.session_state.voice_enabled
)

# Display chat history (NO AUTO PLAY)
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show play button for assistant messages, user decides when to play
        if msg["role"] == "assistant" and st.session_state.voice_enabled:
            if st.button("Play voice", key=f"play_voice_{i}"):
                speak_response(msg["content"])

# Text input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    bot_reply = chatbot_response(
        user_input=user_input,
        data=data,
        clf_model=clf_model,
        reg_1d_model=reg_1d_model,
        reg_5d_model=reg_5d_model,
        context=st.session_state.context
    )

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.rerun()


st.markdown("---")

with st.expander("Voice Input Tips"):
    st.markdown("""
- Speak slowly and clearly
- Use short phrases like: "predict tomorrow", "predict 5-day return", "explain rsi", "top features"
- If a word is misheard, StockBot applies automatic corrections for common terms
""")

col1, col2 = st.columns(2)

with col1:
    if st.button("Voice Input", use_container_width=True):
        result = recognize_speech()
        if result["success"]:
            st.success(f"Heard: {result['text']}")

            st.session_state.messages.append({"role": "user", "content": result["text"]})

            bot_reply = chatbot_response(
                user_input=result["text"],
                data=data,
                clf_model=clf_model,
                reg_1d_model=reg_1d_model,
                reg_5d_model=reg_5d_model,
                context=st.session_state.context
            )

            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            st.rerun()
        else:
            st.error(result.get("error", "Speech recognition failed"))

with col2:
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context = {}
        st.rerun()


with st.expander("Example Questions & Capabilities"):
    st.markdown("""
### Prediction Queries
- Predict tomorrow
- Predict 1-day return
- Predict 5-day return
- How confident is the model

### Historical Price and Date-Based Queries
- Price on 2018-01-05
- Close price on 2021-12-09
- Volume on 2017-02-09

After asking for a date, you can follow up with:
- Predict tomorrow
- Predict 5-day return

### Technical Indicator Explanations
- Explain RSI
- What is MACD
- What is ATR
- What is BB_Position

### Model and Data Questions
- Top features
- Which days were most volatile
- Repeat the last prediction
- Help
""")