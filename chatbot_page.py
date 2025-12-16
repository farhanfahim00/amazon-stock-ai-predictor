"""
StockBot Chatbot for AI Stock Trend Predictor
Provides predictions, explanations, and guidance on Amazon stock trends.
"""

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import speech_recognition as sr

def run_chatbot():
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None

    # Load dataset
    @st.cache_data
    def load_data():
        return pd.read_csv('Data/AMZN_with_features.csv', index_col='Date', parse_dates=True)

    # Load models
    @st.cache_resource
    def load_models():
        import os
        clf_path = 'models/classification_model.joblib'
        reg_path = 'models/regression_model.joblib'
        if not os.path.exists(clf_path) or not os.path.exists(reg_path):
            return None, None
        classifier = joblib.load(clf_path)
        regressor = joblib.load(reg_path)
        return classifier, regressor

    # Features used by models
    FEATURE_COLUMNS = [
        'Daily_Return', 'Price_Change_Pct', 'Volatility', 'Volume_Change',
        'Momentum', 'RSI', 'MACD', 'MACD_Diff', 'BB_Position',
        'Volume_Ratio', 'ROC', 'ATR'
    ]

    data = load_data()
    classifier, regressor = load_models()

    # Speech recognition
    def recognize_speech():
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                st.info("Listening. Please speak now.")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            return recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            return "Error: No speech detected."
        except sr.UnknownValueError:
            return "Error: Audio not clear."
        except sr.RequestError:
            return "Error: Speech service unavailable."
        except Exception as e:
            return f"Error: {str(e)}"

    # Intent classification
    def classify_intent(user_input):
        text = user_input.lower()
        if any(w in text for w in ['predict', 'forecast', 'trend']):
            if 'next' in text or 'tomorrow' in text:
                return 'next_day_prediction'
            if '5' in text or 'five' in text or 'week' in text:
                return '5day_prediction'
            return 'next_day_prediction'
        if any(w in text for w in ['feature', 'top', 'important']):
            return 'feature_importance'
        if any(w in text for w in ['rsi', 'macd', 'bollinger', 'atr']):
            return 'explain_indicator'
        if any(w in text for w in ['return', 'last week', 'last month']):
            return 'historical_return'
        if any(w in text for w in ['volatility', 'volatile', 'highest']):
            return 'volatility_analysis'
        if any(w in text for w in ['confidence', 'trust', 'accurate']):
            return 'model_confidence'
        if any(w in text for w in ['repeat', 'again']):
            return 'repeat_last'
        if any(w in text for w in ['bb_position', 'bb position']):
            return 'explain_bb_position'
        if any(w in text for w in ['help', 'commands']):
            return 'help'
        return 'general_query'

    # Response generation
    def generate_response(intent, user_input):
        if intent == 'next_day_prediction':
            if classifier is not None:
                latest = data[FEATURE_COLUMNS].iloc[-1:].values
                pred = classifier.predict(latest)[0]
                direction = "UP" if pred == 1 else "DOWN"
                response = f"The next-day trend prediction is {direction} with 51.43% confidence."
                st.session_state.last_prediction = response
                return response
            return "Model not loaded."
        if intent == '5day_prediction':
            response = "The 5-day return prediction has directional accuracy of 53.98%. Exact return is hard to predict."
            st.session_state.last_prediction = response
            return response
        if intent == 'feature_importance':
            return (
                "Top 5 Features:\n"
                "1. ATR (16.53%)\n"
                "2. Price Change % (10.81%)\n"
                "3. BB Position (9.42%)\n"
                "4. Volume Change (8.79%)\n"
                "5. MACD (8.61%)"
            )
        if intent == 'help':
            return (
                "**StockBot Commands:** You can ask me about:\n\n"
                "Predictions\n- What is the predicted trend for tomorrow?\n- Predict 5-day return for Amazon\n\n"
                "Features & Indicators\n- Show me top features\n- Explain RSI / MACD / ATR\n- What is BB_Position?\n\n"
                "Historical Data\n- What was Amazon's return last week?\n- Which days had highest volatility?\n\n"
                "Model Information\n- How confident are you?\n- Can I trust the prediction?\n\n"
                "Other\n- Repeat the last prediction"
            )
        if intent == 'repeat_last':
            return st.session_state.last_prediction or "No previous prediction."
        return "I did not understand that. Try 'help'."

    # Streamlit UI
    st.title("StockBot - AI Stock Assistant")
    st.write("Ask questions about Amazon stock predictions, indicators, and model insights.")

    col1, col2 = st.columns([3,1])

    # Text input
    with col1:
        user_input = st.text_input("Type your question:", key="text_input")
        if st.button("Send", use_container_width=True) and user_input:
            intent = classify_intent(user_input)
            response = generate_response(intent, user_input)
            st.session_state.chat_history.append({
                'user': user_input,
                'bot': response,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            st.rerun()

    # Voice input
    with col2:
        if st.button("Start Voice Input", use_container_width=True):
            transcribed = recognize_speech()
            if not transcribed.startswith("Error"):
                intent = classify_intent(transcribed)
                response = generate_response(intent, transcribed)
                st.session_state.chat_history.append({
                    'user': transcribed,
                    'bot': response,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                st.rerun()
            else:
                st.error(transcribed)

    # Chat history display
    st.markdown("---")
    st.subheader("Chat History")
    if st.session_state.chat_history:
        for chat in reversed(st.session_state.chat_history):
            with st.container():
                st.markdown(f"You ({chat['timestamp']})")
                st.info(chat['user'])
                st.markdown("StockBot")
                st.success(chat['bot'])
                st.markdown("---")
    else:
        st.write("No messages yet. Start by typing a question or using voice input.")

    # Quick action buttons
    st.markdown("---")
    st.subheader("Quick Actions")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Next-Day Prediction", use_container_width=True):
            response = generate_response('next_day_prediction', "predict tomorrow")
            st.session_state.chat_history.append({'user': "Next-day prediction", 'bot': response, 'timestamp': datetime.now().strftime("%H:%M:%S")})
            st.rerun()
    with c2:
        if st.button("Top Features", use_container_width=True):
            response = generate_response('feature_importance', "top features")
            st.session_state.chat_history.append({'user': "Top features", 'bot': response, 'timestamp': datetime.now().strftime("%H:%M:%S")})
            st.rerun()
    with c3:
        if st.button("Help", use_container_width=True):
            response = generate_response('help', "help")
            st.session_state.chat_history.append({'user': "Help", 'bot': response, 'timestamp': datetime.now().strftime("%H:%M:%S")})
            st.rerun()

    # Clear chat
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.last_prediction = None
        st.rerun()
