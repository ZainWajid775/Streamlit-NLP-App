import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Load model and vectorizer
MODEL_PATH = "D:/Codes/NLP Incremental/sentiment_model.pkl"
VECTORIZER_PATH = "D:/Codes/NLP Incremental/vectorizer.pkl"
CORRECTIONS_PATH = "D:/Codes/NLP Incremental/corrections.csv"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Streamlit page config
st.set_page_config(page_title="🧠 Sentiment Trainer", layout="centered")
st.markdown("<h1 style='text-align:center;'>💬 Sentiment Analyzer & Trainer</h1>", unsafe_allow_html=True)
st.caption("Help the model improve by providing feedback on its predictions!")
st.divider()

# Choose input method
input_mode = st.radio("Choose Input Mode:", ["📄 Upload File", "📝 Type Manually"], horizontal=True)

text = ""
if input_mode == "📄 Upload File":
    uploaded = st.file_uploader("Upload a .txt or .csv file", type=['txt', 'csv'])
    if uploaded:
        if uploaded.name.endswith('.txt'):
            text = uploaded.read().decode("utf-8")
        elif uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
            if "review" in df.columns:
                text = df["review"].iloc[0]
            else:
                st.warning("CSV must contain a 'review' column.")
else:
    text = st.text_area("✍️ Enter a review here", height=150)

# Initialize session state
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

# Analyze button
if st.button("🔍 Analyze Sentiment"):
    if not text.strip():
        st.warning("Please enter or upload some text.")
    else:
        X_input = vectorizer.transform([text])
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        confidence = np.max(proba)

        st.session_state.prediction_made = True
        st.session_state.prediction = prediction
        st.session_state.confidence = confidence
        st.session_state.X_input = X_input
        st.session_state.text_input = text

# Show result
if st.session_state.prediction_made:
    label = "Positive" if st.session_state.prediction == 1 else "Negative"
    color = "#33FF57" if label == "Positive" else "#FF3355"

    st.markdown(
        f"""
        <div style='background-color:#222; padding:20px; border-radius:10px; border-left:8px solid {color}'>
            <h3 style='color:{color}'>Prediction: {label}</h3>
            <p style='color:white;'>Confidence: {st.session_state.confidence * 100:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("### ❓ Was this prediction correct?")
    feedback = st.radio("Your feedback:", ["Yes", "No"], horizontal=True)

    if feedback == "No":
        correction = st.selectbox("✅ What should it have been?", ["Positive", "Negative"])
        if st.button("🧠 Teach the Model"):
            true_label = correction  # "Positive" or "Negative"
            numeric_label = 1 if true_label == "Positive" else 0

            # Retrain the model
            model.partial_fit(st.session_state.X_input, [numeric_label])
            joblib.dump(model, MODEL_PATH)

            # Save correction
            correction_record = pd.DataFrame({
                "text": [st.session_state.text_input],
                "label": [true_label]
            })
            if os.path.exists(CORRECTIONS_PATH):
                old = pd.read_csv(CORRECTIONS_PATH)
                combined = pd.concat([old, correction_record], ignore_index=True)
            else:
                combined = correction_record
            combined.to_csv(CORRECTIONS_PATH, index=False)

            st.success("✅ Thanks! The model has been updated and your correction was saved.")

# Show previous corrections
with st.expander("📊 View Training Corrections", expanded=False):
    if os.path.exists(CORRECTIONS_PATH):
        df_corrections = pd.read_csv(CORRECTIONS_PATH)
        st.dataframe(df_corrections, use_container_width=True)
        st.info(f"🧠 Total corrections: {len(df_corrections)}")
    else:
        st.info("No corrections have been made yet.")
