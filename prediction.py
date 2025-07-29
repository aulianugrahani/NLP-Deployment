# prediction.py
import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import builtins
import os

# Recreate the TF Hub layer
hub_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    input_shape=[],
    dtype=tf.string,
    trainable=False
)

builtins.hub_layer = hub_layer
builtins.tf = tf

# Load the model safely
@st.cache_resource
def load_lstm_model():
    model_path = "model_lstm_2_sw.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the file exists.")
        st.stop()
    return load_model(model_path, safe_mode=False)

# Predict function
@st.cache_data
def predict_sentiment(text):
    model = load_lstm_model()
    input_array = np.array([text], dtype=object)
    prediction = model.predict(input_array)
    pred_class = np.argmax(prediction)
    label_map = ['Negative', 'Neutral', 'Positive']
    return label_map[pred_class], prediction[0][pred_class]

# --- Main function for use in app.py ---
def run():
    st.title("📚 Kindle Review Sentiment Classification")

    st.write("""
    Enter a Kindle book review below. The model will analyze the text and classify it as **Positive**, **Neutral**, or **Negative**.
    """)

    user_input = st.text_area("Your Review", height=150)

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a review before predicting.")
        else:
            label, prob = predict_sentiment(user_input)
            st.subheader("Prediction Result")
            st.write(f"**Sentiment:** {label}")
            st.write(f"**Confidence Score:** {round(prob * 100, 2)}%")
