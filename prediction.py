# prediction.py

import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import os

# Set up the TF Hub layer (Universal Sentence Encoder)
USE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
hub_layer = hub.KerasLayer(USE_URL, input_shape=[], dtype=tf.string, trainable=False)

# Make the hub layer and tf available globally (if needed in Lambda layers)
import builtins
builtins.hub = hub
builtins.tf = tf

# Load model
@st.cache_resource
def load_lstm_model():
    model_path = "model_lstm_2_sw.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please ensure the file is in the correct directory.")
        st.stop()
    return load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

# Predict sentiment
@st.cache_data
def predict_sentiment(text):
    model = load_lstm_model()
    input_array = np.array([text], dtype=object)
    prediction = model.predict(input_array)
    predicted_class = np.argmax(prediction)
    label_map = ['Negative', 'Neutral', 'Positive']
    return label_map[predicted_class], prediction[0][predicted_class]

#Streamlit UI 
def run():
    st.title("📚 Kindle Review Sentiment Classification")

    st.write("""
    Enter a Kindle book review in the text box below. The model will analyze the content and classify it as **Positive**, **Neutral**, or **Negative**.
    """)

    user_input = st.text_area("Your Review", height=150)

    if st.button("Predict Sentiment"):
        if not user_input.strip():
            st.warning("Please enter a review before predicting.")
        else:
            label, confidence = predict_sentiment(user_input)
            st.subheader("Prediction Result")
            st.write(f"**Sentiment:** {label}")
            st.write(f"**Confidence Score:** {round(confidence * 100, 2)}%")
