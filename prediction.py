import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load Universal Sentence Encoder (USE) from TensorFlow Hub
@st.cache_resource
def load_use():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

use_model = load_use()

# Define preprocessing and inference pipeline
def preprocess(text):
    # Embed the input using USE
    embeddings = use_model([text])  # shape: (1, 512)
    return embeddings.numpy()

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_lstm_2_sw.keras', compile=True)

model = load_model()

# Streamlit UI
st.title("Sentiment Prediction App")

user_input = st.text_area("Enter text to classify sentiment", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a valid sentence.")
    else:
        emb = preprocess(user_input)
        emb = emb.reshape((1, 512, 1))  # Match the expected input shape
        prediction = model.predict(emb)
        class_idx = np.argmax(prediction)
        sentiment = ["Negative", "Neutral", "Positive"][class_idx]
        st.success(f"Predicted Sentiment: **{sentiment}**")
