import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

# ========== MODEL RECONSTRUCTION ==========

@st.cache_resource
def load_model():
    # USE Layer
    hub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    use_layer = hub.KerasLayer(hub_url, input_shape=[], dtype=tf.string, trainable=False)

    # Rebuild model
    input_text = tf.keras.Input(shape=(1,), dtype=tf.string, name='input_text')
    x = tf.keras.layers.Lambda(lambda x: use_layer(x))(input_text)
    x = tf.keras.layers.Reshape((512, 1))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_text, outputs=output)

    # Load weights (not full model)
    model.load_weights("model_lstm_2_sw.weights.h5")  # Save weights separately

    return model

model = load_model()

# ========== PREDICTION FUNCTION ==========

def predict_sentiment(text):
    input_array = np.array([text], dtype=object)
    prediction = model.predict(input_array)
    labels = ['Negative', 'Neutral', 'Positive']
    return labels[np.argmax(prediction)], prediction

# ========== STREAMLIT UI ==========

st.title("Kindle Review Sentiment Predictor")

user_input = st.text_area("Enter your Kindle book review:")
if st.button("Predict"):
    if user_input.strip():
        label, probs = predict_sentiment(user_input)
        st.write(f"**Predicted Sentiment:** {label}")
        st.write("**Confidence:**")
        for i, prob in enumerate(probs[0]):
            st.write(f"- {['Negative', 'Neutral', 'Positive'][i]}: {prob:.4f}")
    else:
        st.warning("Please enter some text.")
