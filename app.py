import streamlit as st
st.set_page_config(page_title="Kindle Review App")

import eda
import prediction

# --- Sidebar Navigation ---
with st.sidebar:
    st.image(
        "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExaHhiMWdpcXRqbDgyZnp2ems5dXNjYzMzemU3Y3ZiOXBncGVrdnR6aCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WoWm8YzFQJg5i/giphy.gif",
        width=180
    )
    st.markdown("## **Kindle Review Analyzer**")
    
    page = st.selectbox("Select Page", ["EDA", "Predict Sentiment"])

    st.markdown("---")
    st.markdown("#### About")
    st.write(
        """
        This app analyzes Kindle book reviews using NLP and a deep learning model. 
        You can classify custom reviews into 
        **Positive**, **Neutral**, or **Negative** sentiments. Try Now!
        """
    )

# --- Route to Selected Page ---
if page == "EDA":
    eda.run()
elif page == "Predict Sentiment":
    prediction.run()
