import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from textwrap import wrap

def run():
    st.title("Amazon Kindle Review Sentiment Analysis - Exploratory Data Analysis (EDA)")

    # HEADER IMAGE
    st.markdown("""
    <div style='text-align: center;'>
        <img src='https://sm.pcmag.com/pcmag_uk/review/a/amazon-kin/amazon-kindle-paperwhite_53ba.jpg' width='400'>
    </div>
    """, unsafe_allow_html=True)

    # DESCRIPTION
    st.write("### Description")
    st.write("""
    In the digital publishing industry, customer reviews have become a key factor in shaping consumer behavior and influencing purchase decisions. 
    To address the challenge of extracting insights from a large volume of reviews, this project proposes an NLP-based sentiment analysis system that classifies Kindle book reviews into Positive, Neutral, or Negative sentiments.
    """)

    # LOAD DATA
    data = pd.read_csv('Deployment/preprocessed_kindle_review .csv')

    
    
    # Create Sentiment Column based on Rating
    if 'Sentiment' not in data.columns and 'rating' in data.columns:
        def convert_rating_to_sentiment(rating):
            if rating >= 4:
                return "Positive"
            elif rating == 3:
                return "Neutral"
            elif rating >= 1:
                return "Negative"
            else:
                return "Unknown"
        data['Sentiment'] = data['rating'].apply(convert_rating_to_sentiment)

    # 1. Sentiment Distribution
    st.header("4.2 Sentiment Distribution")
    sentiment_counts = data['Sentiment'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, startangle=140, autopct='%1.1f%%')
    ax1.set_title('Sentiment Distribution')
    ax1.axis('equal')
    st.pyplot(fig1)
    st.markdown("Positive reviews make up the largest portion at 49.7%, followed by negative at 33.6%, and neutral at 16.8%.")

    # 2. Rating Distribution
    st.header("4.3 Rating Distribution")
    pct = data['rating'].value_counts(normalize=True).sort_index() * 100
    labels = [f'Rating = {i}' for i in range(1, 6)]
    sizes = [pct.get(i, 0) for i in range(1, 6)]
    palette = sns.color_palette("pastel", 5)
    explode = (0, 0, 0, 0, 0.1)
    fig2, ax2 = plt.subplots()
    ax2.pie(sizes, explode=explode, labels=labels, colors=palette,
            autopct='%1.1f%%', shadow=True, startangle=140)
    ax2.set_title('Pie Chart of Kindle Review Ratings')
    ax2.axis('equal')
    st.pyplot(fig2)
    st.markdown("Ratings 4 and 5 are the most frequent, each making up about 25% of the data. Ratings 1 and 2 account for roughly one-third of the data, while rating 3 makes up around 17%.")

    # 3. Word Cloud
    st.header("4.4 Word Cloud of Review Texts")
    all_text = ' '.join(data['reviewText'].dropna())
    wc = WordCloud(width=800, height=400, max_words=150, colormap='Dark2').generate(all_text)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.imshow(wc, interpolation='bilinear')
    ax3.axis('off')
    ax3.set_title('\n'.join(wrap("Word Cloud of Kindle Reviews", 60)), fontsize=14)
    st.pyplot(fig3)
    st.markdown("""
    The word cloud shows the most common words found in the Kindle reviews. 
    Words like **“book”**, **“story”**, **“read”**, and **“character”** appear frequently, showing that readers often discuss the plot and their reading experience.
    We also see positive words such as **“love”**, **“great”**, and **“enjoy”**, which matches the sentiment distribution where most reviews are labeled as **positive**.
    """)

    # 4. Most Frequent Words
    st.header("4.5 Most Frequently Occurring Words")
    corpus = data['reviewText'].dropna().tolist()
    words = []
    for review in corpus:
        words.extend(review.split())
    most_common_words = Counter(words).most_common(10)
    word_labels = [item[0] for item in most_common_words]
    word_freqs = [item[1] for item in most_common_words]
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=word_freqs, y=word_labels, ax=ax4)
    ax4.set_title('Top 10 Most Frequently Occurring Words')
    ax4.set_xlabel('Frequency')
    ax4.set_ylabel('Words')
    st.pyplot(fig4)
    st.markdown("""
    The bar chart shows the top 10 most frequently used words in the review dataset. 
    As expected, common English words like “the,” “and,” “to,” and “a” dominate, with “the” alone appearing nearly 24,000 times. 
    This show that the majority of review in the data is filled with stopwrods. Therefore, removing stopwords can influence the overall model's performance. 
    """)

if __name__ == '__main__':
    run()
