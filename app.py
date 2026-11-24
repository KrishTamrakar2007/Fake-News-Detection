import streamlit as st
import joblib
import re

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
)

model = joblib.load('fake_news_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.markdown(
    "<h1 style='text-align: center;'>Fake News Detection</h1>",
      unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>AI powered classification system</p>",
      unsafe_allow_html=True
)

st.divider()

st.subheader("Check your news")

news = st.text_area(
    "Enter your news article below:",
    height=200,
    placeholder="Paste your news article here...",
)

if st.button("Check News",
use_container_width=True):
    if news.strip():
        cleaned = clean_text(news)
        vector = tfidf_vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        fake_index = list(model.classes_).index(0)
        real_index = list(model.classes_).index(1)

        if prediction == 0:
            st.error("Result: Fake News")
            st.write(f"Fake News Probability: {probability[fake_index] * 100:.2f}%")

        else:
            st.success("Result: Real News")
            st.write(f"Real News Probability: {probability[real_index] * 100:.2f}%")

else:
        st.warning("Please enter some news first.")

st.divider()

st.caption("Collage Minor Project | Fake News Detection")
