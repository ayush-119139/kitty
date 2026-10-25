import re
import joblib
import numpy as np
import streamlit as st
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
stemmer = PorterStemmer()
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text):
    text = str(text)
    text = re.sub(r"^.{0,80}?\(Reuters\)\s*-\s*", "", text)
    text = re.sub(r"\(Reuters\)", " ", text, flags=re.I)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [stemmer.stem(w) for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)


st.title("VeriNews AI - Fake News Detector")
text = st.text_area("Paste the news text here", height=200)

col1, col2 = st.columns(2)
verify = col1.button("Verify Authenticity")

if verify:
    if len(text.split()) < 5:
        st.warning("Please enter at least 5 words.")
    else:
        proba = model.predict_proba(vectorizer.transform([clean_text(text)]))[0]
        label = int(np.argmax(proba))
        conf = proba[label] * 100
        if conf < 65:
            st.info(f"Not sure ({conf:.1f}%). Please verify with an official source.")
        elif label == 1:
            st.success(f"REAL NEWS - confidence {conf:.1f}%")
        else:
            st.error(f"FAKE NEWS - confidence {conf:.1f}%. Do not forward without verifying.")
