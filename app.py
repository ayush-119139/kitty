import re
import joblib
import numpy as np
import streamlit as st
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import base64

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_of_bin_file("assets/image_1.jpeg")


# 1. Page Configuration (Browser tab title, icon, and layout)
st.set_page_config(
    page_title="VeriNews AI | Fake News Detector",
    page_icon="📰",
    layout="centered"
)
#Background 
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #0e1117; 
        background-image: url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# 2. Load Model and Vectorizer (Cached for fast loading)
@st.cache_resource
def load_assets():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_assets()
stemmer = PorterStemmer()
STOP_WORDS = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = str(text)
    text = re.sub(r"^.{0,80}?\(Reuters\)\s*-\s*", "", text)
    text = re.sub(r"\(Reuters\)", " ", text, flags=re.I)
    text = re.sub(r"assets/image_1.jpeg", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [stemmer.stem(w) for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)

# 3. Sidebar with Branding, Sticker & Tips
with st.sidebar:
    st.image("assets/image_2.jpeg", width=110)
    st.title("VeriNews AI 🤖")
    st.markdown("Your smart assistant to spot misinformation and viral fake news.")
    st.divider()
    st.subheader("📌 How to get best results:")
    st.markdown("""
    * Paste the **full article** instead of just a short headline.
    * Avoid pasting random links without text.
    * Always cross-check uncertain results.
    """)
    st.info("⚡ Powered by Machine Learning & NLP")

# 4. Main Header & Banner
st.title("📰 VeriNews AI")
st.markdown("#### *Verify before you amplify! Check if a news story is legit or fake.* ✨")

# Optional Top Banner Image (You can replace this URL with your own banner image)
st.image(
    "assets/image_3.jpeg",
    use_container_width=True
)

st.write("")

# 5. Text Input Box
text = st.text_area(
    "✍️ Paste the news article or WhatsApp forward below:",
    height=200,
    placeholder="Paste full news text here to analyze..."
)

# 6. Action Button
verify = st.button("🔍 Verify Authenticity", use_container_width=True, type="primary")

# 7. Prediction & Visual Verdict with Stickers
if verify:
    if len(text.split()) < 5:
        st.warning("⚠️ Please enter at least 5 words so the AI can analyze the context properly.")
    else:
        with st.spinner("🕵️‍♂️ Analyzing writing style and patterns..."):
            proba = model.predict_proba(vectorizer.transform([clean_text(text)]))[0]
            label = int(np.argmax(proba))
            conf = float(proba[label] * 100)

        st.divider()
        st.subheader("📊 Analysis Verdict")

        # Create 2 columns: Left for Sticker, Right for Verdict & Progress Bar
        col_sticker, col_result = st.columns([1, 3])

        if conf < 65:
            with col_sticker:
                # Thinking Sticker
                st.image("assets/thinking.png", width=110)
            with col_result:
                st.info(f"🤔 **UNCERTAIN ({conf:.1f}% Confidence)**")
                st.write("The article shows mixed signals. Please verify this with a trusted news outlet.")
                st.progress(int(conf))

        elif label == 1:
            st.balloons()  # Celebration balloons animation!
            with col_sticker:
                # Verified Real Sticker
                st.image("assets/sticker_2.png", width=110)
            with col_result:
                st.success(f"✅ **REAL NEWS ({conf:.1f}% Confidence)**")
                st.write("Great! This text matches the writing patterns of credible journalism.")
                st.progress(int(conf))

        else:
            st.toast("🚨 Warning: High probability of fake news!", icon="🚨")
            with col_sticker:
                # Fake Alert Sticker
                st.image("assets/fake.png", width=110)
            with col_result:
                st.error(f"🚨 **FAKE NEWS DETECTED ({conf:.1f}% Confidence)**")
                st.write("Be careful! This shows strong signs of misinformation or clickbait. **Do not forward.**")
                st.progress(int(conf))
