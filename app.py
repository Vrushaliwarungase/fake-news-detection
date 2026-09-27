import streamlit as st
import joblib

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .real-news {
        border: 2px solid #2e7d32;
        background-color: #e8f5e9;
    }

    .fake-news {
        border: 2px solid #c62828;
        background-color: #ffebee;
    }

    .result-title {
        font-size: 30px;
        font-weight: bold;
    }

    .confidence {
        font-size: 17px;
        margin-top: 10px;
    }

    .disclaimer {
        font-size: 13px;
        margin-top: 30px;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.header("🔎 About the Project")

    st.write(
        "This application uses Machine Learning and "
        "Natural Language Processing (NLP) to classify "
        "news text as Real or Fake."
    )

    st.markdown("### 🛠️ Technologies")

    st.write("• Python")
    st.write("• Streamlit")
    st.write("• Scikit-learn")
    st.write("• TF-IDF")
    st.write("• Machine Learning")

    st.markdown("### 🤖 How It Works")

    st.write(
        "The entered news text is converted into TF-IDF "
        "features and passed to the trained machine "
        "learning model."
    )

# -----------------------------
# MAIN HEADER
# -----------------------------
st.markdown(
    '<div class="main-title">📰 Fake News Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered news classification using NLP and Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------
# NEWS INPUT
# -----------------------------
st.markdown("### 📝 Enter News Article")

news_text = st.text_area(
    "Paste or type the news article below:",
    height=250,
    placeholder="Example: Enter the complete news article here..."
)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🔍 Analyze News", use_container_width=True):

    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text before analyzing.")

    else:

        # Convert news text into TF-IDF numbers
        news_vector = vectorizer.transform([news_text])

        # Make prediction
        prediction = model.predict(news_vector)[0]

        # -----------------------------
        # FAKE NEWS RESULT
        # -----------------------------
        if prediction == 0:

            confidence = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(news_vector)[0]
                confidence = probabilities[0] * 100

            st.markdown(
                '<div class="result-box fake-news">'
                '<div class="result-title">⚠️ FAKE NEWS</div>'
                '<div class="confidence">'
                'The model classified this article as potentially fake.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            if confidence is not None:
                st.progress(int(confidence))
                st.write(
                    f"**Model confidence:** {confidence:.2f}%"
                )

        # -----------------------------
        # REAL NEWS RESULT
        # -----------------------------
        else:

            confidence = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(news_vector)[0]
                confidence = probabilities[1] * 100

            st.markdown(
                '<div class="result-box real-news">'
                '<div class="result-title">✅ REAL NEWS</div>'
                '<div class="confidence">'
                'The model classified this article as potentially real.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            if confidence is not None:
                st.progress(int(confidence))
                st.write(
                    f"**Model confidence:** {confidence:.2f}%"
                )

        # -----------------------------
        # DISCLAIMER
        # -----------------------------
        st.markdown(
            '<div class="disclaimer">'
            '⚠️ <b>Disclaimer:</b> This prediction is based on patterns '
            'learned from the training dataset. It does not independently '
            'verify the factual accuracy of the news.'
            '</div>',
            unsafe_allow_html=True
        )
