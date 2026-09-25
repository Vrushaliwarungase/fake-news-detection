import streamlit as st
import joblib

# Load the trained model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page title
st.title("📰 Fake News Detection")
st.write("Enter a news article below to check its classification.")

# News input
news_text = st.text_area(
    "Enter news article:",
    height=250
)

# Prediction button
if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        # Convert the news text into TF-IDF numbers
        news_vector = vectorizer.transform([news_text])

        # Make prediction
        prediction = model.predict(news_vector)[0]

        # Display result
        if prediction == 0:
            st.error("⚠️ Classified as FAKE NEWS")
        else:
            st.success("✅ Classified as REAL NEWS")

        st.info(
            "Note: This result is based on the training dataset and "
            "should not be treated as a factual verification."
        )