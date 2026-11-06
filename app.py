
import streamlit as st
import tensorflow as tf
import pickle
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# 1. PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Plagiarism Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Plagiarism Detection System")
st.write("Compare two texts using a Siamese LSTM model.")

# -----------------------------
# 2. LOAD MODEL AND TOKENIZER
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "plagiarism_model.keras"
    )

    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer

model, tokenizer = load_model()

MAX_LEN = 50

# -----------------------------
# 3. TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    return text

# -----------------------------
# 4. PREDICTION FUNCTION
# -----------------------------
def predict_pair(source_text, plag_text):

    source_text = clean_text(source_text)
    plag_text = clean_text(plag_text)

    src_seq = tokenizer.texts_to_sequences(
        [source_text]
    )

    plag_seq = tokenizer.texts_to_sequences(
        [plag_text]
    )

    src_pad = pad_sequences(
        src_seq, maxlen=MAX_LEN
    )

    plag_pad = pad_sequences(
        plag_seq, maxlen=MAX_LEN
    )

    prediction = model.predict(
        [src_pad, plag_pad],
        verbose=0
    )[0][0]

    return float(prediction)

# -----------------------------
# 5. FRONTEND INPUTS
# -----------------------------
st.subheader("Enter the texts")

source_text = st.text_area(
    "Source Text",
    placeholder="Enter the original text here...",
    height=150
)

plag_text = st.text_area(
    "Text to Check",
    placeholder="Enter the text you want to compare...",
    height=150
)

# -----------------------------
# 6. DETECTION BUTTON
# -----------------------------
if st.button("Detect Plagiarism", use_container_width=True):

    if not source_text.strip() or not plag_text.strip():
        st.warning("Please enter both texts.")

    else:
        with st.spinner("Analyzing texts..."):

            score = predict_pair(
                source_text,
                plag_text
            )

        st.divider()
        st.subheader("Detection Result")

        st.write(f"Prediction Score: {score:.4f}")

        if score > 0.5:
            st.error("⚠️ Likely Plagiarized")
        else:
            st.success("✅ Likely Not Plagiarized")

        st.progress(score)
