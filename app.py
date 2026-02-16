import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page config
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

# Title section
st.markdown("<h1 style='text-align: center;'>🧠 Next Word Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: grey;'>LSTM-based Language Model Demo</p>",
    unsafe_allow_html=True
)

# Load model and tokenizer safely
@st.cache_resource
def load_assets():
    model = load_model("next_word_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

with st.spinner("Loading model, please wait..."):
    model, tokenizer = load_assets()

SEQ_LEN = 30

# Prediction function
def predict_next_word(model, tokenizer, text, temperature=1.0, top_k=10):
    tokens = tokenizer.texts_to_sequences([text])[0]
    tokens = tokens[-SEQ_LEN:]
    padded = pad_sequences([tokens], maxlen=SEQ_LEN, padding='pre')

    preds = model.predict(padded, verbose=0)[0]
    preds = np.log(preds + 1e-9) / temperature
    probs = np.exp(preds) / np.sum(np.exp(preds))

    top_indices = np.argsort(probs)[-top_k:]
    top_probs = probs[top_indices]
    top_probs /= np.sum(top_probs)

    return tokenizer.index_word[np.random.choice(top_indices, p=top_probs)]

# UI Card
st.markdown("---")
st.markdown("### ✍️ Enter Text")
user_input = st.text_input(
    "",
    placeholder="Example: once upon a time",
)

st.markdown("### 🎛️ Creativity Control (Temperature)")
temperature = st.slider(
    "",
    min_value=0.5,
    max_value=1.5,
    value=0.8,
    step=0.1
)

# Explanation note (THIS IS WHAT YOU ASKED FOR)
st.info(
    "📌 **What is Temperature?**\n\n"
    "- **Low temperature (0.5 – 0.7):** More safe and predictable words\n"
    "- **Medium temperature (0.8 – 1.0):** Balanced and realistic predictions\n"
    "- **High temperature (1.1 – 1.5):** More creative but less predictable output\n\n"
    "Temperature controls how *creative or conservative* the model’s predictions are."
)

# Predict button
st.markdown("---")
if st.button("🔮 Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = predict_next_word(model, tokenizer, user_input, temperature)
        st.success(f"**Predicted Next Word:** `{next_word}`")

# Footer
st.markdown(
    "<hr><p style='text-align:center; color:grey;'>Built using LSTM & Streamlit</p>",
    unsafe_allow_html=True
)
