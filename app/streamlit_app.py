"""Streamlit UI: upload an image, see the generated caption.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work
# regardless of the directory Streamlit was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image

from src.inference.predict import Predictor

CHECKPOINT_PATH = Path("models/resnet_attention_lstm/best_model.pt")
VOCAB_PATH = Path("data/processed/vocab.json")


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor(checkpoint_path=CHECKPOINT_PATH, vocab_path=VOCAB_PATH, device="cpu")


st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️")
st.title("🖼️ Image Caption Generator")
st.write("Upload an image and get an automatically generated caption (ResNet50 + LSTM baseline).")

if not CHECKPOINT_PATH.exists():
    st.error(f"No trained model found at `{CHECKPOINT_PATH}`. Train a model first.")
    st.stop()

predictor = load_predictor()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Generating caption..."):
        caption = predictor.predict(image)

    st.subheader("Generated Caption")
    st.success(caption)