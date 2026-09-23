"""Streamlit frontend for the Azure ML animal endpoint."""

import base64
import io
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


st.set_page_config(page_title="Animal Predictor")
st.title("Animal Predictor")
endpoint_url = os.environ.get("ENDPOINT_URL")
endpoint_key = os.environ.get("ENDPOINT_KEY")
uploaded = st.file_uploader("Upload an animal photo", type=["jpg", "jpeg", "png"])


def annotate_prediction(image, result):
    """Print the prediction and confidence across the top of the image."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = f"{result['animal']} - confidence {float(result['confidence']) * 100:.0f}%"
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    padding = 10
    banner_height = bottom - top + padding * 2
    draw.rectangle((0, 0, image.width, banner_height), fill=(0, 0, 0))
    draw.text((padding, padding), text, fill=(255, 255, 255), font=font)
    return image

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded photo")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    if not endpoint_url or not endpoint_key:
        st.error("Set ENDPOINT_URL and ENDPOINT_KEY before making a prediction.")
    elif st.button("Predict"):
        payload = {"image": base64.b64encode(buffer.getvalue()).decode("ascii")}
        response = requests.post(endpoint_url, json=payload,
                                 headers={"Authorization": f"Bearer {endpoint_key}"}, timeout=60)
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            st.error(result["error"])
        else:
            confidence = float(result["confidence"])
            st.image(annotate_prediction(image, result), caption="Prediction output")
            st.subheader(f"{result['animal']} ({confidence:.1%})")
            st.bar_chart(pd.DataFrame.from_dict(result["all_scores"], orient="index", columns=["score"]))
            if confidence < 0.60:
                st.warning("The confidence is below 60%; treat this prediction cautiously.")