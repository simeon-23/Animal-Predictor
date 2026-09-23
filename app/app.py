"""Streamlit frontend for the Azure ML animal endpoint."""

import base64
import io
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Animal Predictor")
st.title("Animal Predictor")
endpoint_url = os.environ.get("ENDPOINT_URL")
endpoint_key = os.environ.get("ENDPOINT_KEY")
uploaded = st.file_uploader("Upload an animal photo", type=["jpg", "jpeg", "png"])

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
            st.subheader(f"{result['animal']} ({confidence:.1%})")
            st.bar_chart(pd.DataFrame.from_dict(result["all_scores"], orient="index", columns=["score"]))
            if confidence < 0.60:
                st.warning("The confidence is below 60%; treat this prediction cautiously.")