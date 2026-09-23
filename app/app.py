"""Local and Azure ML browser interface for the animal classifier."""

import base64
import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image


# Allow this app to reuse the shared code in the sibling src folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_utils import annotate_prediction, load_model, predict_image  # noqa: E402


st.set_page_config(page_title="Animal Predictor", page_icon="🐾")
st.title("Animal Predictor")
st.write("Upload a photo to classify it with the local trained model.")

MODEL_DIR = PROJECT_ROOT / "models"
endpoint_url = os.environ.get("ENDPOINT_URL")
endpoint_key = os.environ.get("ENDPOINT_KEY")


@st.cache_resource
def get_local_model():
    """Load the local model once and reuse it between browser interactions."""
    return load_model(MODEL_DIR)


def call_azure_endpoint(image):
    """Send a compressed image to Azure when endpoint settings are provided."""
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    payload = {"image": base64.b64encode(buffer.getvalue()).decode("ascii")}
    response = requests.post(
        endpoint_url,
        json=payload,
        headers={"Authorization": f"Bearer {endpoint_key}"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


uploaded = st.file_uploader(
    "Choose an animal photo",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded:
    original_image = Image.open(uploaded).convert("RGB")
    st.image(original_image, caption="Uploaded photo")

    if st.button("Classify photo", type="primary"):
        try:
            if endpoint_url and endpoint_key:
                result = call_azure_endpoint(original_image.copy())
                mode = "Azure endpoint"
            else:
                model, classes = get_local_model()
                result = predict_image(model, classes, original_image)
                mode = "Local model"

            if "error" in result:
                st.error(result["error"])
            else:
                confidence = float(result["confidence"])
                annotated = annotate_prediction(original_image, result)
                st.image(annotated, caption=f"Prediction output ({mode})")
                st.success(f"Prediction: {result['animal']}")
                st.metric("Confidence", f"{confidence:.2%}")
                scores = pd.DataFrame.from_dict(
                    result["all_scores"], orient="index", columns=["score"]
                )
                st.subheader("All class scores")
                st.bar_chart(scores)
                if confidence < 0.60:
                    st.warning("Confidence is below 60%; treat this prediction cautiously.")
        except Exception as error:
            st.error(f"Could not classify the image: {error}")

st.sidebar.header("Settings")
if endpoint_url and endpoint_key:
    st.sidebar.info("Azure endpoint mode is active.")
else:
    st.sidebar.info("Local model mode is active. Set both ENDPOINT_URL and ENDPOINT_KEY for Azure mode.")