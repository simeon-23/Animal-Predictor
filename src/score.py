"""Azure ML online endpoint scoring script."""

import base64
import io
import json
import os

from PIL import Image

from model_utils import load_model, predict_image

model = None
classes = None


def init():
    global model, classes
    model, classes = load_model(os.environ["AZUREML_MODEL_DIR"])


def run(raw_data):
    try:
        body = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        encoded = body["image"]
        image = Image.open(io.BytesIO(base64.b64decode(encoded)))
        result = predict_image(model, classes, image)
        print(f"Prediction: {result['animal']} confidence={result['confidence']:.4f}")
        return result
    except Exception as error:
        message = str(error)
        print(f"Prediction error: {message}")
        return {"error": message}