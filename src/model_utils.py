"""Shared model, image, persistence, and optional MLflow helpers."""

import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms


def get_transform():
    """Return the one transform used by training and prediction."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_model(num_classes, pretrained=True, freeze=True):
    """Build MobileNetV2 with a classifier sized for discovered classes."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    if freeze:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    return model


def save_model(model, classes, model_dir):
    """Save only the state dict and class list in a model folder."""
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    (output / "classes.json").write_text(
        json.dumps(list(classes), indent=2), encoding="utf-8"
    )


def load_model(model_dir):
    """Find model files recursively, rebuild the model, and load its weights."""
    root = Path(model_dir)
    model_file = next(root.rglob("model.pt"), None)
    classes_file = next(root.rglob("classes.json"), None)
    if model_file is None or classes_file is None:
        raise FileNotFoundError(f"Could not find model.pt and classes.json under {root}")
    classes = json.loads(classes_file.read_text(encoding="utf-8"))
    model = build_model(len(classes), pretrained=False, freeze=True)
    state = torch.load(model_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.features.eval()
    return model, classes


def predict_image(model, classes, pil_image):
    """Predict one PIL image and return JSON-friendly scores."""
    image = pil_image.convert("RGB")
    tensor = get_transform()(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    scores = {name: float(probabilities[index]) for index, name in enumerate(classes)}
    best_index = int(torch.argmax(probabilities))
    return {
        "animal": classes[best_index],
        "confidence": float(probabilities[best_index]),
        "all_scores": scores,
    }


def annotate_prediction(pil_image, result):
    """Return a copy of the image with the animal and confidence printed on it."""
    image = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = f"{result['animal']} - confidence {result['confidence'] * 100:.0f}%"
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    padding = 10
    banner_height = bottom - top + padding * 2
    draw.rectangle((0, 0, image.width, banner_height), fill=(0, 0, 0))
    draw.text((padding, padding), text, fill=(255, 255, 255), font=font)
    return image


def log_metric(name, value, step=None):
    """Log to MLflow only for Azure ML tracking URIs."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("azureml"):
        return
    try:
        import mlflow
        mlflow.log_metric(name, float(value), step=step)
    except ImportError:
        pass