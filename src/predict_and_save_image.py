"""Classify one image and save the result printed on the image.

Edit IMAGE_PATH below, then run this file with the project environment:

    .venv/Scripts/python.exe src/predict_and_save_image.py
"""

from pathlib import Path

from PIL import Image

from model_utils import annotate_prediction, load_model, predict_image


# Change this path to the image you want to classify.
IMAGE_PATH = Path("demo_images/cat_demo.jpg")

# The folder containing model.pt and classes.json.
MODEL_DIR = Path("models")

# The annotated image will be saved here.
OUTPUT_PATH = Path("outputs/prediction_with_confidence.jpg")


def main():
    """Load the model, classify the image, and save an annotated copy."""
    model, classes = load_model(MODEL_DIR)

    with Image.open(IMAGE_PATH) as image:
        result = predict_image(model, classes, image)
        annotated_image = annotate_prediction(image, result)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    annotated_image.save(OUTPUT_PATH, format="JPEG", quality=90)

    print(f"Animal: {result['animal']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Annotated image saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()