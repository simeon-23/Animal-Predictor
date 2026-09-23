"""Predict an animal in a local image."""

import argparse
from pathlib import Path

from PIL import Image

from model_utils import annotate_prediction, load_model, predict_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/prediction.jpg")
    args = parser.parse_args()
    model, classes = load_model(args.model_dir)
    with Image.open(args.image) as image:
        result = predict_image(model, classes, image)
        annotated = annotate_prediction(image, result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output, format="JPEG", quality=90)
    print(f"Prediction: {result['animal']}  (confidence {result['confidence'] * 100:.0f}%)")
    print(f"Output image: {output.resolve()}")


if __name__ == "__main__":
    main()