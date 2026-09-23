"""Create a small JSON request body for the scoring endpoint."""

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="sample-request.json")
    args = parser.parse_args()
    with Image.open(args.image) as image:
        image = image.convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
    body = {"image": base64.b64encode(buffer.getvalue()).decode("ascii")}
    Path(args.out).write_text(json.dumps(body), encoding="utf-8")
    print(f"Wrote {args.out} ({len(buffer.getvalue())} JPEG bytes)")


if __name__ == "__main__":
    main()