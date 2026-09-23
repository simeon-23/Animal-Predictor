"""Small data-only checks; deliberately does not import torch."""

from pathlib import Path

from PIL import Image


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "animals"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def test_data_folder_exists():
    assert DATA_DIR.is_dir()


def test_has_at_least_two_animals():
    assert len([path for path in DATA_DIR.iterdir() if path.is_dir()]) >= 2


def test_folder_names_are_lowercase_without_spaces():
    for path in DATA_DIR.iterdir():
        if path.is_dir():
            assert path.name == path.name.lower()
            assert " " not in path.name


def test_no_loose_images():
    assert not [path for path in DATA_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def test_each_animal_has_readable_images():
    folders = [path for path in DATA_DIR.iterdir() if path.is_dir()]
    for folder in folders:
        images = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        assert len(images) >= 10
        for image_path in images:
            with Image.open(image_path) as image:
                image.verify()