from pathlib import Path
from PIL import Image
import pytest

ROOT = Path(__file__).parents[1] / "data" / "animals"


def test_data_layout_and_readability():
    assert ROOT.is_dir()
    assert not any(path.is_file() for path in ROOT.iterdir())
    animals = sorted(path for path in ROOT.iterdir() if path.is_dir())
    assert len(animals) >= 2
    assert all(path.name == path.name.lower() and " " not in path.name for path in animals)
    for animal in animals:
        images = list(animal.glob("*.jpg"))
        assert len(images) >= 10, animal.name
        for image in images:
            try:
                with Image.open(image) as opened:
                    opened.verify()
            except Exception as error:
                pytest.fail(f"Unreadable image {image}: {error}")