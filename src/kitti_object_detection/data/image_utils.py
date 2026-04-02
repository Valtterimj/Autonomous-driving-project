from pathlib import Path
from PIL import Image

def get_image_size(image_path: Path) -> tuple[int, int]:
    """Return image size"""
    with Image.open(image_path) as img:
        return img.size