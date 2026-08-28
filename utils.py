from pathlib import Path

import cv2
import numpy as np


SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return image


def save_image(image: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def collect_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    files = []
    for fmt in SUPPORTED_FORMATS:
        files.extend(directory.glob(f"*{fmt}"))
        files.extend(directory.glob(f"*{fmt.upper()}"))
    return sorted(files)


def make_output_path(output_dir: str | Path, image_name: str, suffix: str) -> Path:
    return Path(output_dir) / f"{image_name}_{suffix}.png"
