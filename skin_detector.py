from pathlib import Path

import numpy as np

from config import SkinDetectionConfig, DEFAULT_CONFIG
from detectors import CombinedDetector
from processing import postprocess_mask, extract_skin, apply_black_background
from utils import load_image, save_image, collect_images, make_output_path


class SkinDetector:
    def __init__(self, config: SkinDetectionConfig = DEFAULT_CONFIG):
        self.config = config
        self.detector = CombinedDetector(config)

    def process_image(
        self,
        image_path: str | Path,
        output_dir: str | Path = "output",
    ) -> dict[str, np.ndarray]:
        image = load_image(image_path)
        image_name = Path(image_path).stem
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        masks = self.detector.detect_all(image)
        clean_mask = postprocess_mask(masks["combined"], self.config.morphology)
        skin_extracted = extract_skin(image, clean_mask)
        black_bg = apply_black_background(image, clean_mask)

        save_image(masks["rgb"], make_output_path(output_dir, image_name, "mask_rgb"))
        save_image(masks["hsv"], make_output_path(output_dir, image_name, "mask_hsv"))
        save_image(masks["ycbcr"], make_output_path(output_dir, image_name, "mask_ycbcr"))
        save_image(masks["combined"], make_output_path(output_dir, image_name, "mask_combined"))
        save_image(clean_mask, make_output_path(output_dir, image_name, "mask_clean"))
        save_image(skin_extracted, make_output_path(output_dir, image_name, "skin_extracted"))
        save_image(black_bg, make_output_path(output_dir, image_name, "skin_black_bg"))

        return {
            "original": image,
            "mask_rgb": masks["rgb"],
            "mask_hsv": masks["hsv"],
            "mask_ycbcr": masks["ycbcr"],
            "mask_combined": masks["combined"],
            "mask_clean": clean_mask,
            "skin_extracted": skin_extracted,
            "skin_black_bg": black_bg,
        }

    def process_batch(
        self,
        image_dir: str | Path,
        output_dir: str | Path = "output",
    ) -> None:
        images = collect_images(image_dir)
        if not images:
            print(f"No images found in {image_dir}")
            return
        print(f"Found {len(images)} images to process")
        for image_path in images:
            try:
                self.process_image(image_path, output_dir)
                print(f"Processed: {image_path.name}")
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
