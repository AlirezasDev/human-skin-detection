# Human Skin Detection

A comprehensive skin detection system using multiple color spaces (RGB, HSV, YCbCr) to identify and extract human skin regions from images. Based on the research paper "Human Skin Detection Using RGB, HSV and YCbCr Color Models" by Kolkur et al.

## Features

- **Multi-color space analysis**: Combines RGB, HSV, and YCbCr for robust detection
- **Modular architecture**: Clean separation of config, detectors, processing, and utilities
- **Configurable thresholds**: Easy customization via frozen dataclasses
- **Batch processing**: Process entire directories of images
- **Multiple outputs**: Individual masks, combined mask, cleaned mask, and extracted skin

## Color Space Conditions

### RGB Color Space

| Channel | Condition |
|---------|-----------|
| R | > 95 |
| G | > 40 |
| B | > 20 |
| Constraint | R > G |
| Constraint | R > B |
| Constraint | \|R - G\| > 15 |

### HSV Color Space

| Channel | Condition |
|---------|-----------|
| H (Hue) | 0.0 to 50.0 |
| S (Saturation) | 0.23 to 0.68 (normalized to 0-1) |

**Note**: S and V channels are normalized before applying thresholds:

```
S = S / 255
V = V / 255
```

### YCbCr Color Space

| Channel | Condition |
|---------|-----------|
| Y (Luminance) | > 80 |
| Cb (Blue Chrominance) | > 85 |
| Cr (Red Chrominance) | > 135 |

**Linear constraints**:

- Cr ≤ 1.5862 × Cb + 20
- Cr ≥ 0.3448 × Cb + 76.2069
- Cr ≥ -4.5652 × Cb + 234.5652
- Cr ≤ -1.15 × Cb + 301.75
- Cr ≤ -2.2857 × Cb + 432.85

## Installation

```bash
git clone https://github.com/AlirezasDev/human-skin-detection.git
cd human-skin-detection
pip install -r requirements.txt
```

## Usage

### Single Image

```bash
python main.py photo.jpg
```

### Single Image with Custom Output

```bash
python main.py photo.jpg results/
```

### Batch Processing

```bash
python main.py --batch ./images/
```

### Batch Processing with Custom Output

```bash
python main.py --batch ./images/ results/
```

## Output Files

| File | Description |
|------|-------------|
| `*_mask_rgb.png` | Binary mask from RGB detection |
| `*_mask_hsv.png` | Binary mask from HSV detection |
| `*_mask_ycbcr.png` | Binary mask from YCbCr detection |
| `*_mask_combined.png` | Union of all three masks |
| `*_mask_clean.png` | Combined mask after morphological cleaning |
| `*_skin_extracted.png` | Original image with only skin regions |
| `*_skin_black_bg.png` | Skin regions on black background |

## Programmatic Usage

```python
from skin_detector import SkinDetector

detector = SkinDetector()
results = detector.process_image("photo.jpg", output_dir="output")

original = results["original"]
skin_mask = results["mask_clean"]
extracted = results["skin_extracted"]
```

### Custom Configuration

```python
from skin_detector import SkinDetector
from config import SkinDetectionConfig, RGBThresholds, HSVThresholds

custom_config = SkinDetectionConfig(
    rgb=RGBThresholds(r_min=100, g_min=50, b_min=30),
    hsv=HSVThresholds(h_min=0, h_max=45, s_min=0.2, s_max=0.7),
)
detector = SkinDetector(config=custom_config)
results = detector.process_image("photo.jpg")
```

## Project Structure

```
human-skin-detection/
├── config.py           # Threshold configuration dataclasses
├── detectors.py        # RGB, HSV, YCbCr detector implementations
├── processing.py       # Morphological operations and skin extraction
├── utils.py            # I/O utility functions
├── skin_detector.py    # Main SkinDetector class
├── main.py             # CLI entry point
├── example_usage.py    # Usage examples
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## Algorithm Overview

1. **Load Image**: Read input image in BGR format
2. **Color Space Conversion**: Convert to RGB, HSV, and YCbCr
3. **Apply Thresholds**: Check each pixel against paper conditions
4. **Combine Masks**: Union of all three color space masks
5. **Morphological Cleaning**: Close holes, remove noise, keep largest regions
6. **Extract Skin**: Apply cleaned mask to original image

## Research Basis

This implementation is based on the paper:

> Kolkur, S., Kalbande, D., Shimpi, P., Bapat, C., & Jatakia, J. (2017).
> Human Skin Detection Using RGB, HSV and YCbCr Color Models.
> *Advances in Intelligent Systems Research*, Vol. 137, pp. 324-332.

The algorithm achieves precision of 89.33% and accuracy of 94.43% on the Pratheepan dataset.

## Limitations

- Best results with frontal lighting conditions
- May detect non-skin regions with similar color profiles
- Performance varies with image quality and lighting

## License

MIT License
