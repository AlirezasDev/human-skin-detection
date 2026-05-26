# Human Skin Detection

A comprehensive skin detection system using multiple color spaces (HSV, RGB, YCbCr) to identify and extract human skin regions from images.

## Features

- **Multi-color space analysis**: Combines HSV, RGB, and YCbCr color spaces for robust skin detection
- **Morphological cleaning**: Applies erosion and dilation to remove noise and fill holes
- **Batch processing**: Process multiple images at once
- **Individual masks**: See results from each color space separately
- **Normalized HSV**: Properly normalizes S and V channels to 0-1 range as specified in research literature

## Color Space Conditions

### HSV Color Space
- **Hue (H)**: 0 to 50 degrees
- **Saturation (S)**: 0.1 to 0.4 (normalized to 0-1 range)
- **Value (V)**: 0.15 to 1.0 (normalized to 0-1 range)

**Normalization**:
```
S = S / 255
V = V / 255
```

### RGB Color Space
- **Red (R)**: > 95
- **Green (G)**: > 40
- **Blue (B)**: > 20
- **Constraints**: R > G and R > B

### YCbCr Color Space
- **Cb**: 77 to 127
- **Cr**: 133 to 173

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Single Image Processing

```bash
python skin_detector.py path/to/image.jpg
```

This generates the following outputs in the `output/` directory:
- `image_mask_hsv.png` - Mask from HSV detection
- `image_mask_rgb.png` - Mask from RGB detection
- `image_mask_ycbcr.png` - Mask from YCbCr detection
- `image_mask_combined.png` - Combined mask (OR of all three)
- `image_mask_clean.png` - Combined mask with morphological cleaning
- `image_skin_extracted.png` - Original image with only skin regions visible

### Batch Processing

```bash
python skin_detector.py --batch path/to/images/
```

Processes all images in the directory and saves results to `output/`.

## Output Descriptions

| Output | Description |
|--------|-------------|
| `mask_hsv.png` | Binary mask from HSV color space analysis |
| `mask_rgb.png` | Binary mask from RGB color space analysis |
| `mask_ycbcr.png` | Binary mask from YCbCr color space analysis |
| `mask_combined.png` | Union (OR) of all three masks before cleaning |
| `mask_clean.png` | Combined mask after morphological operations |
| `skin_extracted.png` | Original image with only detected skin regions |

## Algorithm Overview

1. **Color Space Conversion**: Convert input image to HSV, RGB, and YCbCr
2. **Channel Normalization**: Normalize HSV S and V channels to 0-1 range
3. **Condition Checking**: Apply threshold conditions for each color space
4. **Mask Combination**: Combine masks using OR operation
5. **Morphological Cleaning**: Apply close and open operations to remove noise
6. **Skin Extraction**: Apply mask to original image to extract skin regions

## Example

```python
from skin_detector import SkinDetector

# Create detector instance
detector = SkinDetector()

# Process a single image
results = detector.process_image('photo.jpg', output_dir='output')

# Access results
original = results['original']
skin_mask = results['mask_clean']
extracted = results['skin_extracted']
```

## Dependencies

- **OpenCV**: Image processing
- **NumPy**: Numerical operations
- **Matplotlib**: Visualization
- **SciPy**: Scientific computing

## Research Basis

This implementation is based on established skin detection techniques that leverage multiple color spaces:
- HSV-based conditions for hue and saturation filtering
- RGB-based constraints for typical skin color ranges
- YCbCr-based chrominance analysis

## Limitations

- Best results with frontal lighting
- May detect non-skin regions with similar color profiles
- Performance varies with image quality and lighting conditions

## Future Improvements

- Adaptive thresholds based on image lighting
- Machine learning-based refinement
- Real-time video processing
- GPU acceleration

## License

MIT License
