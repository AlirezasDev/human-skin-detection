"""
Human Skin Detection using Multiple Color Spaces
Implements skin detection conditions from research literature
"""

import cv2
import numpy as np
from pathlib import Path


class SkinDetector:
    """Detects human skin regions using multiple color space conditions."""
    
    def __init__(self):
        """Initialize skin detection thresholds."""
        # HSV thresholds (normalized S and V to 0-1 range)
        self.hsv_h_min = 0
        self.hsv_h_max = 50
        self.hsv_s_min = 0.08
        self.hsv_s_max = 0.68
        self.hsv_v_min = 0.20
        self.hsv_v_max = 1.0
        
        # RGB thresholds
        self.rgb_r_min = 95
        self.rgb_g_min = 40
        self.rgb_b_min = 20
        self.rgb_r_max = 255
        self.rgb_g_max = 255
        self.rgb_b_max = 255
        
        # YCbCr thresholds
        self.ycbcr_cb_min = 77
        self.ycbcr_cb_max = 127
        self.ycbcr_cr_min = 133
        self.ycbcr_cr_max = 173
    
    def detect_hsv(self, image):
        """
        Detect skin using HSV color space.
        Normalizes S and V channels to 0-1 range as specified.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Split channels
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32) / 255.0  # Normalize to 0-1
        v = hsv[:, :, 2].astype(np.float32) / 255.0  # Normalize to 0-1
        
        # Apply conditions
        # Hue: 0-50 degrees (0-180 scale in OpenCV, but we check raw values)
        condition_h = (h >= self.hsv_h_min) & (h <= self.hsv_h_max)
        
        # Saturation: 0.1-0.4 (after normalization)
        condition_s = (s >= self.hsv_s_min) & (s <= self.hsv_s_max)
        
        # Value: 0.15-1.0 (after normalization)
        condition_v = (v >= self.hsv_v_min) & (v <= self.hsv_v_max)
        
        # Combine all conditions
        mask = (condition_h & condition_s & condition_v).astype(np.uint8) * 255
        
        return mask
    
    def detect_rgb(self, image):
        """Detect skin using RGB color space conditions."""
        # Split channels
        b, g, r = cv2.split(image)
        
        # Apply conditions
        condition_r = (r >= self.rgb_r_min) & (r <= self.rgb_r_max)
        condition_g = (g >= self.rgb_g_min) & (g <= self.rgb_g_max)
        condition_b = (b >= self.rgb_b_min) & (b <= self.rgb_b_max)
        
        # Additional constraints
        condition_rg = (r > g)
        condition_rb = (r > b)
        condition_max_min = (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15
        condition_rg_diff = np.abs(r - g) > 15
        
        # Combine all conditions
        mask = (
            condition_r
            & condition_g
            & condition_b
            & condition_rg
            & condition_rb
            & condition_max_min
            & condition_rg_diff
        ).astype(np.uint8) * 255
        
        return mask
    
    def detect_normalized_rgb(self, image):
        """Detect skin using normalized RGB ratios."""
        b, g, r = cv2.split(image.astype(np.float32))
        total = r + g + b + 1e-6
        r_norm = r / total
        g_norm = g / total
        b_norm = b / total

        condition_rn = (r_norm >= 0.35) & (r_norm <= 0.53)
        condition_gn = (g_norm >= 0.28) & (g_norm <= 0.46)
        condition_bn = (b_norm >= 0.15) & (b_norm <= 0.30)
        condition_ratio = (r_norm > g_norm) & (r_norm > b_norm)

        mask = (condition_rn & condition_gn & condition_bn & condition_ratio).astype(np.uint8) * 255
        return mask
    
    def detect_ycbcr(self, image):
        """Detect skin using YCbCr color space conditions."""
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Split channels
        y, cr, cb = cv2.split(ycbcr)
        
        # Apply conditions
        condition_cb = (cb >= self.ycbcr_cb_min) & (cb <= self.ycbcr_cb_max)
        condition_cr = (cr >= self.ycbcr_cr_min) & (cr <= self.ycbcr_cr_max)
        
        # Combine conditions
        mask = (condition_cb & condition_cr).astype(np.uint8) * 255
        
        return mask
    
    def detect_lab(self, image):
        """Detect skin using Lab color space conditions."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)

        condition_l = (l >= 20) & (l <= 255)
        condition_a = (a >= 133) & (a <= 168)
        condition_b = (b >= 143) & (b <= 208)
        
        mask = (condition_l & condition_a & condition_b).astype(np.uint8) * 255
        return mask

    def detect_face_region(self, image):
        """Detect the face region to restrict skin detection to the face area."""
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(80, 80))

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for (x, y, w, h) in faces:
            pad_w = int(w * 0.08)
            pad_h = int(h * 0.18)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(image.shape[1], x + w + pad_w)
            y2 = min(image.shape[0], y + h + pad_h)
            mask[y1:y2, x1:x2] = 255

        return mask
    
    def keep_largest_components(self, mask, max_components=2, min_area=1000):
        """Keep the largest connected skin regions and discard small noise."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask

        areas = stats[1:, cv2.CC_STAT_AREA]
        sorted_indices = np.argsort(areas)[::-1]
        selected = sorted_indices[:max_components]

        kept_mask = np.zeros_like(mask)
        for idx in selected:
            area = areas[idx]
            if area >= min_area:
                kept_mask[labels == idx + 1] = 255

        return kept_mask

    def detect_combined(self, image):
        """
        Detect skin using combined conditions from multiple color spaces.
        Uses a majority vote and face-region masking to reduce false positives.
        """
        mask_hsv = self.detect_hsv(image)
        mask_rgb = self.detect_rgb(image)
        mask_ycbcr = self.detect_ycbcr(image)
        mask_lab = self.detect_lab(image)
        mask_rgb_norm = self.detect_normalized_rgb(image)

        # Broad union mask for diagnostics and fallback
        combined_mask = cv2.bitwise_or(
            mask_ycbcr,
            cv2.bitwise_or(
                mask_rgb,
                cv2.bitwise_or(mask_hsv, cv2.bitwise_or(mask_lab, mask_rgb_norm)),
            ),
        )

        # Majority voting across multiple color spaces
        votes = (
            (mask_hsv > 0).astype(np.uint8)
            + (mask_rgb > 0).astype(np.uint8)
            + (mask_ycbcr > 0).astype(np.uint8)
            + (mask_lab > 0).astype(np.uint8)
            + (mask_rgb_norm > 0).astype(np.uint8)
        )
        refined_mask = ((votes >= 2).astype(np.uint8) * 255)

        # Restrict to face region when detected
        face_region = self.detect_face_region(image)
        if face_region.sum() > 0:
            refined_mask = cv2.bitwise_and(refined_mask, face_region)

        return combined_mask, refined_mask, mask_hsv, mask_rgb, mask_ycbcr
    
    def postprocess_mask(self, mask, kernel_size=7, iterations=2):
        """
        Apply morphological operations to clean up the mask.
        Removes noise and fills small holes, then keeps the largest connected regions.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Closing: fill small holes
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        # Opening: remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
        # Keep only the largest regions to remove stray false positives
        filtered = self.keep_largest_components(opened, max_components=2, min_area=800)
        
        return filtered
    
    def extract_skin(self, image, mask):
        """
        Extract skin regions from image using the mask.
        Returns the image with only skin regions visible.
        """
        result = cv2.bitwise_and(image, image, mask=mask)
        return result
    
    def process_image(self, image_path, output_dir="output", use_postprocessing=True):
        """
        Process an image for skin detection.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save output images
            use_postprocessing: Whether to apply morphological operations
        
        Returns:
            Dictionary with all results
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get image name for output files
        image_name = Path(image_path).stem
        
        # Detect skin using combined method
        combined_mask, refined_mask, mask_hsv, mask_rgb, mask_ycbcr = self.detect_combined(image)
        
        # Apply postprocessing if requested
        if use_postprocessing:
            refined_mask_clean = self.postprocess_mask(refined_mask)
        else:
            refined_mask_clean = refined_mask
        
        # Extract skin regions
        skin_extracted = self.extract_skin(image, refined_mask_clean)
        
        # Save outputs
        cv2.imwrite(str(output_path / f"{image_name}_mask_hsv.png"), mask_hsv)
        cv2.imwrite(str(output_path / f"{image_name}_mask_rgb.png"), mask_rgb)
        cv2.imwrite(str(output_path / f"{image_name}_mask_ycbcr.png"), mask_ycbcr)
        cv2.imwrite(str(output_path / f"{image_name}_mask_combined.png"), combined_mask)
        cv2.imwrite(str(output_path / f"{image_name}_mask_refined.png"), refined_mask)
        cv2.imwrite(str(output_path / f"{image_name}_mask_clean.png"), refined_mask_clean)
        cv2.imwrite(str(output_path / f"{image_name}_skin_extracted.png"), skin_extracted)
        
        print(f"\n✓ Processed: {image_path}")
        print(f"  - HSV mask saved: {image_name}_mask_hsv.png")
        print(f"  - RGB mask saved: {image_name}_mask_rgb.png")
        print(f"  - YCbCr mask saved: {image_name}_mask_ycbcr.png")
        print(f"  - Combined mask saved: {image_name}_mask_combined.png")
        print(f"  - Clean mask saved: {image_name}_mask_clean.png")
        print(f"  - Extracted skin saved: {image_name}_skin_extracted.png")
        
        return {
            'original': image,
            'mask_hsv': mask_hsv,
            'mask_rgb': mask_rgb,
            'mask_ycbcr': mask_ycbcr,
            'mask_combined': combined_mask,
            'mask_refined': refined_mask,
            'mask_clean': refined_mask_clean,
            'skin_extracted': skin_extracted
        }
    
    def process_batch(self, image_dir, output_dir="output", use_postprocessing=True):
        """
        Process all images in a directory.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save output images
            use_postprocessing: Whether to apply morphological operations
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Supported image formats
        image_formats = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
        image_files = []
        
        for fmt in image_formats:
            image_files.extend(image_dir.glob(fmt))
            image_files.extend(image_dir.glob(fmt.upper()))
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        for image_path in image_files:
            try:
                self.process_image(image_path, output_dir, use_postprocessing)
            except Exception as e:
                print(f"✗ Error processing {image_path}: {e}")


def main():
    """Main entry point for the skin detection program."""
    import sys
    
    # Create detector
    detector = SkinDetector()
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Skin Detection Tool")
        print("=" * 50)
        print("\nUsage:")
        print("  python skin_detector.py <image_path>")
        print("  python skin_detector.py --batch <image_directory>")
        print("\nExamples:")
        print("  python skin_detector.py photo.jpg")
        print("  python skin_detector.py --batch ./images")
        print("\nOutputs:")
        print("  - HSV mask: shows skin regions detected by HSV color space")
        print("  - RGB mask: shows skin regions detected by RGB color space")
        print("  - YCbCr mask: shows skin regions detected by YCbCr color space")
        print("  - Combined mask: OR combination of all three methods")
        print("  - Clean mask: combined mask with morphological cleaning")
        print("  - Extracted skin: original image with only skin regions visible")
        print("\nColor Space Details:")
        print("  HSV: H=[0-50], S=[0.1-0.4], V=[0.15-1.0] (S,V normalized to 0-1)")
        print("  RGB: R>95, G>40, B>20, R>G, R>B")
        print("  YCbCr: Cb=[77-127], Cr=[133-173]")
        return
    
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Error: please provide image directory")
            return
        detector.process_batch(sys.argv[2])
    else:
        # Single image
        detector.process_image(sys.argv[1])


if __name__ == "__main__":
    main()
