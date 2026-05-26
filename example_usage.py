"""
Example usage of the SkinDetector class.
This script demonstrates different ways to use the skin detection system.
"""

from skin_detector import SkinDetector
import os


def example_single_image():
    """Example: Process a single image."""
    print("\n" + "="*60)
    print("Example 1: Single Image Processing")
    print("="*60)
    
    detector = SkinDetector()
    
    # You would replace this with an actual image path
    # detector.process_image('path/to/your/image.jpg', output_dir='output')
    
    print("To use with your own image:")
    print("  python skin_detector.py your_image.jpg")
    print("\nThis will generate several output files in the 'output/' directory")


def example_batch_processing():
    """Example: Process multiple images in a batch."""
    print("\n" + "="*60)
    print("Example 2: Batch Processing")
    print("="*60)
    
    detector = SkinDetector()
    
    # You would replace this with an actual directory containing images
    # detector.process_batch('path/to/images/', output_dir='output')
    
    print("To process all images in a directory:")
    print("  python skin_detector.py --batch /path/to/images/")
    print("\nSupported formats: jpg, jpeg, png, bmp, tiff")


def example_custom_thresholds():
    """Example: Customize detection thresholds."""
    print("\n" + "="*60)
    print("Example 3: Custom Thresholds")
    print("="*60)
    
    detector = SkinDetector()
    
    # Customize HSV thresholds
    detector.hsv_h_min = 0
    detector.hsv_h_max = 50
    detector.hsv_s_min = 0.1
    detector.hsv_s_max = 0.4
    detector.hsv_v_min = 0.15
    detector.hsv_v_max = 1.0
    
    print("You can customize color space thresholds:")
    print(f"  HSV - H: [{detector.hsv_h_min}-{detector.hsv_h_max}]")
    print(f"  HSV - S: [{detector.hsv_s_min}-{detector.hsv_s_max}] (normalized)")
    print(f"  HSV - V: [{detector.hsv_v_min}-{detector.hsv_v_max}] (normalized)")
    print(f"  RGB - R: [{detector.rgb_r_min}+, G: {detector.rgb_g_min}+, B: {detector.rgb_b_min}+]")
    print(f"  YCbCr - Cb: [{detector.ycbcr_cb_min}-{detector.ycbcr_cb_max}]")
    print(f"  YCbCr - Cr: [{detector.ycbcr_cr_min}-{detector.ycbcr_cr_max}]")


def example_mask_analysis():
    """Example: Access individual color space masks."""
    print("\n" + "="*60)
    print("Example 4: Individual Color Space Analysis")
    print("="*60)
    
    detector = SkinDetector()
    
    print("""
    The detector provides separate masks for each color space:
    
    - mask_hsv: Skin detection using HSV color space
    - mask_rgb: Skin detection using RGB color space  
    - mask_ycbcr: Skin detection using YCbCr color space
    - mask_combined: OR combination of all three masks
    - mask_clean: Combined mask with noise removal
    - skin_extracted: Original image with only skin regions
    
    You can access these through the results dictionary:
    
        results = detector.process_image('photo.jpg')
        hsv_mask = results['mask_hsv']
        rgb_mask = results['mask_rgb']
        ycbcr_mask = results['mask_ycbcr']
        final_skin = results['skin_extracted']
    """)


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Human Skin Detection - Usage Examples")
    print("="*60)
    
    example_single_image()
    example_batch_processing()
    example_custom_thresholds()
    example_mask_analysis()
    
    print("\n" + "="*60)
    print("For more information, see README.md")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
