from skin_detector import SkinDetector
from config import SkinDetectionConfig, RGBThresholds, HSVThresholds, YCbCrThresholds


def example_single_image():
    print("\n" + "=" * 60)
    print("Example 1: Single Image Processing")
    print("=" * 60)
    detector = SkinDetector()
    print("To use with your own image:")
    print("  python main.py your_image.jpg")
    print("\nOutputs saved to 'output/' directory")


def example_batch_processing():
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)
    detector = SkinDetector()
    print("To process all images in a directory:")
    print("  python main.py --batch /path/to/images/")
    print("\nSupported formats: jpg, jpeg, png, bmp, tiff")


def example_custom_thresholds():
    print("\n" + "=" * 60)
    print("Example 3: Custom Thresholds")
    print("=" * 60)
    custom_config = SkinDetectionConfig(
        rgb=RGBThresholds(r_min=100, g_min=50, b_min=30, rg_diff_min=20),
        hsv=HSVThresholds(h_min=0, h_max=45, s_min=0.2, s_max=0.7),
        ycbcr=YCbCrThresholds(y_min=85, cb_min=90, cr_min=140),
    )
    detector = SkinDetector(config=custom_config)
    print("Custom thresholds configured:")
    print(f"  RGB - R>{custom_config.rgb.r_min}, G>{custom_config.rgb.g_min}, B>{custom_config.rgb.b_min}")
    print(f"  HSV - H=[{custom_config.hsv.h_min}-{custom_config.hsv.h_max}], S=[{custom_config.hsv.s_min}-{custom_config.hsv.s_max}]")
    print(f"  YCbCr - Y>{custom_config.ycbcr.y_min}, Cb>{custom_config.ycbcr.cb_min}, Cr>{custom_config.ycbcr.cr_min}")


def example_individual_masks():
    print("\n" + "=" * 60)
    print("Example 4: Individual Color Space Masks")
    print("=" * 60)
    detector = SkinDetector()
    print("Available mask keys in results dictionary:")
    print("  'mask_rgb'     - RGB color space detection")
    print("  'mask_hsv'     - HSV color space detection")
    print("  'mask_ycbcr'   - YCbCr color space detection")
    print("  'mask_combined' - Union of all three masks")
    print("  'mask_clean'   - Combined mask after morphological cleaning")
    print("  'skin_extracted' - Original image with only skin regions")
    print("  'skin_black_bg'  - Skin regions on black background")


def main():
    print("\n" + "=" * 60)
    print("Human Skin Detection - Usage Examples")
    print("=" * 60)
    example_single_image()
    example_batch_processing()
    example_custom_thresholds()
    example_individual_masks()
    print("\n" + "=" * 60)
    print("For more information, see README.md")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
