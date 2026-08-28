import sys
from skin_detector import SkinDetector


def print_usage():
    print("Human Skin Detection Tool")
    print("=" * 50)
    print("\nUsage:")
    print("  python main.py <image_path> [output_dir]")
    print("  python main.py --batch <image_directory> [output_dir]")
    print("\nExamples:")
    print("  python main.py photo.jpg")
    print("  python main.py photo.jpg results/")
    print("  python main.py --batch ./images")
    print("  python main.py --batch ./images results/")
    print("\nColor Space Conditions:")
    print("  HSV:    H=[0-50], S=[0.23-0.68] (S normalized to 0-1)")
    print("  RGB:    R>95, G>40, B>20, R>G, R>B, |R-G|>15")
    print("  YCbCr:  Y>80, Cb>85, Cr>135 + linear constraints")


def main():
    if len(sys.argv) < 2:
        print_usage()
        return

    detector = SkinDetector()

    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Error: please provide image directory")
            return
        image_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "output"
        detector.process_batch(image_dir, output_dir)
    else:
        image_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
        detector.process_image(image_path, output_dir)
        print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
