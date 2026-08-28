import cv2
import numpy as np

from config import SkinDetectionConfig, DEFAULT_CONFIG


class RGBDetector:
    def __init__(self, config: SkinDetectionConfig = DEFAULT_CONFIG):
        self.thresholds = config.rgb

    def detect(self, image: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(image)
        condition_r = r > self.thresholds.r_min
        condition_g = g > self.thresholds.g_min
        condition_b = b > self.thresholds.b_min
        condition_rg = r > g
        condition_rb = r > b
        condition_diff = np.abs(r.astype(np.int16) - g.astype(np.int16)) > self.thresholds.rg_diff_min
        mask = (
            condition_r
            & condition_g
            & condition_b
            & condition_rg
            & condition_rb
            & condition_diff
        )
        return mask.astype(np.uint8) * 255


class HSVDetector:
    def __init__(self, config: SkinDetectionConfig = DEFAULT_CONFIG):
        self.thresholds = config.hsv

    def detect(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        v = hsv[:, :, 2].astype(np.float32) / 255.0
        condition_h = (h >= self.thresholds.h_min) & (h <= self.thresholds.h_max)
        condition_s = (s >= self.thresholds.s_min) & (s <= self.thresholds.s_max)
        mask = condition_h & condition_s
        return mask.astype(np.uint8) * 255


class YCbCrDetector:
    def __init__(self, config: SkinDetectionConfig = DEFAULT_CONFIG):
        self.thresholds = config.ycbcr
        self.linear = config.ycbcr_linear

    def detect(self, image: np.ndarray) -> np.ndarray:
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y = ycbcr[:, :, 0].astype(np.float32)
        cr = ycbcr[:, :, 1].astype(np.float32)
        cb = ycbcr[:, :, 2].astype(np.float32)
        condition_y = y > self.thresholds.y_min
        condition_cb = cb > self.thresholds.cb_min
        condition_cr = cr > self.thresholds.cr_min
        condition_linear1 = cr <= (self.linear.cr_upper_a * cb) + self.linear.cr_upper_b
        condition_linear2 = cr >= (self.linear.cr_lower_a * cb) + self.linear.cr_lower_b
        condition_linear3 = cr >= (self.linear.cr_upper2_a * cb) + self.linear.cr_upper2_b
        condition_linear4 = cr <= (self.linear.cr_lower2_a * cb) + self.linear.cr_lower2_b
        condition_linear5 = cr <= (self.linear.cr_upper3_a * cb) + self.linear.cr_upper3_b
        mask = (
            condition_y
            & condition_cb
            & condition_cr
            & condition_linear1
            & condition_linear2
            & condition_linear3
            & condition_linear4
            & condition_linear5
        )
        return mask.astype(np.uint8) * 255


class CombinedDetector:
    def __init__(self, config: SkinDetectionConfig = DEFAULT_CONFIG):
        self.rgb_detector = RGBDetector(config)
        self.hsv_detector = HSVDetector(config)
        self.ycbcr_detector = YCbCrDetector(config)

    def detect_all(self, image: np.ndarray):
        mask_rgb = self.rgb_detector.detect(image)
        mask_hsv = self.hsv_detector.detect(image)
        mask_ycbcr = self.ycbcr_detector.detect(image)
        combined = cv2.bitwise_or(
            mask_rgb, cv2.bitwise_or(mask_hsv, mask_ycbcr)
        )
        return {
            "rgb": mask_rgb,
            "hsv": mask_hsv,
            "ycbcr": mask_ycbcr,
            "combined": combined,
        }
