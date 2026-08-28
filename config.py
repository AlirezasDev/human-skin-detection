import dataclasses


@dataclasses.dataclass(frozen=True)
class RGBThresholds:
    r_min: int = 95
    g_min: int = 40
    b_min: int = 20
    rg_diff_min: int = 15


@dataclasses.dataclass(frozen=True)
class HSVThresholds:
    h_min: float = 0.0
    h_max: float = 50.0
    s_min: float = 0.23
    s_max: float = 0.68


@dataclasses.dataclass(frozen=True)
class YCbCrThresholds:
    y_min: int = 80
    cb_min: int = 85
    cr_min: int = 135


@dataclasses.dataclass(frozen=True)
class YCbCrLinearConstraints:
    cr_upper_a: float = 1.5862
    cr_upper_b: float = 20.0
    cr_lower_a: float = 0.3448
    cr_lower_b: float = 76.2069
    cr_upper2_a: float = -4.5652
    cr_upper2_b: float = 234.5652
    cr_lower2_a: float = -1.15
    cr_lower2_b: float = 301.75
    cr_upper3_a: float = -2.2857
    cr_upper3_b: float = 432.85


@dataclasses.dataclass(frozen=True)
class MorphologyConfig:
    kernel_size: int = 7
    iterations: int = 2
    max_components: int = 2
    min_area: int = 1000


@dataclasses.dataclass(frozen=True)
class SkinDetectionConfig:
    rgb: RGBThresholds = dataclasses.field(default_factory=RGBThresholds)
    hsv: HSVThresholds = dataclasses.field(default_factory=HSVThresholds)
    ycbcr: YCbCrThresholds = dataclasses.field(default_factory=YCbCrThresholds)
    ycbcr_linear: YCbCrLinearConstraints = dataclasses.field(
        default_factory=YCbCrLinearConstraints
    )
    morphology: MorphologyConfig = dataclasses.field(default_factory=MorphologyConfig)


DEFAULT_CONFIG = SkinDetectionConfig()
