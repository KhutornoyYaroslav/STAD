import logging
import numpy as np
from core.config import CfgNode
from core.data.transforms.transforms import (
    TransformInterface,
    Compose,
    CheckFormat,
    ConvertColor,
    Clip,
    ToType,
    Normalize,
    Denormalize,
    ToTensor,
    FromTensor,
    Resize,
    PadResize,
    RandomJpeg,
    RandomPerspective,
    RandomMirror,
    RandomContrast,
    RandomBrightness,
    RandomHue,
    RandomGray
)


def build_transforms(cfg: CfgNode, is_train: bool = True) -> TransformInterface:
    logger = logging.getLogger('CORE')

    transform = [
        CheckFormat(),
        ConvertColor("BGR", cfg.INPUT.COLOR)
    ]

    if is_train:
        transform += [
            RandomJpeg(0.3, 0.5),
            RandomMirror(x_axis_only=True, probability=0.5),
            RandomPerspective(translate=0.1, scale=2.0, rotate=5.0, probabilty=0.75, keep_aspect=False, downscale_only=True),
            PadResize(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.PAD_BORDER_VALUE, cfg.INPUT.MAKE_DIVISIBLE_BY),
            ToType(img_dtype=np.float32),
            RandomHue(delta=30.0, src_color=cfg.INPUT.COLOR, probability=0.25),
            RandomContrast(lower=0.75, upper=1.25, probability=0.25),
            RandomBrightness(delta=30, probability=0.25),
            RandomGray(0.25),
            Clip()
        ]
    else:
        transform += [
            PadResize(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.PAD_BORDER_VALUE, cfg.INPUT.MAKE_DIVISIBLE_BY),
            ToType(img_dtype=np.float32),
            Clip()
        ]

    transform += [
        Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_SCALE),
        ToTensor()
    ]

    # log transforms
    transforms_output_str = ""
    for trans in transform:
        transforms_output_str += f"'{trans.__class__.__name__}'\n"
        for field_name, field_value in vars(trans).items():
            transforms_output_str += f"\t{field_name}: {field_value}\n"
    type_str = "training" if is_train else "validation"
    logger.info(f"Transforms used for {type_str}:\n\n{transforms_output_str}")

    return Compose(transform)


def build_inv_transforms(cfg: CfgNode) -> TransformInterface:
    logger = logging.getLogger('CORE')

    transform = [
        FromTensor(),
        Denormalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_SCALE),
        ToType(img_dtype=np.uint8),
        ConvertColor(cfg.INPUT.COLOR, "BGR"),
        CheckFormat(),
    ]

    # log transforms
    transforms_output_str = ""
    for trans in transform:
        transforms_output_str += f"'{trans.__class__.__name__}'\n"
        for field_name, field_value in vars(trans).items():
            transforms_output_str += f"\t{field_name}: {field_value}\n"

    logger.info(f"Inverse transforms:\n\n{transforms_output_str}")

    return Compose(transform)
