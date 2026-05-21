import logging
from core.config import CfgNode
from core.data.transforms.transforms import (
    Clip,
    Resize,
    ToFloat,
    Normalize,
    ToTensor,
    CheckFormat,
    ConvertColor,
    MakeDivisibleBy,
    Compose,
    RandomJpeg,
    RandomPerspective,
    PadResize,
    RandomCrop,
    RandomMirror,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    RandomHue,
    RandomGray
)


def build_transforms(cfg: CfgNode, is_train: bool = True):
    logger = logging.getLogger('CORE')

    transform = [
        CheckFormat(),
        ConvertColor("BGR", "RGB")
    ]

    if is_train:
        transform += [
            RandomJpeg(0.3, 0.5),
            RandomMirror(0.5),
            RandomPerspective(translate=0.1, scale=1.5, rotate=5.0, probabilty=0.75, keep_aspect=False),
            PadResize(cfg.INPUT.IMAGE_SIZE, make_divisible_by=cfg.INPUT.MAKE_DIVISIBLE_BY),
            ToFloat(),
            RandomHue(delta=60.0, src_color='RGB',probability=0.25),
            RandomGamma(lower=0.75, upper=1.25, probability=0.25),
            RandomBrightness(delta=30, probability=0.5),
            RandomGray(0.25),
            Clip()
        ]
    else:
        transform += [
            PadResize(cfg.INPUT.IMAGE_SIZE, make_divisible_by=cfg.INPUT.MAKE_DIVISIBLE_BY),
            ToFloat(),
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
