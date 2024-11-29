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
    PadResize
)


def build_transforms(cfg: CfgNode, is_train: bool = True):
    transform = [
        CheckFormat(),
        ConvertColor("BGR", "RGB")
    ]

    if is_train:
        transform += [
            RandomJpeg(0.5, 0.5),
            RandomPerspective(rotate=15.0, translate=0.25, scale=0.25, perspective=0.0),
            Resize(cfg.INPUT.IMAGE_SIZE),
            # PadResize(cfg.INPUT.IMAGE_SIZE),
            ToFloat(),
            Clip()
        ]
    else:
        transform += [
            Resize(cfg.INPUT.IMAGE_SIZE),
            # PadResize(cfg.INPUT.IMAGE_SIZE),
            ToFloat(),
            Clip()
        ]

    transform += [
        Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_SCALE),
        ToTensor()
    ]

    return Compose(transform)
