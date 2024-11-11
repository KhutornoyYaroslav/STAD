from core.config import CfgNode
from core.data.transforms.transforms import (
    Clip,
    Resize,
    ToFloat,
    Normalize,
    ToTensor,
    CheckData,
    ConvertColor,
    MakeDivisibleBy,
    TransformCompose
)


def build_transforms(cfg: CfgNode, is_train: bool = True):
    transform = [CheckData(), ConvertColor("BGR", "RGB")]

    if is_train:
        transform += [
            Resize(cfg.INPUT.IMAGE_SIZE),
            ToFloat(),
            Clip()
        ]
    else:
        transform += [
            Resize(cfg.INPUT.IMAGE_SIZE),
            ToFloat(),
            Clip()
        ]

    transform += [Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_SCALE), ToTensor()]

    return TransformCompose(transform)
