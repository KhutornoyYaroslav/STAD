from torch.utils.data import Dataset
from core.config import CfgNode
from core.data.transforms.transforms import BaseTransform
from core.data.datasets.single_image import SingleImageDataset
from core.data.datasets.imagenet_vid import ImagenetVidDataset
from core.data.datasets.crowdhuman import CrowdHumanDataset
from core.data.datasets.imagenet_vidvrd import ImagenetVidVrdDataset


def build_dataset(cfg: CfgNode,
                  data_path: str,
                  anno_path: str,
                  transforms: BaseTransform) -> Dataset:
    dstype = cfg.DATASET.TYPE
    if dstype == "SingleImageDataset":
        return SingleImageDataset(cfg, data_path, anno_path, transforms)
    elif dstype == "ImagenetVidDataset":
        return ImagenetVidDataset(cfg, data_path, anno_path, transforms)
    elif dstype == "CrowdHumanDataset":
        return CrowdHumanDataset(cfg, data_path, anno_path, transforms)
    elif dstype == "ImagenetVidVrdDataset":
        return ImagenetVidVrdDataset(cfg, data_path, anno_path, transforms)
    else:
        raise ValueError(f"Can't find dataset type '{dstype}'")
