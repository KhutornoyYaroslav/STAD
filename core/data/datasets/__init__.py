from torch.utils.data import Dataset
from core.config import CfgNode
from core.data.transforms.transforms import BaseTransform
from core.data.datasets.single_image import SingleImageDataset


def build_dataset(cfg: CfgNode,
                  img_path: str,
                  label_path: str,
                  transforms: BaseTransform) -> Dataset:
    return SingleImageDataset(cfg, img_path, label_path, transforms)
