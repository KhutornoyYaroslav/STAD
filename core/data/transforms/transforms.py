import torch
import cv2 as cv
import numpy as np
from typing import Dict, List, Tuple
from numpy.typing import ArrayLike
from core.data.transforms.functional import make_divisible_by


class BaseTransform:
    def __init__(self) -> None:
        pass

    def __call__(self, data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        pass


class TransformCompose(BaseTransform):
    def __init__(self, transforms: List[BaseTransform]) -> None:
        self.transforms = transforms

    def __call__(self, data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        for t in self.transforms:
            data = t(data)
        return data


class CheckData(BaseTransform):
    def __call__(self, data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        if "img" in data:
            data["img"] = np.asarray(data["img"])
            if not data["img"].ndim == 4:
                raise ValueError("Expected 4D array in data['img']")
        if "bbox" in data:
            data["bbox"] = np.asarray(data["bbox"])
            if not data["bbox"].ndim == 2 or not data["bbox"].shape[-1] == 4:
                raise ValueError("Expected shape (num_targets, 4) in data['bbox']")
        if "cls" in data:
            data["cls"] = np.asarray(data["cls"])
            if not data["cls"].ndim == 2:
                raise ValueError("Expected 2D array in data['cls']")
        if "imgidx" in data:
            data["imgidx"] = np.asarray(data["imgidx"])
            if not data["imgidx"].ndim == 2 or not data["imgidx"].shape[-1] == 1:
                raise ValueError("Expected shape (num_targets, 1) in data['imgidx']")
        return data


class ConvertColor(BaseTransform):
    def __init__(self, src: str, dst: str) -> None:
        self.src = src
        self.dst = dst

    @staticmethod
    def _apply_img(img: np.ndarray, cv_color_transform: int) -> None:
        for i, _ in enumerate(img):
            img[i] = cv.cvtColor(img[i], cv_color_transform)

    def __call__(self, data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        if "img" in data:
            if self.src == 'BGR' and self.dst == 'HSV':
                self._apply_img(data["img"], cv.COLOR_BGR2HSV)
            elif self.src == 'RGB' and self.dst == 'HSV':
                self._apply_img(data["img"], cv.COLOR_RGB2HSV)
            elif self.src == 'BGR' and self.dst == 'RGB':
                self._apply_img(data["img"], cv.COLOR_BGR2RGB)
            elif self.src == 'HSV' and self.dst == 'BGR':
                self._apply_img(data["img"], cv.COLOR_HSV2BGR)
            elif self.src == 'HSV' and self.dst == "RGB":
                self._apply_img(data["img"], cv.COLOR_HSV2RGB)
            else:
                raise NotImplementedError
        return data


class Resize(BaseTransform):
    def __init__(self,
                 size: Tuple[int, int],
                 bbox_normalized: bool = False) -> None:
        self.size = size
        self.bbox_normalized = bbox_normalized

    @staticmethod
    def _apply_img(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        res = []
        for i, _ in enumerate(img):
            res.append(cv.resize(img[i], size, interpolation=cv.INTER_AREA))
        return np.stack(res, 0)

    @staticmethod
    def _apply_bbox(bbox: np.ndarray,
                    src_size: Tuple[int, int],
                    dst_size: Tuple[int, int]) -> None:
        bbox[:, ::2] = bbox[:, ::2] * dst_size[0] / src_size[0]
        bbox[:, 1::2] = bbox[:, 1::2] * dst_size[1] / src_size[1]

    def __call__(self, data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        if "img" in data:
            src_size = data["img"].shape[-2:-4:-1] # (T, H, W, C) -> (W, H)
            data["img"] = self._apply_img(data["img"], self.size)
            if "bbox" in data and not self.bbox_normalized:
                if src_size is not None:
                    self._apply_bbox(data["bbox"], src_size, self.size)
        return data


class MakeDivisibleBy(BaseTransform):
    def __init__(self, factor: int) -> None:
        self.factor = factor

    @staticmethod
    def _apply_bbox(bbox: np.ndarray,
                    src_size: Tuple[int, int],
                    dst_size: Tuple[int, int]) -> None:
        bbox[:, ::2] = bbox[:, ::2] * dst_size[0] / src_size[0]
        bbox[:, 1::2] = bbox[:, 1::2] * dst_size[1] / src_size[1]

    def __call__(self, data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        if "img" in data:
            src_size = data["img"].shape[-2:-4:-1] # (T, H, W, C) -> (W, H)
            data["img"] = make_divisible_by(data["img"], self.factor)
            dst_size = data["img"].shape[-2:-4:-1] # (T, H, W, C) -> (W, H)
            if "bbox" in data:
                self._apply_bbox(data["bbox"], src_size, dst_size)
        return data


class ToFloat(BaseTransform):
    def __call__(self, data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        if "img" in data:
            data["img"] = data["img"].astype(np.float32)
        if "bbox" in data:
            data["bbox"] = data["bbox"].astype(np.float32)
        if "cls" in data:
            data["cls"] = data["cls"].astype(np.float32)
        return data


class Normalize(BaseTransform):
    def __init__(self,
                 mean_rgb: Tuple[float, float, float] = [0.485, 0.456, 0.406],
                 scale_rgb: Tuple[float, float, float] = [0.229, 0.224, 0.225]) -> None:
        self.mean_rgb = mean_rgb
        self.scale_rgb = scale_rgb

    def __call__(self, data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        if "img" in data:
            data["img"][..., 0] = (data["img"][..., 0] - self.mean_rgb[0]) / self.scale_rgb[0]
            data["img"][..., 1] = (data["img"][..., 1] - self.mean_rgb[1]) / self.scale_rgb[1]
            data["img"][..., 2] = (data["img"][..., 2] - self.mean_rgb[2]) / self.scale_rgb[2]
        return data


class Clip(BaseTransform):
    def __init__(self, min: float = 0.0, max: float = 255.0) -> None:
        self.min = min
        self.max = max
        assert self.max >= self.min, "min val must be >= max val"

    def __call__(self, data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        if "img" in data:
            data["img"] = np.clip(data["img"], self.min, self.max)
        return data


class ToTensor(BaseTransform):
    def __call__(self, data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        if "img" in data: # (T, H, W, C) -> (T, C, H, W)
            data["img"] = torch.from_numpy(data["img"]).type(torch.float32).permute(0, 3, 1, 2)
        if "bbox" in data:
            data["bbox"] = torch.from_numpy(data["bbox"]).type(torch.float32)
        if "cls" in data:
            data["cls"] = torch.from_numpy(data["cls"]).type(torch.float32)
        if "imgidx" in data:
            data["imgidx"] = torch.from_numpy(data["imgidx"]).type(torch.long)

        return data
