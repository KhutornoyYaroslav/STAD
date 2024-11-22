import math
import torch
import cv2 as cv
import numpy as np
from typing import Dict, Tuple, Sequence, Any, Optional, Union
from core.data.transforms.functional import make_array_divisible_by
from core.utils.ops import xywh2xyxy, xyxy2xywh


class BaseTransform:
    def __init__(self):
        pass

    def apply_img(self, img: np.ndarray) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """
        Applies transformation to images.

        Args:
            'img' (numpy.ndarray): Array of images with shape (T, H, W, C),
                where T is sequence length, H is image height, W is image width,
                C is number of image channels.
        
        Returns:
            (numpy.ndarray): Transformed array of images with shape (T, H, W, C).
        """
        pass

    def apply_bbox(self, bbox: np.ndarray) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """
        Applies transformation to bounding boxes.
        If some bounding box is out of image borders after transformation,
        fills this box with zeros.

        Args:
            'bbox' (numpy.ndarray): Array of bounding boxes with shape (T, N, 4),
                where T is sequence length, N is number of bounding boxes per image.
                Assumes box coordinates are normalized in range [0, 1) and have format
                'cxcywh'.

        Returns:
            (numpy.ndarray): Transformed array of bounding boxes with shape (T, N, 4).
        """
        pass

    def apply_cls(self, cls: np.ndarray) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """
        Applies transformation to class scores of bounding boxes.

        Args:
            'cls' (numpy.ndarray): Array of class scores with shape (T, N, num_classes),
                where T is sequence length, N is number of bounding boxes per image.
                Assumes class scores are normalized in range [0, 1].

        Returns:
            (numpy.ndarray): Transformed array of class scores with shape (T, N, num_classes).
        """
        pass

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies transformation to data.

        Args:
            'data' (Dict): A dictionary containing image data and annotations.
                May include:
                    'img' (numpy.ndarray): The input image.
                    'bbox' (numpy.ndarray): Bounding boxes.
                    'cls' (numpy.ndarray): Class scores.

        Returns:
            (Dict): Transformed data dictionary.
        """


class Compose(BaseTransform):
    def __init__(self, transforms: Sequence[BaseTransform]):
        super().__init__()
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class CheckFormat(BaseTransform):
    def __init__(self):
        super().__init__()

    def apply_img(self, img):
        if not isinstance(img, np.ndarray):
            raise ValueError("Expected 'img' as numpy ndarray")
        if not img.ndim == 4:
            raise ValueError("Expected 'img' with shape (T, H, W, C)")
        return None

    def apply_bbox(self, bbox):
        if not isinstance(bbox, np.ndarray):
            raise ValueError("Expected 'bbox' as numpy ndarray")
        if not bbox.ndim == 3 or not bbox.shape[-1] == 4:
            raise ValueError("Expected 'bbox' with shape (T, N, 4)")
        return None
    
    def apply_cls(self, cls):
        if not isinstance(cls, np.ndarray):
            raise ValueError("Expected 'cls' as numpy ndarray")
        if not cls.ndim == 3:
            raise ValueError("Expected 'cls' with shape (T, N, num_classes)")

    def __call__(self, data):
        if 'img' in data:
            self.apply_img(data['img'])
        if 'bbox' in data:
            self.apply_bbox(data['bbox'])
        if 'cls' in data:
            self.apply_cls(data['cls'])
        return data


class ConvertColor(BaseTransform):
    def __init__(self, src: str, dst: str):
        super().__init__()
        self._str_to_cvtype(src, dst)

    def _str_to_cvtype(self, src: str, dst: str):
        if src == 'BGR' and dst == 'HSV':
            self.cvt_cvtype = cv.COLOR_BGR2HSV
        elif src == 'RGB' and dst == 'HSV':
            self.cvt_cvtype = cv.COLOR_RGB2HSV
        elif src == 'HSV' and dst == 'BGR':
            self.cvt_cvtype = cv.COLOR_HSV2BGR
        elif src == 'HSV' and dst == "RGB":
            self.cvt_cvtype = cv.COLOR_HSV2RGB
        elif src == 'RGB' and dst == 'BGR':
            self.cvt_cvtype = cv.COLOR_RGB2BGR
        elif src == 'BGR' and dst == 'RGB':
            self.cvt_cvtype = cv.COLOR_BGR2RGB
        else:
            raise NotImplementedError

    def apply_img(self, img):
        for i, _ in enumerate(img):
            img[i] = cv.cvtColor(img[i], self.cvt_cvtype)
        return None

    def __call__(self, data):
        if 'img' in data:
            self.apply_img(data['img'])
        return data


class Resize(BaseTransform):
    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.size = size

    def apply_img(self, img):
        res = []
        for i, _ in enumerate(img):
            res.append(cv.resize(img[i], self.size, interpolation=cv.INTER_AREA))
        return np.stack(res, 0)

    def __call__(self, data):
        if 'img' in data:
            data['img'] = self.apply_img(data['img'])
        return data


class MakeDivisibleBy(BaseTransform):
    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def apply_img(self, img):
        return make_array_divisible_by(img, self.factor)

    def apply_bbox(self, bbox: np.ndarray, w_scale: float, h_scale: float):
        bbox[..., ::2] = bbox[..., ::2] * w_scale
        bbox[..., 1::2] = bbox[..., 1::2] * h_scale

    def __call__(self, data):
        if 'img' in data:
            h, w = data['img'].shape[1:3]
            data['img'] = self.apply_img(data['img'])
            h_new, w_new = data['img'].shape[1:3]
            if 'bbox' in data:
                self.apply_bbox(data['bbox'], w / w_new, h / h_new)
        return data


class ToFloat(BaseTransform):
    def __init__(self):
        super().__init__()

    def apply_img(self, img):
        return img.astype(np.float32)

    def __call__(self, data):
        if 'img' in data:
            data['img'] = self.apply_img(data['img'])
        return data


class Normalize(BaseTransform):
    def __init__(self, mean_rgb: Sequence[float], scale_rgb: Sequence[float]):
        super().__init__()
        self.mean_rgb = mean_rgb
        self.scale_rgb = scale_rgb

    def apply_img(self, img):
        return (img - self.mean_rgb) / self.scale_rgb

    def __call__(self, data):
        if 'img' in data:
            data['img'] = self.apply_img(data['img'])
        return data


class Denormalize(BaseTransform):
    def __init__(self, mean_rgb: Sequence[float], scale_rgb: Sequence[float]):
        super().__init__()
        self.mean_rgb = mean_rgb
        self.scale_rgb = scale_rgb

    def apply_img(self, img):
        return img * self.scale_rgb + self.mean_rgb

    def __call__(self, data):
        if 'img' in data:
            data['img'] = self.apply_img(data['img'])
        return data


class Clip(BaseTransform):
    def __init__(self, min: float = 0.0, max: float = 255.0):
        super().__init__()
        self.min = min
        self.max = max
        assert self.max >= self.min, "min must be >= max"

    def apply_img(self, img):
        return np.clip(img, self.min, self.max)

    def __call__(self, data):
        if 'img' in data:
            data['img'] = self.apply_img(data['img'])
        return data


class ToTensor(BaseTransform):
    def __init__(self):
        super().__init__()

    def apply_img(self, img):
        res = torch.from_numpy(img).type(torch.float32)
        if res.ndim == 4:
            res = res.permute(0, 3, 1, 2)
        elif res.ndim == 3:
            res = res.permute(2, 0, 1)
        else:
            raise ValueError("Expected 3D or 4D array")
        return res

    def apply_bbox(self, bbox):
        return torch.from_numpy(bbox).type(torch.float32)
    
    def apply_cls(self, cls):
        return torch.from_numpy(cls).type(torch.float32)

    def __call__(self, data):
        if 'img' in data:
            data['img'] = self.apply_img(data['img'])
        if 'bbox' in data:
            data['bbox'] = self.apply_bbox(data['bbox'])
        if 'cls' in data:
            data['cls'] = self.apply_cls(data['cls'])
        return data


class ToNumpy(BaseTransform):
    def __init__(self):
        super().__init__()

    def apply_img(self, img: torch.Tensor) -> np.ndarray:
        if img.dim() == 4:
            res = img.permute(0, 2, 3, 1)
        elif img.dim() == 3:
            res = img.permute(1, 2, 0)
        else:
            raise ValueError("Expected 3D or 4D array")
        return res.cpu().numpy()

    def apply_bbox(self, bbox: torch.Tensor) -> np.ndarray:
        return bbox.cpu().numpy()
    
    def apply_cls(self, cls: torch.Tensor) -> np.ndarray:
        return cls.cpu().numpy()

    def __call__(self, data):
        if 'img' in data:
            data['img'] = self.apply_img(data['img'])
        if 'bbox' in data:
            data['bbox'] = self.apply_bbox(data['bbox'])
        if 'cls' in data:
            data['cls'] = self.apply_cls(data['cls'])
        return data


class RandomJpeg(BaseTransform):
    def __init__(self, min_quality: float = 0.6, probabilty: float = 0.5):
        super().__init__()
        self.prob = np.clip(probabilty, 0.0, 1.0)
        self.min_quality = np.clip(min_quality, 0.0, 1.0)

    def apply_img(self, img):
        quality = min(self.min_quality + np.random.random() * (1.0 - self.min_quality), 1.0)
        encode_params = [int(cv.IMWRITE_JPEG_QUALITY), int(100 * quality)]
        for i, _ in enumerate(img):
            _, encimg = cv.imencode('.jpg', img[i], encode_params)
            img[i] = cv.imdecode(encimg, 1)

    def __call__(self, data):
        if np.random.choice([0, 1], size=1, p=[1 - self.prob, self.prob]):
            if 'img' in data:
                self.apply_img(data['img'])
        return data


class RandomPerspective(BaseTransform):
    def __init__(self,
                 rotate: float = 0.0,
                 translate: float = 0.0,
                 scale: float = 0.0,
                 shear: float = 0.0,
                 perspective: float = 0.0,
                 border_value: int = 114):
        super().__init__()
        self.rotate = np.clip(rotate, 0.0, 360.0)
        self.translate = np.clip(translate, 0.0, 1.0)
        self.scale = np.clip(scale, 0.0, 0.9)
        self.shear = np.clip(shear, 0.0, 90.0)
        self.perspective = np.clip(perspective, 0.0, 0.001)
        self.border_value = border_value

    def _construct_matrix(self, img_w: int, img_h: int) -> np.ndarray:
        # center
        mat_c = np.eye(3, dtype=np.float32)
        mat_c[0, 2] = -img_w / 2  # x translation (pixels)
        mat_c[1, 2] = -img_h / 2  # y translation (pixels)

        # perspective
        mat_p = np.eye(3, dtype=np.float32)
        mat_p[2, 0] = np.random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        mat_p[2, 1] = np.random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # rotation and scale
        mat_r = np.eye(3, dtype=np.float32)
        a = np.random.uniform(-self.rotate, self.rotate)
        s = np.random.uniform(1 - self.scale, 1 + self.scale)
        mat_r[:2] = cv.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # shear
        mat_s = np.eye(3, dtype=np.float32)
        mat_s[0, 1] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        mat_s[1, 0] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # translation
        mat_t = np.eye(3, dtype=np.float32)
        mat_t[0, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * img_w  # x translation (pixels)
        mat_t[1, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * img_h  # y translation (pixels)

        return mat_t @ mat_s @ mat_r @ mat_p @ mat_c

    def _box_candidates(self,
                        bbox1: np.ndarray, # original, (4, N), 'xyxy'
                        bbox2: np.ndarray, # augmented, (4, N), 'xyxy'
                        wh_thr: float = 2,
                        ar_thr: float = 100,
                        area_thr: float = 0.1,
                        eps: float = 1e-16):
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        # aspect ratio
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
        # candidates
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)

    def apply_img(self, img: np.ndarray, trans_mat: np.ndarray):
        bval = img.shape[-1]*[self.border_value]
        if np.any(trans_mat != np.eye(3)):
            for i, _ in enumerate(img):
                if self.perspective:
                    img[i] = cv.warpPerspective(img[i], trans_mat, dsize=None, borderValue=bval)
                else:
                    img[i] = cv.warpAffine(img[i], trans_mat[:2], dsize=None, borderValue=bval)

    def apply_bbox(self, bbox: np.ndarray, w: int, h: int, trans_mat: np.ndarray) -> np.ndarray:
        t, n = bbox.shape[0:2]

        # to xyxy, denormalize
        bbox = xywh2xyxy(bbox)
        bbox *= [w, h, w, h]

        # as corner points x,y,1
        total_boxes = bbox.shape[0] * bbox.shape[1]
        xy = np.ones(shape=(4 * total_boxes, 3), dtype=bbox.dtype)
        xy[:, :2] = bbox.reshape(-1, 4)[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(4 * total_boxes, 2) # x1y1, x2y2, x1y2, x2y1

        # transform
        xy = xy @ trans_mat.T
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(total_boxes, 8) # perspective rescale or affine

        # new bboxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new_bbox = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bbox.dtype).reshape(4, total_boxes).T
        new_bbox = new_bbox.reshape(t, n, 4)

        # clip
        new_bbox[..., 0::2] = np.clip(new_bbox[..., 0::2], 0, w)
        new_bbox[..., 1::2] = np.clip(new_bbox[..., 1::2], 0, h)

        # filter bad bboxes (filling by zeros)
        new_bbox = new_bbox.reshape(-1, 4)
        mask = self._box_candidates(bbox.reshape(-1, 4).T, new_bbox.T)
        mask = np.expand_dims(mask, -1).repeat(4, -1).astype(np.int32)
        new_bbox *= mask
        new_bbox = new_bbox.reshape(t, n, 4)

        # normalize, to xywh
        new_bbox /= [w, h, w, h]
        return xyxy2xywh(new_bbox)

    def __call__(self, data):
        if 'img' in data:
            h, w = data['img'].shape[1:3]
            mat = self._construct_matrix(w, h)
            self.apply_img(data['img'], mat)
            if 'bbox' in data:
                data['bbox'] = self.apply_bbox(data['bbox'], w, h, mat)
        return data





# class PadResize(BaseTransform):
#     # TODO: add _apply_box
#     def __init__(self, size: Tuple[int, int]):
#         self.size = size

#     @staticmethod
#     def _apply_img(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
#         t, h, w, c = img.shape

#         # pad
#         source_aspect = w / h
#         target_aspect = size[0] / size[1]

#         pads = [0, 0, 0, 0] # tblr
#         if source_aspect > target_aspect:
#             # pad top and bottom
#             new_height = int(np.round(w / target_aspect))
#             pads[0] = (new_height - h) // 2
#             pads[1] = new_height - h - pads[0]
#         else:
#             # pad left and right
#             new_width = int(np.round(h * target_aspect))
#             pads[2] = (new_width - w) // 2
#             pads[3] = new_width - w - pads[2]

#         img = np.pad(img,
#                      [(0, 0), (pads[0], pads[1]), (pads[2], pads[3]), (0, 0)],
#                      mode='constant', constant_values=0)
#         assert img.shape[1] == img.shape[2]

#         # resize
#         res = np.zeros(shape=(t, *size, c), dtype=img.dtype)
#         for i, _ in enumerate(res):
#             res[i] = cv.resize(img[i], size, interpolation=cv.INTER_AREA)

#         return res

#     def __call__(self, data: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
#         if 'img' in data:
#             data['img'] = self._apply_img(data['img'], self.size)
#         return data
