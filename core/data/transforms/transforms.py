import math
import torch
import cv2 as cv
import numpy as np
from core.utils.ops import xywh2xyxy, xyxy2xywh
from typing import Dict, Tuple, Sequence, Union, List
from core.data.transforms.functional import (
    make_array_divisible_by, 
    make_size_divisible_by
)
from abc import ABC, abstractmethod


class TransformInterface(ABC):
    @abstractmethod
    def __call__(
            self, 
            data: Dict[str, np.ndarray]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Applies transformation to data.

        Args:
            'data' (Dict): A dictionary containing image data and annotations, including:
                'img' (numpy.ndarray): Array of images with shape (T, H, W, C),
                    where T is sequence length, H is image height, W is image width,
                    C is number of image channels.
                'bbox' (numpy.ndarray): Array of bounding boxes with shape (T, N, 4),
                    where T is sequence length, N is number of bounding boxes per image.
                    Assumes box coordinates are normalized in range [0, 1) and have format
                    'cxcywh'. If some bounding box is out of image borders after transformation,
                    fills this box with zeros.
                'cls' (numpy.ndarray): Array of class scores with shape (T, N, num_classes),
                    where T is sequence length, N is number of bounding boxes per image.
                    Assumes class scores are normalized in range [0, 1].

        Returns:
            'data_out' (Dict): Transformed data with the same shape as input data.
        """


class Compose(TransformInterface):
    def __init__(self, transforms: Sequence[TransformInterface]):
        super(Compose, self).__init__()
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class CheckFormat(TransformInterface):
    def __init__(self):
        super(CheckFormat, self).__init__()

    def __call__(self, data):
        if 'img' in data:
            if not isinstance(data['img'], np.ndarray):
                raise ValueError("Expected 'img' as numpy ndarray")
            if not data['img'].ndim == 4:
                raise ValueError("Expected 'img' with shape (T, H, W, C)")
        if 'bbox' in data:
            if not isinstance(data['bbox'], np.ndarray):
                raise ValueError("Expected 'bbox' as numpy ndarray")
            if not data['bbox'].ndim == 3 or not data['bbox'].shape[-1] == 4:
                raise ValueError("Expected 'bbox' with shape (T, N, 4)")
        if 'cls' in data:
            if not isinstance(data['cls'], np.ndarray):
                raise ValueError("Expected 'cls' as numpy ndarray")
            if not data['cls'].ndim == 3:
                raise ValueError("Expected 'cls' with shape (T, N, num_classes)")
        return data


class ConvertColor(TransformInterface):
    def __init__(self, color_from: str, color_to: str):
        super(ConvertColor, self).__init__()
        self.cvt_cvtype = self._str_to_cvtype(color_from, color_to)

    def _str_to_cvtype(self, src: str, dst: str) -> int:
        if src == 'BGR' and dst == 'HSV':
            return cv.COLOR_BGR2HSV
        elif src == 'RGB' and dst == 'HSV':
            return cv.COLOR_RGB2HSV
        elif src == 'HSV' and dst == 'BGR':
            return cv.COLOR_HSV2BGR
        elif src == 'HSV' and dst == "RGB":
            return cv.COLOR_HSV2RGB
        elif src == 'RGB' and dst == 'BGR':
            return cv.COLOR_RGB2BGR
        elif src == 'BGR' and dst == 'RGB':
            return cv.COLOR_BGR2RGB
        else:
            raise ValueError("Invalid color transformation type")

    def __call__(self, data):
        if 'img' in data:
            for i in range(len(data['img'])):
                data['img'][i] = cv.cvtColor(data['img'][i], self.cvt_cvtype)
        return data


class Resize(TransformInterface):
    def __init__(
            self, 
            size: Tuple[int, int], 
            make_divisible_by: int = 1):
        super(Resize, self).__init__()
        self.size = make_size_divisible_by(*size, make_divisible_by)

    def __call__(self, data):
        if 'img' in data:
            r_imgs = []
            for i in range(len(data['img'])):
                r_img = cv.resize(data['img'][i], self.size, interpolation=cv.INTER_AREA)
                r_imgs.append(r_img)
            data['img'] = np.stack(r_imgs, 0)

        return data


class PadResize(TransformInterface):
    def __init__(
            self,
            size: Tuple[int, int],
            border_value: int = 114,
            make_divisible_by: int = 1):
        super(PadResize, self).__init__()
        self.size = make_size_divisible_by(*size, make_divisible_by)
        self.border_value = border_value

    def _calc_pads(self, img_w: int, img_h: int) -> List[int]:
        source_aspect = img_w / img_h
        target_aspect = self.size[0] / self.size[1]
        # tblr
        pads = [0, 0, 0, 0]
        if source_aspect > target_aspect:
            # pad top and bottom
            new_height = int(np.round(img_w / target_aspect))
            pads[0] = (new_height - img_h) // 2
            pads[1] = new_height - img_h - pads[0]
        else:
            # pad left and right
            new_width = int(np.round(img_h * target_aspect))
            pads[2] = (new_width - img_w) // 2
            pads[3] = new_width - img_w - pads[2]
        return pads

    def _apply_img(self, img):
        t, h, w, c = img.shape

        # pad
        pads = self._calc_pads(w, h)
        img = np.pad(img,
                     [(0, 0), (pads[0], pads[1]), (pads[2], pads[3]), (0, 0)],
                     mode='constant', constant_values=self.border_value)

        # resize
        res = np.zeros(shape=(t, *self.size[::-1], c), dtype=img.dtype)
        for i, _ in enumerate(res):
            res[i] = cv.resize(img[i], self.size, interpolation=cv.INTER_AREA)

        return res
    
    def _apply_bbox(self, bbox: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        # calc size of padded image
        pads = self._calc_pads(img_w, img_h)
        new_w = img_w + pads[2] + pads[3]
        new_h = img_h + pads[0] + pads[1]

        # to xyxy, denormalize
        bbox = xywh2xyxy(bbox)
        bbox[..., 0::2] *= img_w
        bbox[..., 1::2] *= img_h

        # translate bboxes
        bbox[..., 0::2] += pads[2] # x offset
        bbox[..., 1::2] += pads[0] # y offset

        # normalize
        bbox = xyxy2xywh(bbox)
        bbox[..., 0::2] /= new_w
        bbox[..., 1::2] /= new_h

        return bbox

    def _inv_apply_bbox(self, bbox: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        # calc size of padded image
        pads = self._calc_pads(img_w, img_h)
        new_w = img_w + pads[2] + pads[3]
        new_h = img_h + pads[0] + pads[1]

        # resize
        bbox[..., 0::2] *= (new_w / self.size[0])
        bbox[..., 1::2] *= (new_h / self.size[1])

        # translate
        bbox[..., 0::2] -= pads[2] # x offset
        bbox[..., 1::2] -= pads[0] # y offset

        return bbox

    def __call__(self, data):
        if 'img' in data:
            h, w = data['img'].shape[1:3]
            data['img'] = self._apply_img(data['img'])
            if 'bbox' in data:
                data['bbox'] = self._apply_bbox(data['bbox'], w, h)
        return data


# class CheckBBoxes(TransformInterface):
#     def __init__(self):
#         super(CheckBBoxes, self).__init__()

#     def _apply_bbox(self, bbox: np.ndarray):
#         in_nonzeros = np.count_nonzero((bbox[..., 2] * bbox[..., 3] > 0))

#         # clip
#         bbox_ = xyxy2xywh(np.clip(xywh2xyxy(bbox), 0, 1))
#         mask = bbox_[..., 2] * bbox_[..., 3] > 0

#         num_filtered = np.count_nonzero(mask) - in_nonzeros
#         assert num_filtered == 0, f"Found invalid (empty) boxes"

#     def __call__(self, data):
#         if 'bbox' in data:
#             self._apply_bbox(data['bbox'])
#         return data


# class MakeDivisibleBy(BaseTransform):
#     def __init__(self, factor: int):
#         super().__init__()
#         self.factor = factor

#     def apply_img(self, img):
#         return make_array_divisible_by(img, self.factor)

#     def apply_bbox(self, bbox: np.ndarray, w_scale: float, h_scale: float):
#         bbox[..., ::2] = bbox[..., ::2] * w_scale
#         bbox[..., 1::2] = bbox[..., 1::2] * h_scale

#     def __call__(self, data):
#         if 'img' in data:
#             h, w = data['img'].shape[1:3]
#             data['img'] = self.apply_img(data['img'])
#             h_new, w_new = data['img'].shape[1:3]
#             if 'bbox' in data:
#                 self.apply_bbox(data['bbox'], w / w_new, h / h_new)
#         return data


class ToFloat(TransformInterface):
    def __init__(self):
        super(ToFloat, self).__init__()

    def apply_img(self, img):
        return img.astype(np.float32)

    def __call__(self, data):
        if 'img' in data:
            data['img'] = data['img'].astype(np.float32)
        return data


class Normalize(TransformInterface):
    def __init__(self, mean_rgb: Sequence[float], scale_rgb: Sequence[float]):
        super(Normalize, self).__init__()
        self.mean_rgb = mean_rgb
        self.scale_rgb = scale_rgb

    def __call__(self, data):
        if 'img' in data:
            data['img'] = (data['img'] - self.mean_rgb) / self.scale_rgb
        return data


class Denormalize(TransformInterface):
    def __init__(self, mean_rgb: Sequence[float], scale_rgb: Sequence[float]):
        super(Denormalize, self).__init__()
        self.mean_rgb = mean_rgb
        self.scale_rgb = scale_rgb

    def __call__(self, data):
        if 'img' in data:
            data['img'] = data['img'] * self.scale_rgb + self.mean_rgb
        return data


class Clip(TransformInterface):
    def __init__(self, min: float = 0.0, max: float = 255.0):
        super(Clip, self).__init__()
        assert max >= min, "min must be >= max"
        self.min = min
        self.max = max

    def __call__(self, data):
        if 'img' in data:
            np.clip(data['img'], self.min, self.max, out=data['img'])
        return data


class ToTensor(TransformInterface):
    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self, data):
        if 'img' in data:
            data['img'] = torch.from_numpy(data['img']).type(torch.float32)
            if data['img'].ndim == 4:
                data['img'] = data['img'].permute(0, 3, 1, 2)
            elif data['img'].ndim == 3:
                data['img'] = data['img'].permute(2, 0, 1)
            else:
                raise ValueError("Expected 3D or 4D array")
        if 'bbox' in data:
            data['bbox'] = torch.from_numpy(data['bbox']).type(torch.float32)
        if 'cls' in data:
            data['cls'] = torch.from_numpy(data['cls']).type(torch.float32)
        return data


class ToNumpy(TransformInterface):
    def __init__(self):
        super(ToNumpy, self).__init__()

    def apply_img(self, img: torch.Tensor) -> np.ndarray:
        if img.dim() == 4:
            res = img.permute(0, 2, 3, 1)
        elif img.dim() == 3:
            res = img.permute(1, 2, 0)
        else:
            raise ValueError("Expected 3D or 4D array")
        return res.cpu().numpy()

    # TODO: tensors as inputs, not numpy
    def __call__(self, data):
        if 'img' in data:
            if data['img'].dim() == 4:
                data['img'] = data['img'].permute(0, 2, 3, 1)
            elif data['img'].dim() == 3:
                data['img'] = data['img'].permute(1, 2, 0)
            else:
                raise ValueError("Expected 3D or 4D array")
            data['img'] = data['img'].cpu().numpy()
        if 'bbox' in data:
            data['bbox'] = data['bbox'].cpu().numpy()
        if 'cls' in data:
            data['cls'] = data['cls'].cpu().numpy()
        return data


class RandomJpeg(TransformInterface):
    def __init__(self, min_quality: float = 0.6, probability: float = 0.5):
        super(RandomJpeg, self).__init__()
        self.prob = np.clip(probability, 0.0, 1.0)
        self.min_quality = np.clip(min_quality, 0.0, 1.0)

    def __call__(self, data):
        if np.random.choice([0, 1], size=1, p=[1 - self.prob, self.prob]):
            if 'img' in data:
                quality = min(self.min_quality + np.random.random() * (1.0 - self.min_quality), 1.0)
                encode_params = [int(cv.IMWRITE_JPEG_QUALITY), int(100 * quality)]
                for i in range(len(data['img'])):
                    _, encimg = cv.imencode('.jpg', data['img'][i], encode_params)
                    data['img'][i] = cv.imdecode(encimg, 1)
        return data


class RandomPerspective(TransformInterface):
    def __init__(self,
                 rotate: float = 0.0,
                 translate: float = 0.0,
                 scale: float = 1.0,
                 shear: float = 0.0,
                 perspective: float = 0.0,
                 border_value: int = 114,
                 probabilty: float = 0.5,
                 keep_aspect: bool = True,
                 downscale_only: bool = False):
        super(RandomPerspective, self).__init__()
        self.rotate = np.clip(rotate, 0.0, 360.0)
        self.translate = np.clip(translate, 0.0, 1.0)
        self.scale = np.clip(scale, 1.0, None)
        self.shear = np.clip(shear, 0.0, 90.0)
        self.perspective = np.clip(perspective, 0.0, 0.001)
        self.border_value = border_value
        self.prob = np.clip(probabilty, 0.0, 1.0)
        self.keep_aspect = keep_aspect
        self.downscale_only = downscale_only

    def _construct_matrix(self, img_w: int, img_h: int) -> np.ndarray:
        # center
        mat_c = np.eye(3, dtype=np.float32)
        mat_c[0, 2] = -img_w / 2  # x translation (pixels)
        mat_c[1, 2] = -img_h / 2  # y translation (pixels)

        # perspective
        mat_p = np.eye(3, dtype=np.float32)
        mat_p[2, 0] = np.random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        mat_p[2, 1] = np.random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # scale
        mat_sc = np.eye(3, dtype=np.float32)
        sc_min_ln = np.log(1 / self.scale)
        sc_max_ln = 0.0 if self.downscale_only else np.log(self.scale)
        mat_sc[0, 0] = np.exp(np.random.uniform(sc_min_ln, sc_max_ln))
        mat_sc[1, 1] = mat_sc[0, 0] if self.keep_aspect else np.exp(np.random.uniform(sc_min_ln, sc_max_ln))

        # mat_sc[0, 0] = np.random.uniform(1.0, self.scale)
        # mat_sc[1, 1] = mat_sc[0, 0] if self.keep_aspect else np.random.uniform(1.0, self.scale)
        # if self.downscale_only or np.random.choice([True, False]):
        #     mat_sc[0, 0] = 1 / mat_sc[0, 0]
        #     mat_sc[1, 1] = 1 / mat_sc[1, 1]

        # rotation
        mat_r = np.eye(3, dtype=np.float32)
        a = np.random.uniform(-self.rotate, self.rotate)
        mat_r[:2] = cv.getRotationMatrix2D(angle=a, center=(0, 0), scale=1.0)

        # shear
        mat_sh = np.eye(3, dtype=np.float32)
        mat_sh[0, 1] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        mat_sh[1, 0] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # translation
        mat_t = np.eye(3, dtype=np.float32)
        mat_t[0, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * img_w  # x translation (pixels)
        mat_t[1, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * img_h  # y translation (pixels)

        return mat_t @ mat_sh @ mat_r @ mat_sc @ mat_p @ mat_c

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

    def _apply_img(self, img: np.ndarray, trans_mat: np.ndarray):
        bval = img.shape[-1] * [self.border_value]
        if np.any(trans_mat != np.eye(3)):
            for i, _ in enumerate(img):
                if self.perspective:
                    img[i] = cv.warpPerspective(img[i], trans_mat, dsize=None, borderValue=bval)
                else:
                    img[i] = cv.warpAffine(img[i], trans_mat[:2], dsize=None, borderValue=bval)

    def _apply_bbox(self, bbox: np.ndarray, w: int, h: int, trans_mat: np.ndarray) -> np.ndarray:
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
        if np.random.choice([0, 1], size=1, p=[1 - self.prob, self.prob]):
            if 'img' in data:
                h, w = data['img'].shape[1:3]
                mat = self._construct_matrix(w, h)
                self._apply_img(data['img'], mat)
                if 'bbox' in data:
                    data['bbox'] = self._apply_bbox(data['bbox'], w, h, mat)
        return data


# class RandomCrop(BaseTransform):
#     def __init__(self, min_crop: float, probability: float, freeze_crop_y: bool = False):
#         super(RandomCrop, self).__init__()
#         self.min_crop = np.clip(min_crop, 0.0, 1.0)
#         self.prob = np.clip(probability, 0.0, 1.0)
#         self.freeze_crop_y = freeze_crop_y

#     def apply_img(self, img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
#         x_pix = int(x * img.shape[2])
#         y_pix = int(y * img.shape[1])
#         w_pix = int(w * img.shape[2])
#         h_pix = int(h * img.shape[1])

#         return img[:, y_pix:y_pix + h_pix, x_pix:x_pix + w_pix, :]

#     def apply_bbox(self, bbox: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
#         # recalc coordinates
#         bbox = xywh2xyxy(bbox)
#         bbox[:, :, 0::2] = (bbox[:, :, 0::2] - x) / w
#         bbox[:, :, 1::2] = (bbox[:, :, 1::2] - y) / h
#         bbox = np.clip(bbox, 0.0, 1.0)
#         bbox = xyxy2xywh(bbox)

#         # filter empty bboxes
#         mask = bbox[..., 2] * bbox[..., 3] > 0
#         mask = np.expand_dims(mask, -1).repeat(4, -1).astype(np.int32)
#         bbox *= mask

#         return bbox

#     def apply_cls(self, cls: np.ndarray) -> Optional[Union[np.ndarray, torch.Tensor]]:
#         return cls

#     def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         if np.random.choice([0, 1], size=1, p=[1 - self.prob, self.prob]):
#             # sample crop coordinates
#             crop_w = min(self.min_crop + np.random.random() * (1.0 - self.min_crop), 1.0)
#             crop_h = min(self.min_crop + np.random.random() * (1.0 - self.min_crop), 1.0)
#             crop_x = np.random.random() * (1.0 - crop_w)
#             if self.freeze_crop_y:
#                 crop_y = 1.0 - crop_h
#             else:
#                 crop_y = np.random.random() * (1.0 - crop_h)

#             # apply crop to data
#             if 'img' in data:
#                 data['img'] = self.apply_img(data['img'], crop_x, crop_y, crop_w, crop_h)
#             if 'bbox' in data:
#                 data['bbox'] = self.apply_bbox(data['bbox'], crop_x, crop_y, crop_w, crop_h)

#         return data


class RandomMirror(TransformInterface):
    def __init__(self, x_axis_only: bool, probability: float):
        super(RandomMirror, self).__init__()
        self.x_axis_only = x_axis_only
        self.prob = np.clip(probability, 0.0, 1.0)

    def __call__(self, data):
        flip_type = 0 if self.x_axis_only else np.random.randint(3)
        if np.random.choice([0, 1], size=1, p=[1 - self.prob, self.prob]):
            # x_axis
            if flip_type in [0, 2]:
                if 'img' in data:
                    data['img'] = data['img'][:, :, ::-1, :]
                if 'bbox' in data:
                    data['bbox'][..., 0] = 1.0 - data['bbox'][..., 0]
            # y_axis
            if flip_type in [1, 2]:
                if 'img' in data:
                    data['img'] = data['img'][:, ::-1, :, :]
                if 'bbox' in data:
                    data['bbox'][..., 1] = 1.0 - data['bbox'][..., 1]
        return data


class RandomContrast(TransformInterface):
    def __init__(self, lower: float = 0.75, upper: float = 1.25, probability: float = 0.5):
        super(RandomContrast, self).__init__()
        assert upper >= lower, "contrast upper must be >= lower"
        assert lower >= 0, "contrast lower must be non-negative"
        self.lower = lower
        self.upper = upper
        self.prob = np.clip(probability, 0.0, 1.0)

    def __call__(self, data):
        """
            Expects float images
        """
        if np.random.choice([0, 1], size=1, p=[1 - self.prob, self.prob]):
            alpha = np.random.uniform(self.lower, self.upper)
            if 'img' in data:
                data['img'] *= alpha
        return data


class RandomGamma(TransformInterface):
    def __init__(self, lower: float = 0.5, upper: float = 2.0, probability: float = 0.5):
        super(RandomGamma, self).__init__()
        assert upper >= lower, "gamma upper must be >= lower"
        assert lower >= 0, "gamma lower must be non-negative"
        self.lower = lower
        self.upper = upper
        self.prob = np.clip(probability, 0.0, 1.0)

    def __call__(self, data):
        """
            Expects float images
        """
        if np.random.choice([0, 1], size=1, p=[1 - self.prob, self.prob]):
            gamma = np.random.uniform(self.lower, self.upper)
            if 'img' in data:
                data['img'] = np.power(data['img'], gamma)
        return data


class RandomBrightness(TransformInterface):
    def __init__(self, delta: int = 26, probability: float = 0.5):
        super(RandomBrightness, self).__init__()
        self.delta = np.clip(delta, 0, 255)
        self.prob = np.clip(probability, 0.0, 1.0)

    def __call__(self, data):
        """
            Expects float images
        """
        if np.random.choice([0, 1], size=1, p=[1 - self.prob, self.prob]):
            if 'img' in data:
                data['img'] += np.random.uniform(-self.delta, self.delta)
        return data


class RandomHue(TransformInterface):
    def __init__(self, delta: float = 30.0, src_color: str = 'RGB', probability: float = 0.5):
        super(RandomHue, self).__init__()
        self.delta = np.clip(delta, 0, 360.0)
        self.prob = np.clip(probability, 0.0, 1.0)
        self.to_hsv = ConvertColor(src_color, 'HSV')
        self.from_hsv = ConvertColor('HSV', src_color)

    def apply_img(self, img: np.ndarray, delta: float) -> np.ndarray:
        self.to_hsv.apply_img(img)
        img[..., 0] += delta
        img[..., 0][img[..., 0] > 360.0] -= 360.0
        img[..., 0][img[..., 0] < 0.0] += 360.0
        self.from_hsv.apply_img(img)

        return img

    def __call__(self, data):
        if np.random.choice([0, 1], size=1, p=[1 - self.prob, self.prob]):
            delta = np.random.uniform(-self.delta, self.delta)
            if 'img' in data:
                self.to_hsv(data)
                data['img'][..., 0] += delta
                data['img'][..., 0][data['img'][..., 0] > 360.0] -= 360.0
                data['img'][..., 0][data['img'][..., 0] < 0.0] += 360.0
                self.from_hsv(data)
        return data


class RandomGray(TransformInterface):
    def __init__(self, probability: float = 0.5):
        super(RandomGray, self).__init__()
        self.prob = np.clip(probability, 0.0, 1.0)

    def __call__(self, data):
        if np.random.choice([0, 1], size=1, p=[1 - self.prob, self.prob]):
            if 'img' in data:
                for i in range(len(data['img'])):
                    gray = cv.cvtColor(data['img'][i], cv.COLOR_RGB2GRAY)
                    data['img'][i] = np.stack([gray, gray, gray], axis=-1)
        return data