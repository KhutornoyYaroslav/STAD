import os
import cv2 as cv
import numpy as np
from glob import glob
from typing import Dict, List, Optional
from core.config import CfgNode
from torch.utils.data import Dataset
from core.data.transforms.transforms import BaseTransform


class SingleImageDataset(Dataset):
    def __init__(self,
                 cfg: CfgNode,
                 img_path: str,
                 label_path: str,
                 transforms: Optional[BaseTransform] = None):
        self.files = self._scan_files(img_path, label_path)
        self.transforms = transforms
        self.max_labels = cfg.DATASET.PAD_LABELS_TO
        self.num_classes = cfg.MODEL.HEAD.NUM_CLASSES

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        item = {}

        # read image
        img = cv.imread(file["image_path"], cv.IMREAD_COLOR)
        item["img"] = np.expand_dims(img, axis=0) # (1, H, W, C)

        # read label
        item["box"], item["cls"] = self._read_label(file["label_path"])
        item["box"] = np.expand_dims(item["box"], axis=0) # (1, max_labels, 4)
        item["cls"] = np.expand_dims(item["cls"], axis=0) # (1, max_labels, num_classes)

        # apply transforms
        if self.transforms:
            item = self.transforms(item)

        return item

    def _get_basename(self, path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    def _scan_files(self, img_path: str, label_root: str) -> Dict[str, List[str]]:
        images = sorted(glob(img_path))
        labels = sorted(glob(label_root))
        data = []
        for image, label in zip(images, labels):
            assert self._get_basename(image) == self._get_basename(label)
            data.append({"image_path": image, "label_path": label})
        return data

    def _read_label(self, label_path: str):
        box = np.zeros(shape=(self.max_labels, 4), dtype=np.float32)
        cls = np.zeros(shape=(self.max_labels, self.num_classes), dtype=np.float32)

        with open(label_path) as f:
            for i, line in enumerate(f.readlines()):
                elements = list(map(float, line.split()))
                cls_idx = int(elements[0])
                if cls_idx < self.num_classes:
                    cls[i][cls_idx] = 1.0
                    box[i] = elements[1:5]

        return box, cls

    def visualize(self, tick_ms: int = 0):
        for i in range(0, self.__len__()):
            item = self.__getitem__(i)
            for img, box, cls in zip(item["img"], item["box"], item["cls"]):
                img = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                w, h = img.shape[-2:-4:-1]

                for b, c in zip(box, cls):
                    b = b.cpu().numpy()
                    b[::2] *= w
                    b[1::2] *= h
                    tl = b[:2] - (b[2:4] / 2)
                    br = b[:2] + (b[2:4] / 2)
                    cv.rectangle(img, tl.astype(np.int32), br.astype(np.int32), (0, 255, 0), 2)

                cv.imshow('img', img)
                if cv.waitKey(tick_ms) & 0xFF == ord('q'):
                    return
