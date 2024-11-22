import os
import json
import cv2 as cv
import numpy as np
from glob import glob
from typing import Dict, List, Optional
from core.config import CfgNode
from torch.utils.data import Dataset
from core.data.transforms.transforms import BaseTransform


class CrowdHumanDataset(Dataset):
    def __init__(self,
                 cfg: CfgNode,
                 data_path: str,
                 anno_path: str,
                 transforms: Optional[BaseTransform] = None):
        self.data_root = data_path
        self.annos = self._parse_anno(anno_path)
        self.transforms = transforms
        self.max_labels = cfg.INPUT.PAD_LABELS_TO
        self.num_classes = cfg.MODEL.HEAD.NUM_CLASSES
        assert self.num_classes == 2

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = self.annos[idx]
        item = {}

        # read image
        img_file = os.path.join(self.data_root, anno['ID'] + ".jpg")
        img = cv.imread(img_file, cv.IMREAD_COLOR)
        item['img'] = np.expand_dims(img, axis=0) # (1, H, W, C)

        # read label
        item['bbox'], item['cls'] = self._read_label(anno, img.shape[1], img.shape[0])
        item['bbox'] = np.expand_dims(item['bbox'], axis=0) # (1, 2 * max_labels, 4)
        item['cls'] = np.expand_dims(item['cls'], axis=0) # (1, 2 * max_labels, num_classes)

        # apply transforms
        if self.transforms:
            item = self.transforms(item)

        return item

    def _parse_anno(self, anno_path: str) -> List[dict]:
        data = []
        with open(anno_path, "r") as f:
            for line in f.read().splitlines():
                data.append(json.loads(line))
        return data

    def _read_label(self, anno: dict, img_w: int, img_h: int):
        box = np.zeros(shape=(2 * self.max_labels, 4), dtype=np.float32)
        cls = np.zeros(shape=(2 * self.max_labels, self.num_classes), dtype=np.float32)
        for i, object in enumerate(anno.get('gtboxes', [])):
            if object['tag'] == 'person':
                # person bbox
                x, y, w, h = object['vbox']
                x = x / img_w
                y = y / img_h
                w = w / img_w
                h = h / img_h
                box[i] = [(x + w / 2), (y + h / 2), w, h]
                cls[i][0] = 1.0
                # head bbox
                x, y, w, h = object['hbox']
                x = x / img_w
                y = y / img_h
                w = w / img_w
                h = h / img_h
                box[i + self.max_labels // 2] = [(x + w / 2), (y + h / 2), w, h]
                cls[i + self.max_labels // 2][1] = 1.0
        return box, cls

    def visualize(self, tick_ms: int = 0):
        for i in range(0, self.__len__()):
            item = self.__getitem__(i)
            for img, box, cls in zip(item["img"], item["bbox"], item["cls"]):
                img = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                w, h = img.shape[-2:-4:-1]

                for b, c in zip(box, cls):
                    b = b.cpu().numpy()
                    b[::2] *= w
                    b[1::2] *= h
                    tl = b[:2] - (b[2:4] / 2)
                    br = b[:2] + (b[2:4] / 2)
                    cv.rectangle(img, tl.astype(np.int32), br.astype(np.int32), (0, 255, 0), 1)
                    cv.circle(img, b[:2].astype(np.int32), 1, (0, 0, 255), 1)

                cv.imshow('img', img)
                if cv.waitKey(tick_ms) & 0xFF == ord('q'):
                    return
