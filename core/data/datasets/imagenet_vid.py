import os
import cv2 as cv
import numpy as np
from glob import glob
from typing import Dict, List, Optional
from core.config import CfgNode
from torch.utils.data import Dataset
from core.data.transforms.transforms import BaseTransform
import xml.etree.ElementTree as ET


class ImagenetVidDataset(Dataset):
    def __init__(self,
                 cfg: CfgNode,
                 data_root: str,
                 anno_root: str,
                 transforms: Optional[BaseTransform] = None):
        self.data_root = data_root
        self.anno_seqs = self._prepare_anno_seqs(anno_root,
                                                 cfg.DATASET.SEQUENCE_LENGTH,
                                                 cfg.DATASET.SEQUENCE_STRIDE)
        self.class_labels = self._parse_class_labels(cfg.DATASET.CLASS_LABELS_FILE)
        self.transforms = transforms
        self.max_labels = cfg.INPUT.PAD_LABELS_TO
        self.num_classes = cfg.MODEL.HEAD.NUM_CLASSES
        assert len(self.class_labels) == self.num_classes

    def __len__(self):
        return len(self.anno_seqs)

    def __getitem__(self, idx):
        imgs = []
        boxes = []
        classes = []
        for anno in self.anno_seqs[idx]:
            # parse xml
            tree = ET.parse(anno)
            root = tree.getroot()
            img_w = int(root.find('size/width').text)
            img_h = int(root.find('size/height').text)

            # read image
            folder = root.find('folder').text
            filename = root.find('filename').text
            img_path = os.path.join(self.data_root, folder, filename + ".JPEG")
            img = cv.imread(img_path, cv.IMREAD_COLOR)
            imgs.append(img)

            # read objects
            box = np.zeros(shape=(self.max_labels, 4), dtype=np.float32)
            cls = np.zeros(shape=(self.max_labels, self.num_classes), dtype=np.float32)

            for i, obj in enumerate(root.findall('object')):
                # bbox
                x1 = int(obj.find('bndbox/xmin').text) / img_w
                y1 = int(obj.find('bndbox/ymin').text) / img_h
                x2 = int(obj.find('bndbox/xmax').text) / img_w
                y2 = int(obj.find('bndbox/ymax').text) / img_h
                box[i] = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1] # xcycwh

                # class
                obj_name = obj.find('name').text
                cls_idx = self.class_labels.index(obj_name)
                cls[i][cls_idx] = 1.0

            boxes.append(box)
            classes.append(cls)

        item = {}
        item["img"] = np.stack(imgs, 0) # (T, H, W, C)
        item["bbox"] = np.stack(boxes, 0) # (T, max_labels, 4)
        item["cls"] = np.stack(classes, 0) # (T, max_labels, num_classes)

        # apply transforms
        if self.transforms:
            item = self.transforms(item)

        return item

    def _parse_class_labels(self, file: str) -> Dict[str, str]:
        result = []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                id, name = line[:-1].split(':', 1)
                result.append(id)
        return result

    def _prepare_anno_seqs(self,
                           anno_root: str,
                           seq_length: int,
                           seq_stride: int = 1) -> List[List[str]]:
        result = []
        anno_dirs = sorted(glob(os.path.join(anno_root, "*")))
        for anno_dir in anno_dirs:
            annos = sorted(glob(os.path.join(anno_dir, "*.xml")))
            while seq_length * seq_stride <= len(annos):
                anno_seq, annos = annos[:seq_length:seq_stride], annos[seq_length * seq_stride:]
                result.append(anno_seq)
        return result

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
                    cv.rectangle(img, tl.astype(np.int32), br.astype(np.int32), (0, 255, 0), 2)

                cv.imshow('img', img)
                if cv.waitKey(tick_ms) & 0xFF == ord('q'):
                    return
