import os
import cv2 as cv
import numpy as np
from glob import glob
from core.config import CfgNode
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
from core.data.transforms.transforms import BaseTransform
from core.utils.ops import xyxy2xywh


class UCF101_24Dataset(Dataset):
    num_classes = 24

    def __init__(self,
                 cfg: CfgNode,
                 data_path: str,
                 anno_path: str,
                 transforms: Optional[BaseTransform] = None):
        self.seqs = self._parse_seqs(anno_path,
                                     data_path,
                                     cfg.DATASET.SEQUENCE_LENGTH,
                                     cfg.DATASET.SEQUENCE_STRIDE,
                                     cfg.DATASET.SEQUENCE_DILATE)
        self.transforms = transforms
        self.max_labels = cfg.INPUT.PAD_LABELS_TO

    def __len__(self):
        return len(self.seqs)

    def _parse_seqs(self,
                    anno_root: str,
                    data_root: str,
                    seq_length: int,
                    seq_stride: int,
                    seq_dilate: int) -> List[List[Tuple[str, dict]]]:
        result = []

        # parse seqpaths
        def get_seqpath(path: str) -> str:
            head, ssd = os.path.split(path)
            sd = os.path.split(head)[1]
            return os.path.join(sd, ssd)
        seqpaths = sorted(glob(os.path.join(anno_root, "*/*")))
        seqpaths = [get_seqpath(s) for s in seqpaths]

        # prepare seqs
        for seqpath in seqpaths:
            annoimgs = []
            imgs = sorted(glob(os.path.join(data_root, seqpath, "*.jpg")))
            annos = sorted(glob(os.path.join(anno_root, seqpath, "*.txt")))
            for img in imgs:
                filename = os.path.basename(img)
                filename = os.path.splitext(filename)[0]
                anno = os.path.join(anno_root, seqpath, filename + ".txt")
                if anno not in annos:
                    anno = None
                annoimgs.append((img, anno))
            # split on seqs
            while seq_length * seq_dilate <= len(annoimgs):
                anno_seq, annoimgs = annoimgs[:seq_length:seq_dilate], annoimgs[seq_stride:]
                result.append(anno_seq)

        return result

    def __getitem__(self, idx):
        imgs, bboxes, classes = [], [], []
        for ipath, apath in self.seqs[idx]:
            # read image
            img = cv.imread(ipath, cv.IMREAD_COLOR)
            imgs.append(img)

            # read objects
            box = np.zeros(shape=(self.max_labels, 4), dtype=np.float32)
            cls = np.zeros(shape=(self.max_labels, self.num_classes), dtype=np.float32)
            if apath != None:
                with open(apath, 'r') as f:
                    for i, line in enumerate(f.readlines()):
                        elements = list(map(float, line.split()))
                        cls_idx = int(elements[0]) - 1
                        if cls_idx < self.num_classes:
                            assert self.num_classes > cls_idx >= 0
                            cls[i][cls_idx] = 1.0
                            box[i] = xyxy2xywh(np.asarray(elements[1:5]))
            box[:, 0::2] /= img.shape[1]
            box[:, 1::2] /= img.shape[0]
            bboxes.append(box)
            classes.append(cls)

        item = {}
        item['img'] = np.stack(imgs, 0)     # (T, H, W, C)
        item['bbox'] = np.stack(bboxes, 0)  # (T, max_labels, 4)
        item['cls'] = np.stack(classes, 0)  # (T, max_labels, num_classes)

        # apply transforms
        if self.transforms:
            item = self.transforms(item)

        return item

    def visualize(self, tick_ms: int = 0):
        for i in range(0, self.__len__()):
            item = self.__getitem__(i)
            for img, box, cls in zip(item["img"], item["bbox"], item["cls"]):
                img = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                w, h = img.shape[-2:-4:-1]

                for b, c in zip(box, cls):
                    b = b.cpu().numpy()

                    # skip empty
                    if b[2] * b[3] == 0:
                        continue

                    # draw bbox
                    b[::2] *= w
                    b[1::2] *= h
                    tl = b[:2] - (b[2:4] / 2)
                    br = b[:2] + (b[2:4] / 2)
                    cv.rectangle(img, tl.astype(np.int32), br.astype(np.int32), (0, 255, 0), 2)

                    # draw labels
                    pt = tl.astype(np.int32) + [0, 10]
                    for idx in np.where(c == 1.0)[0]:
                        text = f"{idx} {c[idx]:.2f}"
                        cv.putText(img, text, pt, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        pt += [0, 10]

                cv.imshow('img', img)
                if cv.waitKey(tick_ms) & 0xFF == ord('q'):
                    return
