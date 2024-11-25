import os
import pickle
import cv2 as cv
import numpy as np
from glob import glob
from core.config import CfgNode
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
from core.data.transforms.transforms import BaseTransform


class UCF101_24Dataset(Dataset):
    def __init__(self,
                 cfg: CfgNode,
                 data_path: str,
                 anno_path: str,
                 transforms: Optional[BaseTransform] = None):
        self.data_root = data_path
        self.anno_seqs = self._parse_anno(anno_path,
                                          cfg.DATASET.SEQUENCE_LENGTH,
                                          cfg.DATASET.SEQUENCE_STRIDE)
        self.class_labels = self._parse_class_labels(anno_path)
        self.transforms = transforms
        self.max_labels = cfg.INPUT.PAD_LABELS_TO
        self.num_classes = cfg.MODEL.HEAD.NUM_CLASSES
        assert self.num_classes == len(self.class_labels) == 24

    def __len__(self):
        return len(self.anno_seqs)

    def _parse_class_labels(self, filename: str) -> List[str]:
        result = []
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding="bytes")
            for label in data[b'labels']:
                result.append(label.decode('utf-8'))
        return result

    def _parse_anno(self,
                    filename: str,
                    seq_length: int,
                    seq_stride: int = 1) -> List[List[Tuple[str, dict]]]:
        result = []
        with open(filename, 'rb') as f:
            # deserialize
            data = pickle.load(f, encoding="bytes")
            labels = [label.decode('utf-8') for label in data[b'labels']]

            # for each video
            for gttubes, nframes, res in zip(data[b'gttubes'].items(),
                                             data[b'nframes'].items(),
                                             data[b'resolution'].items()):
                assert gttubes[0] == nframes[0] == res[0]

                # video props
                img_h, img_w = res[1]
                video_length = nframes[1]
                video_name = gttubes[0].decode("utf-8")

                # video frames
                frames = sorted(glob(os.path.join(self.data_root, video_name, "*.*")))
                assert len(frames) == video_length

                # parse gttubes
                annos_per_frames = [(fname, []) for fname in frames] # fname, list(dict)
                for label_idx, tubes in gttubes[1].items():
                    for tube in tubes:
                        tube = np.asarray(tube, dtype=np.int32)
                        for frame_idx, x1, y1, x2, y2 in tube: # use as frame_idx = (frame_idx - 1)
                            obj_anno = {
                                'label': labels[label_idx],
                                'bbox': [
                                    (x1 + x2) / 2 / img_w,
                                    (y1 + y2) / 2 / img_h,
                                    (x2 - x1) / img_w,
                                    (y2 - y1) / img_h
                                ]
                            }
                            annos_per_frames[frame_idx - 1][1].append(obj_anno)

                # split on seqs
                while seq_length * seq_stride <= len(annos_per_frames):
                    anno_seq, annos_per_frames = annos_per_frames[:seq_length:seq_stride], annos_per_frames[seq_length * seq_stride:]
                    # filter seq with at least one empty frame
                    seq_has_empty = False
                    for f in anno_seq:
                        if not len(f[1]):
                            seq_has_empty = True
                            break
                    if not seq_has_empty:
                        result.append(anno_seq)

        return result

    def __getitem__(self, idx):
        imgs = []
        bboxes = []
        classes = []

        for file, anno in self.anno_seqs[idx]:
            # read image
            img = cv.imread(file, cv.IMREAD_COLOR)
            imgs.append(img)

            # read objects
            box = np.zeros(shape=(self.max_labels, 4), dtype=np.float32)
            cls = np.zeros(shape=(self.max_labels, self.num_classes), dtype=np.float32)
            for i, obj in enumerate(anno):
                box[i] = obj['bbox']
                cls[i][self.class_labels.index(obj['label'])] = 1.0
            bboxes.append(box)
            classes.append(cls)

        item = {}
        item['img'] = np.stack(imgs, 0) # (T, H, W, C)
        item['bbox'] = np.stack(bboxes, 0) # (T, max_labels, 4)
        item['cls'] = np.stack(classes, 0) # (T, max_labels, num_classes)

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
                    b[::2] *= w
                    b[1::2] *= h
                    tl = b[:2] - (b[2:4] / 2)
                    br = b[:2] + (b[2:4] / 2)
                    cv.rectangle(img, tl.astype(np.int32), br.astype(np.int32), (0, 255, 0), 2)

                cv.imshow('img', img)
                if cv.waitKey(tick_ms) & 0xFF == ord('q'):
                    return
