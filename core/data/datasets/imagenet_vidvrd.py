import os
import json
import cv2 as cv
import numpy as np
from glob import glob
from typing import List, Optional, Tuple, Any
from core.config import CfgNode
from torch.utils.data import Dataset
from core.data.transforms.transforms import BaseTransform
from core.utils.ops import xyxy2xywh


class ImagenetVidVrdDataset(Dataset):
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

    def _parse_class_labels(self, file: str) -> List[str]:
        result = []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                name = line[:-1]
                if name:
                    result.append(name)
        return result

    def _parse_anno(self, path: str) -> List[Tuple[str, Any]]:
        with open(path, 'r') as f:
            data = json.load(f)
        # parse labels
        tid_cat_map = {}
        for d in data['subject/objects']:
            tid_cat_map[d['tid']] = d['category']

        # # filter empty frames
        # for d in data['relation_instances']:
        #    print(path, d['begin_fid'], d['end_fid'], len(data['trajectories']))

        # scan frame files
        files = sorted(glob(os.path.join(self.data_root, data['video_id'], '*')))

        # parse objects
        frames_anno = []
        for file, frame_data in zip(files, data['trajectories']):
            if not len(frame_data):
                continue

            frame_objects = []
            for obj in frame_data:
                bbox = [
                    obj['bbox']['xmin'],
                    obj['bbox']['ymin'],
                    obj['bbox']['xmax'],
                    obj['bbox']['ymax'],
                ]
                bbox = np.asarray(bbox, dtype=np.float32)
                bbox[::2] /= data['width']
                bbox[1::2] /= data['height']
                bbox = xyxy2xywh(bbox)
                frame_objects.append({'label': tid_cat_map[obj['tid']], 'bbox': bbox})
            frames_anno.append((file, frame_objects))
        return frames_anno

    def _prepare_anno_seqs(self,
                           anno_root: str,
                           seq_length: int,
                           seq_stride: int = 1) -> List[dict]:
        result = []
        anno_files = sorted(glob(os.path.join(anno_root, "*.json")))
        for anno_file in anno_files:
            frames_data = self._parse_anno(anno_file)
            while seq_length * seq_stride <= len(frames_data):
                anno_seq, frames_data = frames_data[:seq_length:seq_stride], frames_data[seq_length * seq_stride:]
                result.append(anno_seq)
        return result

    def __len__(self):
        return len(self.anno_seqs)

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
