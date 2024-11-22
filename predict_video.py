# import os
import torch
# from torch import nn
# import logging
import argparse
import cv2 as cv
import numpy as np
# from tqdm import tqdm
# from core.config import cfg, skeleton
# from core.utils.logger import setup_logger
# from core.utils import dist_util
# from core.modelling.model import build_model
# from core.utils.checkpoint import CheckPointer
from core.config import cfg
from core.modeling.head.yolov8 import Detect
from core.utils.ops import non_max_suppression
from core.modeling.model.stad import build_stad
from core.data.transforms import build_transforms
from core.utils.checkpoint import CheckPointer


def main() -> int:
    # parse arguments
    parser = argparse.ArgumentParser(description='Spatio Temporal Action Detection With PyTorch')
    parser.add_argument('-c', '--cfg', dest='config_file', required=False, type=str, metavar="FILE",
                        default="configs/cfg.yaml",
                        help="path to config file")
    parser.add_argument('-i', '--input-video', dest='input_video', required=False, type=str, metavar="FILE",
                        default="/media/yaroslav/SSD/khutornoy/data/sim_videos/olvia/04-09-2024/SKAT_12-54-02.mp4",
                        # default="/media/yaroslav/SSD/khutornoy/data/sim_videos/olvia/04-09-2024/SKAT_09-40-17.mp4",
                        # default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/videos/ufa1/ufa1.mkv",
                        # default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/mp4/uid_vid_00035.mp4",
                        # default="/media/yaroslav/SSD/khutornoy/data/ImageNet/data/ImageNet2017/object_detection_from_video/ILSVRC2017_VID_new/ILSVRC/Data/VID/snippets/test/ILSVRC2017_test_00000000.mp4",
                        help="path to input image")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()

    # create config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # # Create logger
    # logger = setup_logger("INFER", dist_util.get_rank(), cfg.OUTPUT_DIR)
    # logger.info(args)
    # logger.info("Loaded configuration file {}".format(args.config_file))

    # create device
    device = torch.device(cfg.MODEL.DEVICE)

    # create model
    model = build_stad(cfg)
    model = model.to(device)

    # load weights
    checkpointer = CheckPointer(model, None, None, cfg.OUTPUT_DIR)
    checkpointer.load(cfg.MODEL.PRETRAINED_WEIGHTS)

    model.eval()

    # read input
    cap = cv.VideoCapture(args.input_video)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return -1

    transforms = build_transforms(cfg, False)

    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        if image is None:
            # logger.error("Failed to read frame")
            print("Failed to read frame")
            break

        # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # image = np.stack([image, image, image], -1)

        detects = []
        with torch.no_grad():
            data = {
                "img": np.expand_dims(image.copy(), 0)
            }
            data = transforms(data)

            inputs = data["img"]
            INPUT_IMG_SIZE = inputs.shape[-1:-3:-1]
            out_y, _ = model(inputs.to(device))

            # out_y = out_y.permute(0, 2, 1)
            # outputs = Detect.postprocess(out_y, max_det=300, nc=cfg.MODEL.HEAD.NUM_CLASSES)
            # outputs = outputs.squeeze(0)

            # TODP: multi_label=True ?
            outputs = non_max_suppression(out_y, conf_thres=0.25, iou_thres=0.45, nc=cfg.MODEL.HEAD.NUM_CLASSES)[0]
            detects = outputs.to('cpu').numpy()

        # allowed_classes = [0] # , 1, 3, 36]

        for det in detects:
            if len(det) > 6:
                print(det)
            x, y, w, h, conf, class_idx = det

            # if class_idx not in allowed_classes:
            #     continue

            img_h, img_w  = image.shape[:2]
            x = (x / INPUT_IMG_SIZE[0]) * img_w
            y = (y / INPUT_IMG_SIZE[1]) * img_h
            w = (w / INPUT_IMG_SIZE[0]) * img_w
            h = (h / INPUT_IMG_SIZE[1]) * img_h
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            if conf > 0.5:
                color = (0, 255, 0) # if class_idx != 1 else (0, 0, 255)
                cv.rectangle(image, (x, y), (w, h), color, 2) # NMS
                # cv.rectangle(image, (x - w//2, y-h//2), (x + w//2, y + h//2), (0, 255, 0), 1) # Detect.postprocess

        resize_k = 1400.0 / image.shape[1]
        # resize_k = 1.0
        cv.imshow('Result', cv.resize(image, dsize=None, fx=resize_k, fy=resize_k, interpolation=cv.INTER_AREA))
        if cv.waitKey(0) & 0xFF == ord('q'):
            break

    print("Done.")
    return 0


if __name__ == '__main__':
    exit(main())
