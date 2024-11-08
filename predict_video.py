# import os
import torch
# from torch import nn
# import logging
import argparse
import cv2 as cv
# import numpy as np
# from tqdm import tqdm
# from core.config import cfg, skeleton
# from core.utils.logger import setup_logger
# from core.utils import dist_util
# from core.modelling.model import build_model
# from core.utils.checkpoint import CheckPointer
from core.config import cfg
from core.modeling.head.yolov8 import Detect

from core.data.transforms.transforms import (
    ConvertFromInts,
    Normalize,
    Standardize,
    ToTensor,
    Resize,
    ConvertColor
)

from core.utils.ops import non_max_suppression
from core.modeling.model.stad import build_stad


def main() -> int:
    # parse arguments
    parser = argparse.ArgumentParser(description='Spatio Temporal Action Detection With PyTorch')
    parser.add_argument('-c', '--cfg', dest='config_file', required=False, type=str, metavar="FILE",
                        default="configs/cfg.yaml",
                        help="path to config file")
    parser.add_argument('-i', '--input-video', dest='input_video', required=False, type=str, metavar="FILE",
                        default="/media/yaroslav/SSD/khutornoy/data/sim_videos/olvia/04-09-2024/SKAT_12-25-32.mp4",
                        help="path to input image")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()

    # Create config
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
    # model = load_model(cfg, args)
    model = build_stad(cfg)
    model = model.to(device)
    model.eval()

    # read input
    cap = cv.VideoCapture(args.input_video)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return -1

    IMG_SIZE = (320, 160)

    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        if image is None:
            # logger.error("Failed to read frame")
            print("Failed to read frame")
            break

        detects = []
        with torch.no_grad():
            input, _, _ = Resize(IMG_SIZE)(image)
            input, _, _ = ConvertFromInts()(input)
            input, _, _ = Normalize()(input)
            # input, _, _ = Standardize()(input)
            input, _, _ = ConvertColor("BGR", "RGB")(input)
            input, _, _ = ToTensor()(input)

            out_y, _ = model(input.unsqueeze(0).to(device))

            # out_y = out_y.permute(0, 2, 1)
            # outputs = Detect.postprocess(out_y, max_det=300, nc=80)
            # outputs = outputs.squeeze(0)
            outputs = non_max_suppression(out_y, conf_thres=0.5, iou_thres=0.5)[0] # multi_label=True
            detects = outputs.to('cpu').numpy()

        for det in detects:
            if len(det) > 6:
                print(det)
            x, y, w, h, conf, class_idx = det

            img_h, img_w, _ = image.shape
            x = (x / IMG_SIZE[0]) * img_w
            y = (y / IMG_SIZE[1]) * img_h
            w = (w / IMG_SIZE[0]) * img_w
            h = (h / IMG_SIZE[1]) * img_h
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            if conf > 0.5:
                # cv.rectangle(image, (x - w//2, y-h//2), (x + w//2, y + h//2), (0, 255, 0), 1) # Detect.postprocess
                cv.rectangle(image, (x, y), (w, h), (0, 255, 0), 2) # NMS

        resize_k = 1400.0 / image.shape[1]
        # resize_k = 1.0
        cv.imshow('Result', cv.resize(image, dsize=None, fx=resize_k, fy=resize_k, interpolation=cv.INTER_AREA))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    print("Done.")
    return 0


if __name__ == '__main__':
    exit(main())
