import torch
import argparse
import cv2 as cv
import numpy as np
from glob import glob
from core.config import cfg
from core.utils.checkpoint import CheckPointer
from core.utils.ops import non_max_suppression
from core.modeling.model.stad import build_stad
from core.data.transforms import build_transforms


def main() -> int:
    # parse arguments
    parser = argparse.ArgumentParser(description='Spatio Temporal Action Detection With PyTorch')
    parser.add_argument('-c', '--cfg', dest='config_file', required=False, type=str, metavar="FILE",
                        default="configs/cfg.yaml",
                        help="path to config file")
    parser.add_argument('-i', '--input', dest='input', required=False, type=str, metavar="FILE",
                        # default="/media/yaroslav/SSD/khutornoy/data/sim_videos/olvia/04-09-2024/SKAT_12-54-02.mp4",
                        # default="/media/yaroslav/SSD/khutornoy/data/sim_videos/olvia/04-09-2024/SKAT_09-40-17.mp4",
                        # default="/media/yaroslav/SSD/khutornoy/data/sim_videos/olvia/04-09-2024/SKAT_09-40-17.mp4",
                        # default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/videos/ufa1/ufa1.mkv",
                        # default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/mp4/uid_vid_00035.mp4",
                        # default="/media/yaroslav/SSD/khutornoy/data/ImageNet/data/ImageNet2017/object_detection_from_video/ILSVRC2017_VID_new/ILSVRC/Data/VID/snippets/test/ILSVRC2017_test_00000000.mp4",
                        # default="/media/yaroslav/Terra/datasets/UCF101/ucf_test_videos/Basketball/v_Basketball_g07_c01.mp4",
                        # default="/media/yaroslav/Terra/datasets/UCF101/ucf_test_videos/BasketballDunk/v_BasketballDunk_g01_c01.mp4",
                        # default="/media/yaroslav/Terra/datasets/UCF101/ucf_test_videos/IceDancing/v_IceDancing_g05_c01.mp4",
                        # default="/media/yaroslav/Terra/datasets/UCF101/ucf_test_videos/WalkingWithDog/*.mp4",
                        # default="/media/yaroslav/Terra/datasets/UCF101/ucf_test_videos/Biking/*.mp4",
                        # default="/media/yaroslav/Terra/datasets/UCF101/ucf_test_videos/LongJump/*.mp4",
                        # default="/media/yaroslav/Terra/datasets/UCF101/ucf_test_videos/Surfing/*.mp4",
                        # default="/media/yaroslav/Terra/datasets/UCF101/ucf_test_videos/FloorGymnastics/*.mp4",
                        # default="/media/yaroslav/Terra/datasets/UCF101/ucf_test_videos/TennisSwing/*.mp4",
                        default="/media/yaroslav/Terra/datasets/UCF101/ucf_test_videos/Basketball/*.mp4",
                        help="path to input image")
    args = parser.parse_args()

    # create config
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # create device
    device = torch.device(cfg.MODEL.DEVICE)

    # create model
    model = build_stad(cfg)
    model = model.to(device)
    model.eval()

    # load weights
    checkpointer = CheckPointer(model, None, None, cfg.OUTPUT_DIR)
    checkpointer.load(cfg.MODEL.PRETRAINED_WEIGHTS)

    # transforms
    transforms = build_transforms(cfg, False)

    # processs input
    inputs = sorted(glob(args.input))
    for input in inputs:
        print(f"Processing {input} ...")

        cap = cv.VideoCapture(input)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return -1

        frame_cnt = 0
        image_buffer = []
        while cap.isOpened():
            ret, image = cap.read()

            if not ret:
                break
            if image is None:
                print("Failed to read frame")
                break
            frame_cnt += 1

            if len(image_buffer) >= cfg.DATASET.SEQUENCE_LENGTH:
                image_buffer.pop(0)
            image_buffer.append(image)
            if len(image_buffer) < cfg.DATASET.SEQUENCE_LENGTH:
                continue

            detects = []
            with torch.no_grad():
                data = {
                    'img': np.stack(image_buffer, 0)# .copy() # (T, H, W, C)
                }
                np_keyframe = data['img'][-1, :, :, :].copy()

                data = transforms(data) # (T, C, H, W)

                inputs = data['img'].unsqueeze(0) # (B, T, C, H, W)
                clip = inputs.permute(0, 2, 1, 3, 4) # (B, C, T, H, W)
                keyframe = clip[:, :, -1] # (B, C, H, W)
                INPUT_IMG_SIZE = clip.shape[-1:-3:-1]

                out_y, _ = model(clip.to(device), keyframe.to(device))

                # out_y = out_y.permute(0, 2, 1)
                # outputs = Detect.postprocess(out_y, max_det=300, nc=cfg.MODEL.HEAD.NUM_CLASSES)
                # outputs = outputs.squeeze(0)

                # TODP: multi_label=True ?
                # outputs = non_max_suppression(out_y, conf_thres=0.0005, iou_thres=0.45, nc=cfg.MODEL.HEAD.NUM_CLASSES)[0]
                outputs = non_max_suppression(out_y, conf_thres=0.005, iou_thres=0.45)[0]
                detects = outputs.to('cpu').numpy()

            for det in detects:
                x, y, w, h, conf, class_idx = det

                img_h, img_w  = np_keyframe.shape[:2]
                x = (x / INPUT_IMG_SIZE[0]) * img_w
                y = (y / INPUT_IMG_SIZE[1]) * img_h
                w = (w / INPUT_IMG_SIZE[0]) * img_w
                h = (h / INPUT_IMG_SIZE[1]) * img_h
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                if conf > 0.2:
                    color = (0, 255, 0)
                    cv.rectangle(np_keyframe, (x, y), (w, h), color, 2) # NMS
                    # cv.rectangle(image, (x - w//2, y-h//2), (x + w//2, y + h//2), (0, 255, 0), 1) # Detect.postprocess
                    text = f"{int(class_idx)} {conf:.2f}"
                    cv.putText(np_keyframe, text, (x, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            resize_k = 800.0 / np_keyframe.shape[1]
            cv.imshow('Result', cv.resize(np_keyframe, dsize=None, fx=resize_k, fy=resize_k, interpolation=cv.INTER_AREA))
            key = cv.waitKey(10) & 0xFF
            if key == ord(' '):
                break
            if key == ord('q'):
                return

    print("Done.")
    return 0


if __name__ == '__main__':
    exit(main())
