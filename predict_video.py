import os
import torch
import argparse
import cv2 as cv
import numpy as np
from glob import glob
from core.config import cfg
from core.modeling import build_model
from core.utils.logger import setup_logger
from core.utils.checkpoint import CheckPointer
from core.utils.ops import non_max_suppression
from core.data.transforms import build_transforms, PadResize


def main() -> int:
    # parse arguments
    parser = argparse.ArgumentParser(description='Spatio Temporal Action Detection With PyTorch')
    parser.add_argument('-c', '--cfg', dest='config_file', required=False, type=str,
                        default="/home/yaroslav/repos/STAD/outputs/yolov8s_dataset_1/cfg.yaml",
                        help="Path to configuration file")
    parser.add_argument('-i', '--input', dest='input', required=False, type=str,
                        # default="/media/yaroslav/Новый том/skatpp_clean/на_распределение_цв/*",
                        # default="/media/yaroslav/Новый том/skatpp_clean/сложные/15278877581604258-camera_data_12_trim_crop.avi",
                        # default="/media/yaroslav/Terra/datasets/video_practice/vika/15247737201305756-camera_data_0034.avi",
                        # default="/media/yaroslav/Terra/datasets/video_practice/16_06_26/DJI_20260616142412_0132_D.MP4",
                        # default="/media/yaroslav/Terra/datasets/video_practice/18_06_26/DJI_20260618140056_0229_D.MP4",
                        # default="/media/yaroslav/Terra/datasets/video_practice/01_07_26/DJI_20260701145934_0073_D.MP4",
                        # default="/media/yaroslav/SSD/khutornoy/data/sim_videos/videos/olvia/dybenko_street/chemusov/done/SKAT_12-25-32.mp4",
                        # default="/media/yaroslav/SSD/khutornoy/data/sim_videos/datasets/2026/ulia_pav/PAD/PADD/video/2024_09_09__10_48_38_250_cut_1.mp4",
                        # default="/media/yaroslav/SSD/khutornoy/data/sim_videos/datasets/2026/anndre_eva/PAD/PADD/video/2024_09_09__15_26_02_878_cut.mp4",
                        # default="/media/yaroslav/SSD/khutornoy/data/sim_videos/datasets/2026/anndre_eva/PAD/PADD/video/2024_09_11__15_05_30_511_cut.mp4",
                        # default="/media/yaroslav/SSD/khutornoy/data/sim_videos/datasets/2026/butterfly_catastrophe/PAD/PADD/video/2024_09_10__12_31_29_327_cut_5.mp4",
                        default="/media/yaroslav/SSD/khutornoy/data/sim_videos/videos/vika/team_1/IMG_1561.MOV",
                        # default="/media/yaroslav/SSD/khutornoy/data/sim_videos/videos/youtube/Lakhta Center-HMW4sYtEtU8-4.mp4",
                        # default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/videos/kolo2/kolo2.mkv",
                        # default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/videos/kbr1.mkv",
                        # default="/media/yaroslav/SSD/khutornoy/data/VIDEOS/videos/poly1.mkv",
                        help="Path to input video(s)")
    parser.add_argument('-o', '--output-dir', dest='output_dir', required=False, type=str,
                        default="/media/yaroslav/SSD/khutornoy/data/sim_videos/predictions",
                        help="Directory to write output videos with predictions")
    parser.add_argument('-conf', '--conf', dest='conf', required=False, type=float,
                        default=0.5,
                        help="Confidence threshold")
    parser.add_argument('--win-size', dest='win_size', required=False, type=int,
                        default=1200,
                        help="Size of debug window (width)")
    parser.add_argument("--frame-rate", dest="frame_rate", required=False, type=int,
                        default=0)
    parser.add_argument("--frame-step", dest="frame_step", required=False, type=int,
                        default=1)
    args = parser.parse_args()

    # create logger
    logger = setup_logger("CORE", 0)
    logger.info(args)

    # create config
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # create device
    device = torch.device(cfg.MODEL.DEVICE)

    # create model
    model = build_model(cfg)
    model = model.to(device)
    model = model.eval()
    model = model.half()
    # model.prepare_inference(*cfg.INPUT.IMAGE_SIZE)

    # load weights
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    checkpointer.load()

    # transforms
    transforms = build_transforms(cfg, False)
    inv_resize_transforms = PadResize(cfg.INPUT.IMAGE_SIZE, make_divisible_by=cfg.INPUT.MAKE_DIVISIBLE_BY)

    # process inputs
    for video in sorted(glob(args.input)):
        logger.info(f"Processing video: '{video}'")

        # create video reader
        reader = cv.VideoCapture(video)
        if not reader.isOpened():
            logger.error("Failed to open video stream")
            return -1

        # create video writer
        video_name = os.path.basename(video)
        output_video_path = os.path.join(args.output_dir, video_name)
        logger.info(f"Writing result to: '{output_video_path}'")

        frame_width  = int(reader.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(reader.get(cv.CAP_PROP_FRAME_HEIGHT))
        video_fps = reader.get(cv.CAP_PROP_FPS)
        writer = cv.VideoWriter(
            output_video_path, 
            cv.VideoWriter_fourcc(*'mp4v'), 
            video_fps,
            (frame_width, frame_height))

        # process frames
        frame_idx = 0
        frame_step = max(1, args.frame_step)

        while reader.isOpened():
            reader.set(cv.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = reader.read()
            if not ret: break
            if frame is None:
                logger.error("Failed to read frame")
                break

            # infer model
            with torch.no_grad():
                data = {
                    'img': np.expand_dims(frame.copy(), 0) # (1, H, W, C)
                }

                data = transforms(data)
                input = data['img'].unsqueeze(0) # (B, 1, C, H, W)
                input = input.half()
                out_y, _ = model(input.to(device))

                outputs_person = non_max_suppression(out_y, conf_thres=args.conf, iou_thres=0.45, classes=[0, 1, 2, 3, 4, 7], agnostic=True, in_place=False)[0]
                outputs_vehicle = non_max_suppression(out_y, conf_thres=args.conf, iou_thres=0.45, classes=[5, 6], in_place=False)[0]
                outputs = torch.cat([outputs_person, outputs_vehicle], dim=0)
                detects = outputs.to('cpu').numpy() # (N, 6)

            # visualize
            mask = np.zeros_like(frame)
            img_h, img_w = frame.shape[:2]
            detects[:, :4] = inv_resize_transforms._inv_apply_bbox(detects[:, :4], img_w, img_h)

            for det in detects:
                x1, y1, x2, y2, conf, class_idx = det
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                color = (200, 200, 255)

                if int(class_idx) == 7:
                    color = (0, 0, 255)

                text_label = cfg.DATASET.LABELS[int(class_idx)]

                cv.rectangle(mask, (x1, y1), (x2, y2), color, -1) # NMS
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 1) # NMS
                text = f"{text_label} {conf:.2f}"
                cv.putText(frame, text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            frame = cv.addWeighted(frame, 0.8, mask, 0.2, 0)
            writer.write(frame)

            resize_k = args.win_size / frame.shape[1]
            cv.imshow('Result', cv.resize(frame, dsize=None, fx=resize_k, fy=resize_k, interpolation=cv.INTER_AREA))
            key = cv.waitKey(args.frame_rate) & 0xFF
            if key == ord('q'):
                writer.release()
                return
            elif key == 81:
                frame_idx = frame_idx - frame_step
            else:
                frame_idx = frame_idx + frame_step

        reader.release()
        writer.release()

    logger.info("Done")
    return 0


if __name__ == '__main__':
    exit(main())
