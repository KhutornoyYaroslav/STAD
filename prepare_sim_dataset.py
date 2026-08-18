import os
import json
import shutil
import logging
import argparse
import cv2 as cv
from glob import glob
from core.utils.logger import setup_logger


_LOGGER_NAME = "DATASET PREP"


def get_object_class(class_title: str, tags_dict: dict) -> int:
    """
    0 - person, walking
    1 - person, scootering
    2 - person, biking
    3 - person, carrying_scooter
    4 - person, carrying_bicycle
    5 - scooter
    6 - bicycle
    7 - undefined
    """
    logger = logging.getLogger(_LOGGER_NAME)

    class_title = class_title.lower()

    if class_title == "person":
        if "undefined" in tags_dict:
            return 7
            
        if "action" in tags_dict:
            tag = tags_dict["action"]
            if tag == "walking":
                return 0
            elif tag == "scootering":
                return 1
            elif tag == "biking":
                return 2
            elif tag == "carrying_scooter":
                return 3
            elif tag == "carrying_bike":
                return 4
            else:
                logger.warning(f"Found invalid action tag: '{tag}'")
                return None
        else:
            logger.warning("Action tag not found. Use default - 'walking'.")
            return 0
            
    elif class_title == "scooter":
        return 5
    elif class_title == "bicycle":
        return 6
    else:
        logger.warning(f"Found invalid class title: '{class_title}'")
        return None


def parse_anno_objects(data: dict):
    objects = {}
    for obj in data.get("objects", []):
        objects[obj["key"]] = {
            "classTitle": obj["classTitle"],
            # сохр список тегов объекта вместе с frame range
            "tags": obj.get("tags", [])
        }
    return objects


def convert_dataset(anno_path: str,
                    video_path: str,
                    dst_root: str,
                    filename_template = "%08d"):
    logger = logging.getLogger(_LOGGER_NAME)

    annos = sorted(glob(anno_path))
    videos = sorted(glob(video_path))

    for anno, video in zip(annos, videos):
        basename = os.path.basename(video)
        assert basename in anno
        logger.info(f"Processing flie: {anno}")

        # prepare output dir
        video_name = os.path.splitext(basename)[0]
        dst_dir = os.path.join(dst_root, video_name)
        shutil.rmtree(dst_dir, True)
        os.makedirs(dst_dir, exist_ok=False)

        # parse anno objects
        with open(anno, 'r') as f:
            anno_data = json.load(f)
        objects = parse_anno_objects(anno_data)
        anno_frames = anno_data.get("frames", [])

        # read video frame by frame
        frame_cnt = 0
        cap = cv.VideoCapture(video)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret != True:
                break

            dst_image_file = os.path.join(dst_dir, filename_template % frame_cnt + ".png")
            dst_label_file = os.path.join(dst_dir, filename_template % frame_cnt + ".txt")

            # save anno
            anno_empty = True
            for frame_info in anno_frames:
                if frame_info["index"] != frame_cnt:
                    continue
                for figure in frame_info.get("figures", []):
                    obj_key = figure["objectKey"]
                    
                    obj_info = objects.get(obj_key)
                    if obj_info is None:
                        continue
                        
                    class_title = obj_info["classTitle"]
                    
                    # фильтруем теги объекта для текущего кадра 
                    active_tags = {}
                    for t in obj_info["tags"]:
                        frame_range = t.get("frameRange")
                        if frame_range is not None:
                            start, end = frame_range
                            # если кадр не попадает в отрезок тега то пропускаем 
                            if not (start <= frame_cnt <= end):
                                continue 
                                
                        # сохраняем если он активен (или если он глобальный)
                        active_tags[t["name"].lower()] = t.get("value")
                    
                    # сразу вычисляем класс на основе активных тегов
                    obj_cls = get_object_class(class_title, active_tags)

                    if obj_cls is not None:
                        x1y1, x2y2 = figure["geometry"]["points"]["exterior"]
                        with open(dst_label_file, 'a') as f:
                            f.write("%d %d %d %d %d\n" % (obj_cls, *x1y1, *x2y2))
                        anno_empty = False
                    # print(obj_key, obj_cls)
                    # assert obj_cls is not None, "Expected obj_cls is not none"
                    

            # save image
            if not anno_empty:           
                cv.imwrite(dst_image_file, frame)
            else:
                # logger.warning(f"Found frame without annotations. Frame index: {frame_cnt}. Save empty annotation file.")
                # open(dst_label_file, 'a').close()
                logger.warning(f"Found frame without annotations. Frame index: {frame_cnt}. Skip file.")

            # update frame counter
            frame_cnt += 1


def str2bool(s):
    return s.lower() in ('true', '1')


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Supervisely To Internal Format Dataset Convertor')
    parser.add_argument('--anno-path', dest='anno_path', type=str,
                        default="/media/yaroslav/SSD/khutornoy/data/sim_videos/datasets/tatarina/*/*/ann/*.json",
                        help="Pattern-like path to supervisely annotation files")
    parser.add_argument('--video-path', dest='video_path', type=str,
                        default="/media/yaroslav/SSD/khutornoy/data/sim_videos/datasets/tatarina/*/*/video/*.mp4",
                        help="Pattern-like path to video files")
    parser.add_argument('--dst-root', dest='dst_root', type=str, default="/media/yaroslav/SSD/khutornoy/data/sim_videos/outputs/dataset_1/PADv2/",
                        help="Path to save result dataset")
    args = parser.parse_args()

    # create logger
    logger = setup_logger(_LOGGER_NAME, distributed_rank=0)
    logger.info(args)

    # convert dataset
    convert_dataset(args.anno_path, args.video_path, args.dst_root)


if __name__ == "__main__":
    main()
