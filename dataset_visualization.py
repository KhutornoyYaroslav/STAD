import argparse
from core.config import cfg
from core.data.datasets import build_dataset
from core.data.transforms import build_transforms


def str2bool(s):
    return s.lower() in ('true', '1')


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Spatio Temporal Action Detection Dataset Visualization')
    parser.add_argument("-d", "--data-path", dest="data_path", required=False, type=str,
                        default="/media/yaroslav/SSD/khutornoy/data/UCF101_24/rgb-images")
    parser.add_argument("-a", "--anno-path", dest="anno_path", required=False, type=str,
                        default="/media/yaroslav/SSD/khutornoy/data/UCF101_24/labels_train")
    parser.add_argument("-t", "--istrain", dest="istrain", required=False, type=str2bool,
                        default=True)
    parser.add_argument("-n", "--dataloader-name", dest="dataloader_name", required=False, type=str,
                        default='UCF101_24Dataset')
    parser.add_argument("-r", "--frame-rate", dest="frame_rate", required=False, type=int,
                        default=25)
    parser.add_argument("-l", "--seq-length", dest="seq_length", required=False, type=int,
                        default=32)
    args = parser.parse_args()

    # set config
    cfg.DATASET.SEQUENCE_DILATE = 1
    cfg.DATASET.SEQUENCE_LENGTH = args.seq_length
    cfg.DATASET.SEQUENCE_STRIDE = args.seq_length
    cfg.DATASET.TYPE = args.dataloader_name
    cfg.freeze()

    # check dataset
    transforms = build_transforms(cfg, is_train=args.istrain)
    dataset = build_dataset(cfg, args.data_path, args.anno_path, transforms)
    print(f"Dataset size: {len(dataset)} sequences")
    dataset.visualize(args.frame_rate)


if __name__ == '__main__':
    main()
