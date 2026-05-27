import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from matplotlib.axes import Axes
from tqdm import tqdm
from core.config import cfg
from core.data.datasets import build_dataset, Dataset
from typing import Sequence


def plot_hist(x: ArrayLike, 
              axes: Axes, 
              title: str = "", 
              bins: int = 32, 
              color: Sequence[float] = (0.9, 0.6, 0.6)):
    color = (0.9, 0.6, 0.6)
    probs, edges, _ = axes.hist(x, bins=bins, density=False,  color=color, alpha=0.5)
    axes.set_title(title)


def calc_stat(dataset: Dataset):
    bbox_all, cls_all = [], []

    # gather items
    for i in tqdm(range(0, dataset.__len__())):
        item = dataset.__getitem__(i)
        non_empty_mask = item['bbox'][..., 2] * item['bbox'][..., 3] > 0
        bbox_all.append(item['bbox'][non_empty_mask])
        cls_all.append(item['cls'][non_empty_mask])
    bbox_all = np.concatenate(bbox_all, axis=0)
    cls_all = np.concatenate(cls_all, axis=0)
    assert bbox_all.shape[0] == cls_all.shape[0]

    # create plot
    plot_size = 8
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(plot_size, plot_size))
    plt.tight_layout(pad=2.0)

    # one-hot class to label
    labels = np.argmax(cls_all, axis=-1)
    walking_mask = labels == 0
    person_mask = labels < 5

    # class
    labels_bins = np.arange(min(labels), max(labels) + 2) - 0.5
    plot_hist(labels, axes[0, 0], bins=labels_bins, title="class (all)")
    plot_hist(labels[~walking_mask], axes[0, 1], bins=labels_bins, title="class (w/o walking)")

    # bbox (person only)
    plot_hist(bbox_all[person_mask, 0], axes[1, 0], title="x (person)")
    plot_hist(bbox_all[person_mask, 1], axes[1, 1], title="y (person)")
    plot_hist(bbox_all[person_mask, 2], axes[2, 0], title="width (person)")
    plot_hist(bbox_all[person_mask, 3], axes[2, 1], title="height (person)")

    # bbox (non person)
    plot_hist(bbox_all[~person_mask, 0], axes[3, 0], title="x (vehicle)")
    plot_hist(bbox_all[~person_mask, 1], axes[3, 1], title="y (vehicle)")
    plot_hist(bbox_all[~person_mask, 2], axes[4, 0], title="width (vehicle)")
    plot_hist(bbox_all[~person_mask, 3], axes[4, 1], title="height (vehicle)")

    plt.show()


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Spatio Temporal Action Detection Dataset Statistics')
    parser.add_argument("-d", "--data-path", dest="data_path", required=False, type=str,
                        default="/media/yaroslav/SSD/khutornoy/data/sim_videos/outputs/2026/PADv1_split/train"
                        # default="/media/yaroslav/SSD/khutornoy/data/sim_videos/outputs/2026/PADv1/ppolique"
                        )
    parser.add_argument("-a", "--anno-path", dest="anno_path", required=False, type=str,
                        default="/media/yaroslav/SSD/khutornoy/data/sim_videos/outputs/2026/PADv1_split/train"
                        # default="/media/yaroslav/SSD/khutornoy/data/sim_videos/outputs/2026/PADv1/ppolique"
                        )
    parser.add_argument("-n", "--dataloader-name", dest="dataloader_name", required=False, type=str,
                        default='SIMDataset')
    args = parser.parse_args()

    # adjust config
    seq_len = 1
    cfg.DATASET.SEQUENCE_DILATE = 1
    cfg.DATASET.SEQUENCE_LENGTH = seq_len
    cfg.DATASET.SEQUENCE_STRIDE = seq_len
    cfg.INPUT.PAD_LABELS_TO = 256
    cfg.DATASET.TYPE = args.dataloader_name
    cfg.freeze()

    # create dataset
    dataset = build_dataset(cfg, args.data_path, args.anno_path, None)
    print(f"Dataset size: {len(dataset)} sequences")

    # process
    calc_stat(dataset)


if __name__ == '__main__':
    main()
