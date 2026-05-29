import time
import torch
import logging
from typing import Optional
from .datasets import build_dataset
from core.data.transforms import build_transforms, build_inv_transforms
from torch.utils.data import (
    Dataset,
    DataLoader,
    BatchSampler,
    ConcatDataset,
    RandomSampler,
    SequentialSampler
)


def create_loader(dataset: Dataset,
                  shuffle: bool,
                  batch_size: int,
                  num_workers: int = 1,
                  pin_memory: bool = True,
                  prefetch_factor: int = 2) -> DataLoader:
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(time.time()))
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)

    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=True)
    data_loader = DataLoader(dataset,
                             batch_sampler=batch_sampler,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             prefetch_factor=prefetch_factor)

    return data_loader


def make_data_loader(cfg, is_train: bool = True) -> Optional[DataLoader]:
    logger = logging.getLogger('CORE')

    if is_train:
        data_paths = cfg.DATASET.TRAIN_DATA_PATHS
        anno_paths = cfg.DATASET.TRAIN_ANNO_PATHS
    else:
        data_paths = cfg.DATASET.VALID_DATA_PATHS
        anno_paths = cfg.DATASET.VALID_ANNO_PATHS

    ds_type_str = "train" if is_train else "valid"

    # build transforms
    transforms = build_transforms(cfg, is_train)
    inv_transforms = build_inv_transforms(cfg)

    # create dataset
    datasets = []
    for data_path, anno_path in zip(data_paths, anno_paths):
        dataset = build_dataset(cfg, data_path, anno_path, transforms, inv_transforms)
        logger.info(f"Loaded {ds_type_str} dataset from '{data_path}'. Size: {len(dataset)}")
        datasets.append(dataset)

    if not datasets:
        return None

    dataset = ConcatDataset(datasets)
    logger.info(f"Total {ds_type_str} dataset size: {len(dataset)}")

    # create dataloader
    shuffle = is_train
    data_loader = create_loader(
        dataset, 
        shuffle, 
        cfg.SOLVER.BATCH_SIZE, 
        cfg.DATA_LOADER.NUM_WORKERS, 
        cfg.DATA_LOADER.PIN_MEMORY,
        cfg.DATA_LOADER.PREFECTH_FACTOR)

    return data_loader
