import time
import torch
import logging
from typing import Optional
from .datasets import build_dataset
from core.data.transforms import build_transforms
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
                  pin_memory: bool = True) -> DataLoader:
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(time.time()))
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)

    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory)

    return data_loader


def make_data_loader(cfg, is_train: bool = True) -> Optional[DataLoader]:
    logger = logging.getLogger('CORE')

    if is_train:
        data_paths = cfg.DATASET.TRAIN_DATA_PATHS
        anno_paths = cfg.DATASET.TRAIN_ANNO_PATHS
    else:
        data_paths = cfg.DATASET.VALID_DATA_PATHS
        anno_paths = cfg.DATASET.VALID_ANNO_PATHS

    # build transforms
    transforms = build_transforms(cfg, is_train)

    # create dataset
    datasets = []
    for data_path, anno_path in zip(data_paths, anno_paths):
        dataset = build_dataset(cfg, data_path, anno_path, transforms)
        logger.info(f"Loaded dataset from '{data_path}'. Size: {len(dataset)}")
        datasets.append(dataset)

    if not datasets:
        return None

    dataset = ConcatDataset(datasets)

    # create dataloader
    shuffle = is_train
    data_loader = create_loader(dataset, shuffle, cfg.SOLVER.BATCH_SIZE, cfg.DATA_LOADER.NUM_WORKERS, cfg.DATA_LOADER.PIN_MEMORY)

    return data_loader
