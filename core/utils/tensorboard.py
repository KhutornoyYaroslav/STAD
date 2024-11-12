import torch
import numpy as np
from typing import Dict
from numbers import Number
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


def add_metrics(summary_writer: SummaryWriter,
                scalars: Dict[str, Number],
                global_step: int,
                is_train: bool = False):
    if summary_writer is None:
        return

    prefix = 'train' if is_train else 'val'

    for key, val in scalars.items():
        summary_writer.add_scalar(f'{prefix}/{key}', val, global_step=global_step)

    # with torch.no_grad():
    #     # Best samples
    #     if len(result_dict['best_samples'][0]):
    #         for i, l in enumerate(cfg.SOLVER.LAMBDAS):
    #             tb_images = [sample[1] for sample in result_dict['best_samples'][i]]
    #             image_grid = torch.stack(tb_images, dim=0)
    #             image_grid = make_grid(image_grid, nrow=1)
    #             summary_writer.add_image(f'images/{prefix}_best_samples_lambda_{i + 1}_{l}', image_grid,
    #                                      global_step=global_step)

    #     # Worst samples
    #     if len(result_dict['worst_samples'][0]):
    #         for i, l in enumerate(cfg.SOLVER.LAMBDAS):
    #             tb_images = [sample[1] for sample in result_dict['worst_samples'][i]]
    #             image_grid = torch.stack(tb_images, dim=0)
    #             image_grid = make_grid(image_grid, nrow=1)
    #             summary_writer.add_image(f'images/{prefix}_worst_samples_lambda_{i + 1}_{l}', image_grid,
    #                                      global_step=global_step)

    summary_writer.flush()
