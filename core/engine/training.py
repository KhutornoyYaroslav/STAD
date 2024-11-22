import os
import torch
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Optional, Any
from core.config import CfgNode
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from core.utils.checkpoint import CheckPointer
from torch.optim.lr_scheduler import LRScheduler
from core.engine.loss import DetectionLoss
from core.utils.tensorboard import add_metrics
from core.engine.validation import do_validation
from torch.utils.tensorboard import SummaryWriter
from core.utils.tensorboard import select_samples
from core.data.transforms.transforms import Denormalize, ToNumpy, ToTensor, Compose


def do_train(cfg: CfgNode,
             model: nn.Module,
             data_loader_train: DataLoader,
             data_loader_val: Optional[DataLoader],
             optimizer: Optimizer,
             scheduler: Optional[LRScheduler],
             checkpointer: CheckPointer,
             device: torch.device,
             arguments: dict,
             args: Any):
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    logger = logging.getLogger("CORE")
    logger.info("Start training")

    # set model to train mode
    model.train()

    # create tensorboard writer
    if args.use_tensorboard:
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    # tensorboard image transforms
    tb_img_transforms = [
        ToNumpy(),
        Denormalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_SCALE),
        ToTensor()
    ]
    tb_img_transforms = Compose(tb_img_transforms)

    # prepare to train
    iters_per_epoch = len(data_loader_train)
    start_epoch = arguments["epoch"]
    end_epoch = cfg.SOLVER.MAX_EPOCH
    total_steps = iters_per_epoch * cfg.SOLVER.MAX_EPOCH
    logger.info("Iterations per epoch: {0}. Total steps: {1}. Start epoch: {2}".format(iters_per_epoch, total_steps, start_epoch + 1))

    # create metrics
    det_loss = DetectionLoss(num_classes=model.get_num_classes(),
                             strides=model.get_strides(),
                             dfl_bins=model.get_dfl_num_bins(),
                             loss_box_k=cfg.SOLVER.LOSS_BOX_WEIGHT,
                             loss_dfl_k=cfg.SOLVER.LOSS_DFL_WEIGHT,
                             loss_cls_k=cfg.SOLVER.LOSS_CLS_WEIGHT,
                             device=device,
                             tal_topk=cfg.SOLVER.TAL_TOPK)

    # epoch loop
    for epoch in range(start_epoch, end_epoch):
        arguments["epoch"] = epoch + 1

        # create progress bar
        print(('\n' + '%10s' * 7) % ('epoch', 'gpu_mem', 'lr', 'loss', 'loss_box', 'loss_dfl', 'loss_cls'))
        pbar = enumerate(data_loader_train)
        pbar = tqdm(pbar, total=len(data_loader_train))

        # create stats
        stats = {
            'loss_sum': 0,
            'loss_box_sum': 0,
            'loss_dfl_sum': 0,
            'loss_cls_sum': 0,
            'random_samples' : []
        }

        # iteration loop
        for iteration, data_entry in pbar:
            global_step = epoch * iters_per_epoch + iteration

            # get data
            images = data_entry["img"]                              # (B, T, C, H, W)
            bboxes = data_entry["bbox"]                              # (B, T, max_targets, 4)
            classes = data_entry["cls"]                             # (B, T, max_targets, num_classes)

            cur_image = images[:, -1]                               # (B, C, H, W)
            cur_bboxes = bboxes[:, -1]                              # (B, max_targets, 4)
            cur_classes = classes[:, -1]                            # (B, max_targets, num_classes)
            cur_targets = torch.cat([cur_bboxes, cur_classes], -1)  # (B, max_targets, 4 + num_classes)

            cur_image = cur_image.to(device)
            cur_targets = cur_targets.to(device)

            # forward model
            output_y, output_x = model(cur_image)                   # 3 x (B, C, Hi, Wi)
            output_y = output_y.permute(0, 2, 1)

            # calculate loss
            losses = det_loss(output_x, cur_targets)
            loss = losses[0]
            loss_box, loss_cls, loss_dfl = losses[1]

            # optimize model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # select random samples to tensorboard
            select_samples(limit=cfg.TENSORBOARD.BEST_SAMPLES_NUM,
                           accumulator=stats['random_samples'],
                           image=cur_image.detach(),
                           targets=cur_targets.detach(),
                           preds=output_y.detach(),
                           metric=torch.rand(cur_image.shape[0], device=device),
                           conf_thresh=cfg.TENSORBOARD.CONF_THRESH,
                           min_metric_better=False,
                           image_transforms=tb_img_transforms)

            # update stats
            stats['loss_sum'] += loss.item()
            stats['loss_box_sum'] += loss_box.item()
            stats['loss_dfl_sum'] += loss_dfl.item()
            stats['loss_cls_sum'] += loss_cls.item()

            # update progress bar
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' * 2 + '%10.4g' * 5) % ('%g/%g' % (epoch + 1, end_epoch),
                                               mem,
                                               optimizer.param_groups[0]["lr"],
                                               stats['loss_sum'] / (iteration + 1),
                                               stats['loss_box_sum'] / (iteration + 1),
                                               stats['loss_dfl_sum'] / (iteration + 1),
                                               stats['loss_cls_sum'] / (iteration + 1)
            )
            pbar.set_description(s)

        # update learning rate
        if scheduler is not None:
            scheduler.step()

        # do validation
        if (args.val_step > 0) and (epoch % args.val_step == 0) and (data_loader_val is not None):
            print('\n')
            logger.info("Start validation ...")

            torch.cuda.empty_cache()
            model.eval()
            val_stats = do_validation(cfg, model, data_loader_val, device)
            torch.cuda.empty_cache()
            model.train()

            val_loss = val_stats['loss_sum'] / val_stats['iterations']
            val_loss_box = val_stats['loss_box_sum'] / val_stats['iterations']
            val_loss_dfl = val_stats['loss_dfl_sum'] / val_stats['iterations']
            val_loss_cls = val_stats['loss_cls_sum'] / val_stats['iterations']

            log_preamb = 'Validation results: '
            print((log_preamb + '%10s' * 6) % ('loss', 'loss_box', 'loss_dfl', 'loss_cls', 'map_50', 'map_75'))
            print((len(log_preamb) * ' ' + '%10.4g' * 6) % (val_loss,
                                                            val_loss_box,
                                                            val_loss_dfl,
                                                            val_loss_cls,
                                                            val_stats['map_50'],
                                                            val_stats['map_75']))
            print('\n')

            if summary_writer:
                scalars = {
                    'loss': val_loss,
                    'loss_box': val_loss_box,
                    'loss_dfl': val_loss_dfl,
                    'loss_cls': val_loss_cls,
                    'map_50': val_stats['map_50'],
                    'map_75': val_stats['map_75']
                }
                samples = {
                    'best_samples': val_stats['best_samples'],
                    'worst_samples': val_stats['worst_samples']
                }
                add_metrics(summary_writer, scalars, samples, global_step, False)

        # save epoch results
        if epoch % args.save_step == 0:
            checkpointer.save("model_{:06d}".format(global_step), **arguments)
            if summary_writer:
                scalars = {
                    'loss': stats['loss_sum'] / (iteration + 1),
                    'loss_box': stats['loss_box_sum'] / (iteration + 1),
                    'loss_dfl': stats['loss_dfl_sum'] / (iteration + 1),
                    'loss_cls': stats['loss_cls_sum'] / (iteration + 1),
                    'lr': optimizer.param_groups[0]["lr"]
                }
                samples = {
                    'random_samples': stats['random_samples'],
                }
                add_metrics(summary_writer, scalars, samples, global_step, True)

    # save final model
    checkpointer.save("model_final", **arguments)

    return model
