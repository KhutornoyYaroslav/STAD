import os
import torch
import logging
import argparse
from torch import nn
from typing import Any
from core.utils import dist_util
from core.config import cfg, CfgNode
from core.data import make_data_loader
from core.engine.training import do_train
from core.utils.logger import setup_logger
from core.utils.checkpoint import CheckPointer
from core.modeling.model.stad import build_stad
from core.engine.optimization import make_optimizer


def train_model(cfg: CfgNode, args: Any) -> nn.Module:
    logger = logging.getLogger('CORE')
    device = torch.device(cfg.MODEL.DEVICE)

    # create model
    model = build_stad(cfg)
    model.to(device)

    # create data loader
    data_loader_train = make_data_loader(cfg, is_train=True)
    if data_loader_train is None:
        logger.error(f"Failed to create train dataset loader.")
        return None
    data_loader_val = make_data_loader(cfg, is_train=False)

    # create optimizer
    optimizer = make_optimizer(cfg, model)
    scheduler = None

    # create checkpointer
    arguments = {"epoch": 0}
    save_to_disk = dist_util.is_main_process()
    checkpointer = CheckPointer(model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_WEIGHTS)
    arguments.update(extra_checkpoint_data)

    # train model
    model = do_train(cfg,
                     model,
                     data_loader_train,
                     data_loader_val,
                     optimizer,
                     scheduler,
                     checkpointer,
                     device,
                     arguments,
                     args)

    return model


def str2bool(s):
    return s.lower() in ('true', '1')


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Spatio Temporal Action Detection Model Training')
    parser.add_argument("-c", "--config-file", dest="config_file", required=False, type=str,
                        default="configs/cfg.yaml",
                        help="Path to config file")
    parser.add_argument("-s", "--save-step", dest="save_step", required=False, type=int,
                        default=1,
                        help='Save checkpoint every save_step')
    parser.add_argument("-e", "--eval-step", dest="eval_step", required=False, type=int,
                        default=1,
                        help='Evaluate datasets every eval_step, disabled when eval_step < 0')
    parser.add_argument("-t", "--use-tensorboard", dest="use_tensorboard", required=False, type=str2bool,
                        default=True,
                        help='Use tensorboard summary writer')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    NUM_GPUS = 1
    args.distributed = False
    args.num_gpus = NUM_GPUS

    # enable cudnn auto-tuner to find the best algorithm to use for your hardware
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # create config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # create logger
    logger = setup_logger("CORE", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info(args)
    logger.info("Loaded configuration file '{}':\n\n{}\n".format(args.config_file, cfg))

    # create config backup
    with open(os.path.join(cfg.OUTPUT_DIR, 'cfg.yaml'), "w") as cfg_dump:
        ret = cfg_dump.write(str(cfg))

    # train model
    model = train_model(cfg, args)


if __name__ == '__main__':
    main()
