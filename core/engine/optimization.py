import torch
import logging
from core.config import CfgNode


def make_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    logger = logging.getLogger('CORE')

    params_to_train = []
    params_names_to_freeze = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            params_to_train.append(p)
        else:
            params_names_to_freeze.append(n)

    if len(params_names_to_freeze):
        logger.info("No gradient update for following model parameters:\n\n{}\n".format(
            "\n".join(params_names_to_freeze)))

    lr = float(cfg.SOLVER.LR)
    return torch.optim.Adam(params=params_to_train, lr=lr)
