import torch
from torch import nn
import logging
from core.config import CfgNode


def make_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    logger = logging.getLogger('CORE')

    lr = float(cfg.SOLVER.LR)
    wd = float(cfg.SOLVER.WEIGHT_DECAY)

    # filter out parameters w/o gradient update
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
        
    # split to parameter groups
    g = ([], [], [])
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    for m in model.modules():
        for n, p in m.named_parameters(recurse=0):
            # bias (no decay)
            if n == "bias":
                g[2].append(p)
            # weight (no decay)
            elif n == "weight" and isinstance(m, bn):
                g[1].append(p)
            # weight (with decay)
            else:
                g[0].append(p)

    # create optimizer
    optimizer = torch.optim.AdamW(g[0], lr=lr, weight_decay=wd)
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    optimizer.add_param_group({"params": g[2], "weight_decay": 0.0})

    return optimizer
