import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
import numpy as np

from utils import BLoss, scheduler_warmup


def get_optimizer(model, cfg):
    if isinstance(cfg.hyper_params.lr, str):
        cfg.hyper_params.lr = eval(cfg.hyper_params.lr)
    if cfg.hyper_params.optimizer=="AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=cfg.hyper_params.lr, 
                                    weight_decay=cfg.hyper_params.weight_decay, 
                                    betas=(0.9, 0.95)
                                    )
    elif cfg.hyper_params.optimizer=="Adam":
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=cfg.hyper_params.lr, 
                                    weight_decay=cfg.hyper_params.weight_decay, 
                                    betas=(0.9, 0.95)
                                    )
    elif cfg.hyper_params.optimizer=="rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), 
                                        lr=0.01, 
                                        alpha=0.99, 
                                        eps=1e-08, 
                                        weight_decay=0
                                        )
    else:
        print("optim not supported")
        return
    return optimizer
    

def get_scheduler(optimizer, cfg):
    if cfg.hyper_params.scheduler=="reducelronplateau":
        scheduler = ReduceLROnPlateau(optimizer, threshold=1e-8, patience=4, factor=0.5, verbose=True)
    elif cfg.hyper_params.scheduler=="cosineannealing":
        scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0, last_epoch=-1, verbose=True)
    elif cfg.hyper_params.scheduler=="cosineannealingwarmrestart":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.hyper_params.EPOCH, T_mult=2, verbose=True)
    
    if cfg.hyper_params.warmup_lr:
        scheduler = scheduler_warmup.WarmupLR(scheduler, init_lr=1e-9, num_warmup=1000, warmup_strategy="linear")
    return scheduler
        

LOSSES = {
        'mse': nn.MSELoss(reduction='mean'),
        'bmse': BLoss.BMSELoss(),
        'ssim': BLoss.SSIM(window_size=11, size_average=True),
        'mse_iou': BLoss.MSE_IOULoss(),
        'bmse_prcp': BLoss.BMSELoss(weights=[1, 1.1, 2.5, 10, 100], thresholds=[3, 5, 7, 9, 11]),
        'bmse_radar': BLoss.BMSELoss(weights=[1, 6.5, 20, 60, 300], thresholds=[2, 5, 10, 20, 30]),
        'bce_mse': BLoss.BCE_MSELoss(), 
        #'lat_weighted_mse': BLoss.LatWeightedMSE(latweights(), device=0)
        }
