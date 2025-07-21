import torch
import torch_xla.core.xla_model as xm


def get_dataset(mode, cfg):
    from src.data.datasets import ERA5Dataset

    dataset = ERA5Dataset(mode=mode, 
                input_vars=cfg.input, 
                output_vars=cfg.output, 
                sample_interval=cfg.hps.sample_interval, 
                input_hist_hr=cfg.hps.hist4in,
                forecast_step=cfg.hps.fcst4out, 
                use_dem=cfg.hps.use_dem, 
                norm_method=cfg.hps.norm_method,
                fcst_tp=cfg.fcst_tp, 
                start_time=cfg.dataload.start_time, 
                end_time=cfg.dataload.end_time, 
                test_start_time=cfg.dataload.test_start_time, 
                valid_start_time=cfg.dataload.valid_start_time, 
                resize_data=cfg.dataload.resize_data, 
                )
    return dataset


def get_sampler(dataset, mode="train"):
    """
    Create a sampler for the dataset in TPU environment.
    
    Args:
        dataset: The dataset to sample from
        
    Returns:
        torch.utils.data.distributed.DistributedSampler
    """
    from torch.utils.data.distributed import DistributedSampler

    shuffle = mode=="train"
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=shuffle)


def get_loader(dataset, cfg, mode="train"):
    sampler = get_sampler(dataset, mode=mode)
    loader = torch.utils.data.DataLoader(dataset, 
                                    batch_size=cfg.hps.batch_size,
                                    pin_memory=False,
                                    drop_last=False,
                                    num_workers=2, 
                                    sampler=sampler, 
                                    )
    return loader

