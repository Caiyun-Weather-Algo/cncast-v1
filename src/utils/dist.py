import os
import torch.distributed as dist
from datetime import timedelta

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    # initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=timedelta(seconds=3600))
    return


def cleanup():
    dist.destroy_process_group()
    return