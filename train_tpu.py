"""
Training script for weather prediction model on TPU devices.

This script handles the training, validation, and validing of a weather prediction model
using TPU acceleration. It supports distributed training, checkpointing, and evaluation.
"""

# Standard libraries
import os
import time
import yaml
import json
import argparse
from typing import OrderedDict
from pathlib import Path
from functools import partial

# Data handling
import numpy as np
from imageio import imread, imsave
from omegaconf import OmegaConf, DictConfig
from einops import rearrange

# PyTorch and XLA
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# TPU-specific imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.distributed.xla_backend
import torch_xla.debug.profiler as xp
import torch_xla.runtime as xr

# Local imports
from src.utils import util
from src.utils import helpers
from src.models.create_model import load_model
from src.eval.eval_utils import mid_rmse
from src.models.model_utils import save_model, load_training_ckpt
from src.data.data_utils import get_dataset, get_loader

# Parse command line arguments
parser = argparse.ArgumentParser(description="Weather prediction model training on TPU")
parser.add_argument("--config", type=str, default="./configs/train_config.yaml", help="model and hyperparameter settings")
parser.add_argument("--batch_size", type=int, default=2, help="batch size for training")
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
parser.add_argument("--fcst4out", type=int, default=1, help="forecast hour")
parser.add_argument("--exp", type=int, default=1, help="experiment identifier")



def adjust_lr(model, cfg):
    """
    Creates an optimizer with different learning rates for different parameter groups.
    
    Base parameters use the standard learning rate, while boundary condition (bc_)
    parameters use a 10x higher learning rate.
    
    Args:
        model: The model with parameters to optimize
        cfg: Configuration object with hyperparameters
        
    Returns:
        AdamW optimizer with configured parameter groups
    """
    base_params = [p for name, p in model.named_parameters() if "bc_" not in name]
    bc_params = [p for name, p in model.named_parameters() if "bc_" in name]
    optimizer = torch.optim.AdamW([
        {'params': base_params},
        {'params': bc_params, 'lr': cfg.hps.lr * 10},
    ],
    lr=cfg.hps.lr,
    betas=(0.9, 0.999)
    )
    return optimizer


def train_loop(train_loader, model, loss_fn, optimizer, scheduler, writer, cfg, epoch, loss_method, loss_weight):
    """
    Training loop for one epoch.
    
    Args:
        train_loader: DataLoader for training data
        model: Model to train
        loss_fn: Loss function
        optimizer: Optimizer for model parameters
        scheduler: Learning rate scheduler
        writer: TensorBoard writer
        cfg: Configuration object
        epoch: Current epoch number
        loss_method: Method to calculate loss ("mse", "gaussin", etc.)
        loss_weight: Whether to use weighted loss
        
    Returns:
        train_loss: Average training loss for the epoch
        scheduler: Updated scheduler
    """
    train_loss = 0
    iters = len(train_loader)
    t0 = time.time()
    model.train()
    
    for i, data in enumerate(train_loader):
        xm.master_print(f"epoch {epoch}, train loop {i}, lr={optimizer.param_groups[0]['lr']}")
        with xp.StepTrace('Training_step', step_num=i):
            # Extract data
            _, xin, yout, xbc = data
                
            # Forward pass
            output = model(xin, xbc)                
            # Calculate loss
            loss = loss_fn(output, yout, loss_method, loss_weight)

            # Handle gradient accumulation
            if cfg.hps.grad_accum > 1:
                loss = loss / cfg.hps.grad_accum

            train_loss += loss.item()
            loss.backward()
            
            # Update weights based on accumulation schedule
            if i % cfg.hps.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            # Step scheduler if using cosine annealing
            if cfg.hps.scheduler in ["cosineannealingwarmrestart", "cosineannealing"]:
                scheduler.step(epoch + i / iters)
                
            writer.add_scalars('train loop loss', {'train loop loss': loss.item()}, epoch*iters+i+1)
        
    # Calculate average loss across all processes
    train_loss = train_loss / len(train_loader)
    train_loss = xm.mesh_reduce("train loss", train_loss, np.mean)
    
    if xm.is_master_ordinal():
        writer.add_scalars('train loss', {'train loss': train_loss}, epoch)
        
    xm.master_print(f"epoch {epoch}, train loss = {train_loss}, one train loop costs {time.time()-t0}s")
    return train_loss, scheduler


def valid_loop(valid_loader, model, loss_fn, cfg, img_path, era5_valid_loader, epoch, writer, loss_method, loss_weight, scheduler):
    """
    Validation loop for one epoch.
    
    Args:
        valid_loader: DataLoader for validation data
        model: Model to evaluate
        loss_fn: Loss function
        cfg: Configuration object
        img_path: Path to save visualization images
        era5_valid_loader: ERA5 valid data loader for normalization
        epoch: Current epoch number
        writer: TensorBoard writer
        loss_method: Method to calculate loss
        loss_weight: Whether to use weighted loss
        scheduler: Learning rate scheduler
        
    Returns:
        valid_loss: Average validation loss for the epoch
        scheduler: Updated scheduler
    """
    model.eval()
    valid_loss = 0
    start = time.time()
    
    for i, data in enumerate(valid_loader):
        with xp.StepTrace('validing_step', step_num=i):
            # Extract data
            _, xin, yout, xbc = data
            x_surf, x_high = xin[:,:5], rearrange(xin[:,5:], 'b (v l) h w -> b v l h w')
            
            # Forward pass (no gradients)
            with torch.no_grad():
                output = model(xin, xbc)
                loss = loss_fn(output, yout, "mse", loss_weight)
            
            valid_loss += loss.item()
            xm.master_print(f"valid loss={loss.item()}")
            
        # Visualization (only on specific epochs and intervals)
        if i % cfg.hps.verbose_step == 0 and (epoch-1) % 2 == 0 and xm.is_master_ordinal():
            if loss_method == "gaussin":
                xm.master_print(i, 'target and output:', y_surf.max(), y_high.max(), output[0][0].max(), output[0][1].max())
            else:
                xm.master_print(i, 'target and output:', y_surf.max(), y_high.max(), output[0].max(), output[1].max(), output[0].shape, output[1].shape)
            
            # Create directory for visualizations
            t0 = time.time()
            img_ipath = img_path + f'/epoch{epoch}'
            os.makedirs(img_ipath, exist_ok=True)
            xm.master_print("dir made: ", img_ipath)
            
            device = xm.xla_device()
            if loss_method == "gaussin":
                output = output[0]  # output[0] are mean values, this might need revision
            
            # Process and save visualizations for input, target, and output
            for d, ve in zip(([x_surf, x_high], [y_surf, y_high], output), ("input", "target", "output")):
                d = [d[0].to("cpu").numpy(), d[1].to("cpu").numpy()]
                if ve == "input":
                    x_surf, x_high = era5_valid_loader.resume([d[0][:,:-1], d[1]], cfg)
                else:
                    x_surf, x_high = era5_valid_loader.resume(d, cfg)
                
                x_surf_ = x_surf[:,0]
                x_high_ = x_high[:,0]
                
                # Normalize data for visualization
                x_surf_ = np.concatenate([era5_valid_loader.norm(x_surf_[:, k:k+1], "surface", 
                                     era5_valid_loader.input_vars["surface"][k], v=None, 
                                     norm_method="minmax") for k in range(x_surf_.shape[1])], axis=1)
                
                for j, v_ in enumerate(era5_valid_loader.input_lev_idxs):
                    x_high_[:,:,j] = np.concatenate([era5_valid_loader.norm(x_high_[:, k:k+1, j], "high", 
                                              era5_valid_loader.input_vars["high"][k], v=v_, 
                                              norm_method="minmax") for k in range(x_high_.shape[1])], axis=1)
                
                x_surf_ = x_surf_[0]
                x_high_ = x_high_[0]
                x_high_ = rearrange(x_high_, 'v l h w -> (v l) h w')
                
                # Save images
                try:
                    for j, x in enumerate(x_surf_):
                        imsave('{}/{}_{}_step{}_{}.png'.format(img_ipath, i, j, cfg.hps.fcst4out, ve), 
                              (x*255).astype(np.uint8))
                        n = j
                    for j, x in enumerate(x_high_):
                        imsave('{}/{}_{}_step{}_{}.png'.format(img_ipath, i, j+n+1, cfg.hps.fcst4out, ve), 
                              (x*255).astype(np.uint8))
                except Exception as e:
                    print(e)
                    continue
    
    # Calculate average validation loss
    xm.master_print("valid loader length:", len(valid_loader))
    valid_loss = valid_loss / len(valid_loader)
    valid_loss = xm.mesh_reduce("valid loss", valid_loss, np.mean)
    
    if xm.is_master_ordinal():
        writer.add_scalars("valid loss", {"valid loss": valid_loss}, epoch)
    
    # Update scheduler if using ReduceLROnPlateau
    if cfg.hps.scheduler == "reducelronplateau":
        scheduler.step(valid_loss)
        
    xm.master_print(f"epoch {epoch}, valid loss={valid_loss}, one valid loop costs {time.time()-start}s")
    return valid_loss, scheduler


def test_loop(test_loader, model, cfg, static, loss_method, test_dataset):
    """
    test loop for evaluating model performance.
    
    Args:
        test_loader: DataLoader for test data
        model: Model to evaluate
        cfg: Configuration object
        static: Tuple containing (lat_weights, surf_avg, high_avg)
        loss_method: Method used for loss calculation
        
    Returns:
        final_score: Dictionary containing evaluation metrics
    """
    scores = {"surf": {}, "high": {}}
    model.eval()
    start = time.time()
    lat_weights, surf_avg, high_avg = static
    
    for i, data in enumerate(test_loader):
        with xp.StepTrace('testing_step', step_num=i):
            # Extract data
            _, xin, yout, xbc = data
            y_surf, y_high = yout[:,:5], rearrange(yout[:,5:], 'b (v l) h w -> b v l h w')
            
            # Forward pass (no gradients)
            with torch.no_grad():
                output = model(xin, xbc)
            
            # Process outputs for evaluation
            y = (y_surf.cpu().numpy(), y_high.cpu().numpy())
            if loss_method == "gaussin":
                output = output[0]
            output = (output[0].cpu().numpy(), output[1].cpu().numpy())
            
            # Restore original scale
            output = test_dataset.resume(output, cfg)
            y = test_dataset.resume(y, cfg)
            
            xm.master_print(output[0].shape, output[1].shape, y[0].shape, y[1].shape)
            
            # Calculate evaluation metrics
            scores = mid_rmse(output, y, scores, cfg, lat_weights, (surf_avg, high_avg))
    
    # Aggregate scores across processes
    final_score = {"surf": {}, "high": {}}
    
    for k, vv in scores["surf"].items():
        v = np.array([xm.mesh_reduce("mid_rmse", v_, np.sum) for v_ in vv])
        final_score["surf"][k] = [
            np.sqrt(v[1] / v[0] / cfg.hps.batch_size / 241 / 281),  # RMSE
            v[2] / np.sqrt(v[3] * v[4])                                      # ACC
        ]

    for k, vv in scores["high"].items():
        v = np.array([xm.mesh_reduce("mid_rmse", v_, np.sum) for v_ in vv])
        final_score["high"][k] = [
            np.sqrt(v[1] / v[0] / cfg.hps.batch_size / 241 / 281),  # RMSE
            v[2] / np.sqrt(v[3] * v[4])                                      # ACC
        ]
   
    return final_score


def ddp_func(cfg):
    """
    Main training function for distributed data parallel training on TPUs.
    
    Sets up the model, dataloaders, and runs the training, validation, and validing loops.
    
    Args:
        cfg: Configuration object containing all parameters
    """
    # Initialize distributed training
    device = xm.xla_device()
    dist.init_process_group('xla', init_method='xla://')
    xm.master_print("training configs: ", cfg)
    
    # Setup paths for logs, checkpoints, and visualizations
    base_path = cfg.base_path
    writer = SummaryWriter(log_dir=f"{base_path}/runs/")
    
    img_path = os.path.abspath(f"{base_path}/images")
    model_path = os.path.abspath(f"{base_path}/models")
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    # Initialize latitude weights for weighted metrics
    lat_weights = util.lat_weight()
    lat_weights = torch.tensor(lat_weights, dtype=torch.float).to(device)

    # Start profiler server
    server = xp.start_server(9012)
    
    # Initialize model
    model = load_model(cfg.model.name, cfg)
    xm.master_print(model)
    
    # Load checkpoint if available
    ckpt, start_epoch = load_training_ckpt(f"{model_path}/swin_transformer_3d_epoch")
    if ckpt is not None:
        model.load_state_dict(ckpt["model"], strict=False)
        
    # Move model to device and wrap with DDP
    model = model.to(device)
    model = DDP(model, gradient_as_bucket_view=True)
    
    # Setup optimizer and scheduler
    optimizer = helpers.get_optimizer(model, cfg)  
    optimizer = adjust_lr(model, cfg)
    scheduler = helpers.get_scheduler(optimizer, cfg)  
    
    # Load optimizer and scheduler state if resuming
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # Define loss function
    mse = torch.nn.MSELoss(reduction="mean")
    def loss_fn(pred, y, method="mse", weighting=False):
        if method == "mse":
            loss = mse(pred, y)
        
        elif method == "gaussin":  # pred=[[mu_surf, mu_high], [std_surf, std_high]]
            epsilon = 1e-8
            # Gaussian negative log-likelihood loss
            loss = torch.mean(torch.log(pred[1] + epsilon) + 
                             0.5 * ((y - pred[0]) / (pred[1] + epsilon))**2) 
        
        elif method == "latitude-weighted mse":
            # Apply latitude weights to MSE
            loss = mse(pred * lat_weights, y * lat_weights) 
        return loss
    
    loss_method = cfg.loss_method
    loss_weighting = cfg.loss_weighting

    # Initialize dataloaders
    era5_train_dataset = get_dataset(mode="train", cfg=cfg)  
    era5_valid_dataset = get_dataset(mode="valid", cfg=cfg)
    train_loader = get_loader(era5_train_dataset, cfg, mode="train")
    valid_loader = get_loader(era5_valid_dataset, cfg, mode="valid")
        
    # Wrap with TPU device loaders
    train_device_loader = pl.MpDeviceLoader(
        train_loader, 
        device, 
        loader_prefetch_size=cfg.hps.batch_size*2, 
        device_prefetch_size=cfg.hps.batch_size,
    )
    
    valid_device_loader = pl.MpDeviceLoader(
        valid_loader, 
        device, 
        loader_prefetch_size=cfg.hps.batch_size*2, 
        device_prefetch_size=cfg.hps.batch_size
    )
    
    xm.master_print("dataloader done!")
    
    # Initialize early stopping parameters
    least_loss = None
    PATIENCE = 30 
    patience = PATIENCE 
    iters = len(train_loader)
    grad_accum_step = cfg.hps.grad_accum
    
    # Begin training loop
    xm.master_print("----------- start training ------------")
    for epoch in range(start_epoch, cfg.hps.EPOCH + 1):
        t0 = time.time()
        cfg.ckpt_prefix = f"{model_path}/swin_transformer_3d_epoch{epoch}"
        xm.master_print("epoch", epoch)
        
        # Training phase
        train_loss, scheduler = train_loop(
            train_device_loader, model, loss_fn, optimizer, 
            scheduler, writer, cfg, epoch, loss_method, loss_weighting
        )
        
        # Validation phase
        valid_loss, scheduler = valid_loop(
            valid_device_loader, model, loss_fn, cfg, img_path, 
            era5_valid_dataset, epoch, writer, loss_method, loss_weighting, scheduler
        )

        # Early stopping logic
        if least_loss is None:
            least_loss = valid_loss
        if least_loss < valid_loss:
            patience -= 1
            xm.master_print('valid loss stop decreasing!!!', 'Patience is %d' % patience)
        else:
            least_loss = valid_loss
            patience = PATIENCE 
            if xm.is_master_ordinal():
                save_model(epoch, model, optimizer, scheduler, train_loss, valid_loss, cfg.ckpt_prefix, best=True)
        
        # Regular checkpointing
        if epoch % 1 == 0 and xm.is_master_ordinal():
            save_model(epoch, model, optimizer, scheduler, train_loss, valid_loss, cfg.ckpt_prefix)
        
        # Check for early stopping
        if patience <= 0:
            xm.master_print('Early stop!!!')
            break

    # Evaluation phase
    xm.master_print("----------- start evaluation ------------")
    surf_avg, high_avg = util.static4eval(era5_valid_loader)
    lat_weights = lat_weights.detach().cpu().numpy()
        
    # Load best model for evaluation
    model = load_model(cfg.model.name, cfg)
    ckpt = torch.load(f"{base_path}/models/swin_transformer_3d_consolidated_best.pth.tar", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    
    era5_test_dataset = get_dataset(mode="valid", cfg=cfg)
    era5_test_loader = get_loader(era5_test_dataset, cfg, mode="valid")
    test_device_loader = pl.MpDeviceLoader(
        era5_test_loader, 
        device, 
        loader_prefetch_size=cfg.hps.batch_size*2, 
        device_prefetch_size=cfg.hps.batch_size,
    )
    
    # Run evaluation
    static = (lat_weights, surf_avg, high_avg)
    final_score = test_loop(test_device_loader, model, cfg, static, loss_method, era5_test_dataset)
    print(final_score)
    
    # Save evaluation results
    savepath = Path(f"{base_path}/eval")
    savepath.mkdir(exist_ok=True, parents=True)
    fscore = savepath / f"era5_score_exp{cfg.exp}_mean.npz"
    
    if xm.is_master_ordinal:
        np.savez_compressed(
            fscore, 
            surf=np.array(final_score["surf"], dtype=object), 
            high=np.array(final_score["high"], dtype=object)
        )

    return


def main(device):
    """
    Main entry point for TPU training.
    
    Args:
        device: TPU device
    """
    torch.set_default_tensor_type('torch.FloatTensor')  # Set default tensor type
    
    # Parse arguments and load configuration
    args = parser.parse_args()
    cfg = yaml.load(open(args.config), Loader=yaml.Loader)
    cfg = DictConfig(cfg)
    cfg.hps.batch_size = args.batch_size
    cfg.exp = args.exp
    
    # Set random seed for reproducibility
    torch.manual_seed(2023)
    
    # Run distributed training
    ddp_func(cfg)
    return


if __name__ == '__main__':
    # Spawn multiple processes for TPU training
    xmp.spawn(main, args=(), nprocs=None)
