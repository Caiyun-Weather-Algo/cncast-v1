import os
import numpy as np
import torch
from omegaconf import OmegaConf  
import sys
sys.setrecursionlimit(15000)

from src.utils import util


def plot_subfigure(scores, x, ax, xlabel, ylabel, title, text):
    """Plot scores for a single variable"""
    months = list(range(1, 13))
    steps = [1,12,24,48]
    # for i, month in enumerate(months):
    scores = np.array(scores)
    for k,step in enumerate(steps):
        ax.plot(x, scores[:,k], linewidth=1, label=f"Lead {step}hr")
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.grid(linestyle="--", color="gray", alpha=0.7)
    if "ACC" in ylabel:
        ax.legend(loc="lower left", fontsize=8)
    else:
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([1, x[-1]])
    plt.text(0.02, 0.98, text, transform=ax.transAxes, 
            fontsize=12, fontweight="bold", va="top", ha="left")
    return


def load_era5(start, end, levels, cfg):
    step = 6
    surf_vars = ["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind"]
    high_vars = ["geopotential", "temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"]
    surf_var_abbr = ["t2m", "msl", "u10", "v10"]
    high_var_abbr = ["z", "t", "q", "u", "v"]
    # Initialize empty lists to store data for each variable
    surf_data = {var: [] for var in surf_vars}
    high_data = {var: [] for var in high_vars}
    
    # Loop through each day
    for ts in range(start, end, 3600*24):
        t = arrow.get(ts).format("YYYYMMDD")
        # Process surface variables
        # ts2sel = [arrow.get(ts_).format("YYYY-MM-DDTHH:00:00.000000000") for ts_ in range(ts+3600, ts+3600*24, 3600*6)]
        surf_vars_data = []
        for k, var in enumerate(surf_vars):
            f = f"{cfg.directory.home_path}/{cfg.buckets.bucket_era5}/202407/2024/era5.{var}.{t}.nc"
            ds = xr.open_dataset(f)

            # Take every 6th hour (0, 6, 12, 18)
            data = ds[surf_var_abbr[k]].isel(time=[0,6,12,18])
            surf_vars_data.append(data)
            
        # Process upper air variables
        high_vars_data = []
        for k, var in enumerate(high_vars):
            f = f"{cfg.directory.home_path}/{cfg.buckets.bucket_era5}/202407/2024/era5.{var}.{t}.nc"
            ds = xr.open_dataset(f)
            # Take every 6th hour (0, 6, 12, 18)
            data = ds[high_var_abbr[k]].isel(time=[0,6,12,18], level=levels)
            high_vars_data.append(data)
        
        # Concatenate variables for this time step
        # Create DataArrays with variable coordinates
        surf_vars_data = [da.assign_coords(variable=var) for da, var in zip(surf_vars_data, surf_vars)]
        high_vars_data = [da.assign_coords(variable=var) for da, var in zip(high_vars_data, high_vars)]
        
        surf_data['all_vars'] = surf_data.get('all_vars', []) + [xr.concat(surf_vars_data, dim='variable')]
        high_data['all_vars'] = high_data.get('all_vars', []) + [xr.concat(high_vars_data, dim='variable')]
    
    # Concatenate all time steps
    surf_ds = xr.concat(surf_data['all_vars'], dim='time').transpose('time', 'variable', 'latitude', 'longitude')
    high_ds = xr.concat(high_data['all_vars'], dim='time').transpose('time', 'variable', 'level', 'latitude', 'longitude')
    
    return surf_ds, high_ds


def mid_rmse(pred, y, scores, cfg, lat_weights, avgs):
    pred_surf, pred_high = pred
    y_surf, y_high = y
    surf_avg, high_avg = avgs
    for k, var in enumerate(cfg.output.surface):
        name = f"{var}_{k}"
        #print(name, y_surf[:,:,k].max(), y_surf[:,:,k].min(), pred_surf[:,:,k].max(), pred_surf[:,:,k].min(), surf_avg[k:k+1])
        s1 = np.sum((y_surf[:,:,k] - pred_surf[:,:,k])**2 * lat_weights, axis=(0,2,3))
        s2 = np.sum((y_surf[:,:,k]-surf_avg[k:k+1]) * (pred_surf[:,:,k]-surf_avg[k:k+1])*lat_weights, axis=(0,2,3))
        s3 = np.sum((y_surf[:,:,k]-surf_avg[k:k+1])**2 *lat_weights, axis=(0,2,3)) 
        s4 = np.sum((pred_surf[:,:,k]-surf_avg[k:k+1])**2 *lat_weights, axis=(0,2,3))
        if scores["surf"].get(name, None) is not None:
            scores["surf"][name] += np.array([1, s1, s2, s3, s4], dtype=object)
        else:
            scores["surf"][name] = np.array([1, s1, s2, s3, s4], dtype=object)
    print("mid rmse surf")
            
    for k, var in enumerate(cfg.output.high):
        for j,lev in enumerate(cfg.input.levels):
            name = f"{var}_{lev}"
            #print(name, y_high[:,:,k,j].max(), y_high[:,:,k,j].min(), pred_high[:,:,k,j].max(), pred_high[:,:,k,j].min(), high_avg[k:k+1,j:j+1])
            s1 = np.sum((y_high[:,:,k,j] - pred_high[:,:,k,j])**2 * lat_weights, axis=(0,2,3))
            s2 = np.sum((y_high[:,:,k,j]-high_avg[k:k+1,j:j+1]) * (pred_high[:,:,k,j]-high_avg[k:k+1,j:j+1]) * lat_weights, axis=(0,2,3))
            s3 = np.sum((y_high[:,:,k,j] - high_avg[k:k+1,j:j+1])**2 *lat_weights, axis=(0,2,3)) 
            s4 = np.sum((pred_high[:,:,k,j] - high_avg[k:k+1,j:j+1])**2 *lat_weights, axis=(0,2,3))
            
            if scores["high"].get(name, None) is not None:
                scores["high"][name] += np.array([1, s1, s2, s3, s4], dtype=object)
            else:
                scores["high"][name] = np.array([1, s1, s2, s3, s4], dtype=object)
    print("mid rmse high")
    
    return scores


def final_rmse(scores, cfg):
    final_score = {"surf":{}, "high":{}}
    # cnt = xm.xrt_world_size()
    for k, v in scores["surf"].items():
        final_score["surf"][k] = [np.sqrt(v[1]/v[0]/cfg.hyper_params.batch_size/241/281),  # RMSE
                                  v[2]/np.sqrt(v[3]*v[4])]                        # ACC
    
    for k, v in scores["high"].items():
        final_score["high"][k] = [np.sqrt(v[1]/v[0]/cfg.hyper_params.batch_size/241/281),  # RMSE
                                  v[2]/np.sqrt(v[3]*v[4])]                        # ACC
    return final_score
        
