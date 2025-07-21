import os
import numpy as np
import torch
from einops import rearrange

from src.utils import util


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
    fig_prefix = f"../weather-caiyun/lianghl/pretrain_results/exp{cfg.exp}/eval"
    os.makedirs(fig_prefix, exist_ok=True)
    for k, v in scores["surf"].items():
        final_score["surf"][k] = [np.sqrt(v[1]/v[0]/cfg.hyper_params.batch_size/241/281),  # RMSE
                                  v[2]/np.sqrt(v[3]*v[4])]                        # ACC
        util.plot_score(final_score["surf"][k][0], list(range(1,len(final_score["surf"][k][0])+1)), "forecast time(hs)", "RMSE", k, f"{fig_prefix}/exp{cfg.exp}_{k}_rmse.png")
        util.plot_score(final_score["surf"][k][1], list(range(1,len(final_score["surf"][k][1])+1)), "forecast time(hs)", "ACC", k, f"{fig_prefix}/exp{cfg.exp}_{k}_acc.png")
    
    for k, v in scores["high"].items():
        final_score["high"][k] = [np.sqrt(v[1]/v[0]/cfg.hyper_params.batch_size/241/281),  # RMSE
                                  v[2]/np.sqrt(v[3]*v[4])]                        # ACC
        util.plot_score(final_score["high"][k][0], list(range(1,len(final_score["high"][k][0])+1)), "forecast time(hs)", "RMSE", k, f"{fig_prefix}/exp{cfg.exp}_{k}_rmse.png")
        util.plot_score(final_score["high"][k][1], list(range(1,len(final_score["high"][k][1])+1)), "forecast time(hs)", "ACC", k, f"{fig_prefix}/exp{cfg.exp}_{k}_acc.png")
    return final_score
        
