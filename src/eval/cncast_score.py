import xarray as xr
import numpy as np
import os, time
from datetime import datetime
import arrow, hydra
from omegaconf import DictConfig

from evaluation import mid_rmse
from utils import util


def load_cncast(t):
    f = f"/home/lianghongli/weather-caiyun/lianghl/pretrain_results/pred_recurse_bc/tuv_pred/pred_china_{t}_5d.zarr"
    surf = xr.open_zarr(f, group="surf", consolidated=True)["2mt"].to_numpy()
    upper = xr.open_zarr(f, group="upper", consolidated=True)["geopotential"].to_numpy()
    return surf, upper


def load_era5(start_yr, end_yr, bucket_central1="weather-us-central1"):
    surf_data = xr.open_dataset(f"gcs://{bucket_central1}/era5_china_src/surface_1980-2022.zarr", 
                                        backend_kwargs={"storage_options": {"project": "colorful-aia", "token": None}},
                                        engine="zarr",)
    surf_data = surf_data.sel(time=slice(datetime(start_yr,1,1,0), datetime(end_yr+1,1,1,0)))
    high_data = [xr.open_dataset(f"gcs://{bucket_central1}/era5_china_src/high_{yr}.zarr", 
                                    backend_kwargs={"storage_options": {"project": "colorful-aia", "token": None}},
                                    engine="zarr",) for yr in range(start_yr, end_yr+1)]
    # high_925 = [xr.open_dataset(f"gcs://{bucket_central1}/era5_china_src/high_{yr}_925-50hPa.zarr", 
    #                                 backend_kwargs={"storage_options": {"project": "colorful-aia", "token": None}},
    #                                 engine="zarr",) for yr in range(start_yr, end_yr+1)]
    return surf_data, high_data


def timestamp_to_datetime(ts):
    """Convert timestamp to datetime object"""
    return datetime.fromtimestamp(ts)


def final_rmse(scores, cfg):
    final_score = {"surf":{}, "high":{}}
    # cnt = xm.xrt_world_size()
    for k, v in scores["surf"].items():
        final_score["surf"][k] = [np.sqrt(v[1]/v[0]/241/281),  # RMSE
                                  v[2]/np.sqrt(v[3]*v[4])]                        # ACC
    
    for k, v in scores["high"].items():
        final_score["high"][k] = [np.sqrt(v[1]/v[0]/241/281),  # RMSE
                                  v[2]/np.sqrt(v[3]*v[4])]                        # ACC
    return final_score

@hydra.main(version_base=None, config_path="./configs", config_name="online.yaml")
def main(cfg):
    bucket_central1 = cfg.buckets.bucket_base
    lat_weights = util.lat_weight()[None,None,:]
    variables = util.reset_era5_stat()
    era5_levels = variables["levels"]
    levels = [1000, 950, 850, 700, 600, 500,450,400,300,250,200,150,100]
    lev_idxs = [era5_levels.index(lev) for lev in levels]
    avgs = util.static4eval(lev_idxs)
    era5_surf, era5_upper = load_era5(2021, 2021, bucket_central1)
    era5_upper = era5_upper[0]

    start = "2021070100"
    end = "2021080100"
    for month in range(7, 8):
        start = f"2021{month:02d}0100"
        if month == 12:
            end = f"2021122600"
        else:
            end = f"2021{month+1:02d}0100"
        start_ts = int(arrow.get(start, "YYYYMMDDHH").timestamp())
        end_ts = int(arrow.get(end, "YYYYMMDDHH").timestamp())
        scores = {"surf":{}, "high":{}}
        rmse_srf = 0
        acc_srf = 0
        rmse_high = 0
        acc_high = 0
        damaged = 0
        for k, ts in enumerate(range(start_ts, end_ts, 3600)):
            t0 = time.time()
            t = arrow.get(ts).format("YYYYMMDDHH")
            # Load data
            try:
                cncast_surf, cncast_upper = load_cncast(t)
                cncast_surf = cncast_surf[None,:]
                cncast_upper = cncast_upper[None,:]
                lead1 = timestamp_to_datetime(ts+3600)
                lead120 = timestamp_to_datetime(ts+120*3600)
                y_surf = era5_surf.sel(time=slice(lead1, lead120))["2m_temperature"].to_numpy()[None,:]
                y_high = era5_upper.sel(time=slice(lead1, lead120))["geopotential"].to_numpy()[None,:]
                rmse_srf = util.RMSE(y_surf, cncast_surf, lat_weights) + rmse_srf
                acc_srf = util.ACC(y_surf, cncast_surf, avgs[0], lat_weights) + acc_srf
                rmse_high = util.RMSE(y_high, cncast_upper, lat_weights) + rmse_high
                acc_high = util.ACC(y_high, cncast_upper, avgs[1], lat_weights) + acc_high
                # print(t, rmse_srf, acc_srf, rmse_high, acc_high)
                # mid_rmse((cncast_surf, cncast_upper), [y_surf, y_high], scores, cfg, lat_weights, avgs)
                print("one update costs:",t, time.time()-t0)
                # break
            except:
                print(f"file at {t} damaged, skip!")
                damaged += 1
                continue

        final_score = final_rmse(scores, cfg)
        final_score = {"surf":{}, "high":{}}
        
        rmse_srf = np.squeeze(rmse_srf)
        acc_srf = np.squeeze(acc_srf)
        rmse_high = np.squeeze(rmse_high)
        acc_high = np.squeeze(acc_high)
        print(rmse_srf.shape, acc_srf.shape, rmse_high.shape, acc_high.shape)
        for i,var in enumerate(cfg.input.surface):
            final_score["surf"][var] = [rmse_srf[:,i]/(k+1-damaged), acc_srf[:,i]/(k+1-damaged)]
        for i,var in enumerate(cfg.input.high):
            final_score["high"][var] = [rmse_high[:,i]/(k+1-damaged), acc_high[:,i]/(k+1-damaged)]
        print(final_score)
        np.savez_compressed(f"/home/lianghongli/weather-caiyun/lianghl/pretrain_results/pred_recurse_bc/cncast-v1_scores/month{month}_scores_china_5d_averaged.npz", 
                            scores=final_score, mid_rmse=scores)


if __name__=="__main__":
    main()
