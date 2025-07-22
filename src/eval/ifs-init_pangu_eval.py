import numpy as np
import hydra
from glob import glob
from pathlib import Path
import arrow
import xarray as xr
import matplotlib.pyplot as plt
from typing import OrderedDict
from datetime import datetime
from omegaconf import OmegaConf

from src.utils import util
from src.eval.eval_utils import load_era5


@hydra.main(version_base=None, config_path="./configs", config_name="online.yaml")
def main(cfg):
    fpath = Path(f"{cfg.directory.home_path}/{cfg.buckets.bucket_base}/pangu_ifs_202407")
    variables = OmegaConf.load("./utils/pangu_config.yaml")
    
    start = "2024070100"
    end = "2024073100"
    start_ts = int(arrow.get(start, "YYYYMMDDHH").timestamp())
    end_ts = int(arrow.get(end, "YYYYMMDDHH").timestamp())
    era5_2024_levels = [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,925,950,1000]
    levels = [1000, 850, 700,600,500,400,300,250,200,150,100]
    lev_idxs_2024 = [era5_2024_levels.index(lev) for lev in levels]  
    era5_surf, era5_upper = load_era5(start_ts, end_ts, lev_idxs_2024, cfg)
    # print("era5_surf:", era5_surf, "era5_upper:", era5_upper)
    print(era5_surf.isel(time=0).shape, era5_upper.isel(time=0).shape)

    lev_idx_pangu = [0, 2, 3,4,5,6,7,8,9,10,11]  ## shared level: 1000, 850, 700,600,500,400,300,250,200,150,100
    avg_origin_levels = list(range(100, 1001, 50)) + [925, 50]
    avg_idx = [avg_origin_levels.index(l) for l in levels]
    surf_avg, high_avg = util.static4eval(avg_idx)
    surf_avg = np.concatenate((surf_avg[1:], surf_avg[:1]), axis=0)
    high_avg = np.concatenate((high_avg[:1], high_avg[2:3], high_avg[1:2], high_avg[3:]), axis=0)
    print(len(avg_origin_levels), avg_idx)
    bound = [0, 70, 60, 140]
    lat_stride = 60
    lon_stride = 70
    
    lat_weights = util.lat_weight()
    print("lat_weights:", lat_weights.shape)
    surf_avg = surf_avg[...,lat_stride:241-lat_stride,lon_stride:281-lon_stride]
    high_avg = high_avg[...,lat_stride:241-lat_stride,lon_stride:281-lon_stride]
    lat_weights = lat_weights[...,lat_stride:241-lat_stride,lon_stride:281-lon_stride]

    leads = list(range(6,121,6))
    for step in leads:
        scores = {"surf": {}, "high": {}}
        fscore_step = f"{cfg.directory.base_path}/pangu_july_score/ifs-init_score_pangu_step{step}_center{241-lat_stride*2}.npz"
        # if os.path.exists(fscore_step):
        #     print(fscore_step, "exists")
        #     continue
        cnt = 0
        for ts in range(start_ts, end_ts-5*3600*24, 3600*12):
            t = arrow.get(ts).format("YYYYMMDDHH")
            tdata = datetime.fromtimestamp(ts+step*3600)
            df_surf_era5 = era5_surf.sel(time=tdata)
            df_high_era5 = era5_upper.sel(time=tdata)
            ## pangu
            print(fpath/f"init-IFS_pangu_{t}_lead{step}.zarr")
            df_surf_pangu = xr.open_zarr(fpath/f"init-IFS_pangu_{t}_lead{step}.zarr", group="surf", consolidated=True).sel(latitude=slice(bound[2], bound[0]), longitude=slice(bound[1], bound[3]))["surface"][0]
            df_high_pangu = xr.open_zarr(fpath/f"init-IFS_pangu_{t}_lead{step}.zarr", group="high", consolidated=True).sel(latitude=slice(bound[2], bound[0]), longitude=slice(bound[1], bound[3]))["upper"]
            df_high_pangu = df_high_pangu[0,:,lev_idx_pangu]
            print(df_surf_era5.shape, df_surf_pangu.shape, df_high_pangu.shape, df_high_era5.shape)
            idx_era5 = [1,2,3,0]
            for k, var in enumerate(variables["input"]["surface"]):
                name = f"{var}"
                v_era5 = df_surf_era5[idx_era5[k]].to_numpy()
                v_pg = df_surf_pangu[k].to_numpy()#[...,:-1,:-1]
                ''' extract center region '''
                v_era5 = v_era5[...,lat_stride:241-lat_stride,lon_stride:281-lon_stride]
                v_pg = v_pg[...,lat_stride:241-lat_stride,lon_stride:281-lon_stride]
                ''''''
                rmse = util.RMSE(v_era5, v_pg, lat_weight=lat_weights)
                acc = util.ACC(v_era5, v_pg, yavg=surf_avg[k], lat_weight=lat_weights)
                if scores["surf"].get(name) is None:
                    scores["surf"][name] = [rmse, acc]
                else:
                    scores["surf"][name] = [rmse+scores["surf"][name][0], acc+scores["surf"][name][1]]
                # print(name, scores["surf"][name][0], scores["surf"][name][1])
            idx_era5 = [0,2,1,3,4]
            for k, var in enumerate(variables["input"]["high"]):
                name = var
                v_lev_era5 = df_high_era5[idx_era5[k]].to_numpy()#[...,:-1,:-1]
                v_lev_pg = df_high_pangu[k].to_numpy()#[...,:-1,:-1]
                ''' extract center region '''
                v_lev_era5 = v_lev_era5[...,lat_stride:241-lat_stride,lon_stride:281-lon_stride]
                v_lev_pg = v_lev_pg[...,lat_stride:241-lat_stride,lon_stride:281-lon_stride]
                ''''''
                rmse = util.RMSE(v_lev_era5, v_lev_pg, lat_weight=lat_weights)
                acc = util.ACC(v_lev_era5, v_lev_pg, yavg=high_avg[k], lat_weight=lat_weights)
                if scores["high"].get(name) is None:
                    scores["high"][name] = [rmse, acc]
                else:
                    scores["high"][name] = [rmse+scores["high"][name][0], acc+scores["high"][name][1]]
                # print(name, scores["high"][name][0], scores["high"][name][1])
            
            cnt += 1
 
        for k, v in scores["surf"].items():
            scores["surf"][k] = [scores["surf"][k][0]/cnt, scores["surf"][k][1]/cnt]
            # print(k, scores["surf"][k][0], scores["surf"][k][1])
        for k, v in scores["high"].items():
            scores["high"][k] = [scores["high"][k][0]/cnt, scores["high"][k][1]/cnt]
            # print(k, scores["high"][k][0], scores["high"][k][1])
        
        print(scores)
        np.savez_compressed(fscore_step, scores=scores)
        print(fscore_step, "done")
    
    return


if __name__=="__main__":
    main()

