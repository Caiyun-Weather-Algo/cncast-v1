import xarray as xr
import numpy as np
import os, time
from datetime import datetime
import arrow, hydra
from omegaconf import DictConfig

from evaluation import mid_rmse
from utils import util
import matplotlib.pyplot as plt


def load_ifs_cncast(t):
    # f = f"/home/lianghongli/weather-us-central1/cncast_v1_ifs-202407/init-IFS_pred_china_{t}_5d.zarr"
    # surf = xr.open_zarr(f, group="surf", consolidated=True)["surf"].to_numpy()[5::6,:4,...]
    # upper = xr.open_zarr(f, group="upper", consolidated=True)["upper"].to_numpy()[5::6,...]
    f = f"/home/lianghongli/weather-us-central1/cncast_v1_202407_aifs-bound/{t}_5d_cncast_v1.zarr"
    df_surf = xr.open_zarr(f, group="surf", consolidated=True)
    df_high = xr.open_zarr(f, group="upper", consolidated=True)
    surf = np.stack((df_surf["2mt"].to_numpy()[5::6,...], df_surf["mslp"].to_numpy()[5::6,...], df_surf["10m_u_component_of_wind"].to_numpy()[5::6,...], df_surf["10m_v_component_of_wind"].to_numpy()[5::6,...]), axis=1)
    upper = np.stack((df_high["geopotential"].to_numpy()[5::6,...], df_high["temperature"].to_numpy()[5::6,...], df_high["specific_humidity"].to_numpy()[5::6,...], df_high["u_component_of_wind"].to_numpy()[5::6,...], df_high["v_component_of_wind"].to_numpy()[5::6,...]), axis=1)
    print(surf.shape, upper.shape)
    # for i in range(surf.shape[0]):
    #     plt.imshow(surf[i,0])
    #     plt.savefig(f"/home/lianghongli/caiyun-algo-misc/caiyun-era5-pretrain/ifs-init_surf_{t}_{i}.png")
    return surf, upper


def load_ifs(t, use_ai=False):
    if use_ai:
        f = f"/home/lianghongli/weather-us-central1/IFS_china/AIFS_china_5d_{t}.zarr"
    else:
        f = f"/home/lianghongli/weather-us-central1/IFS_china/IFS_china_5d_{t}.zarr"
    surf = xr.open_zarr(f, group="surf", consolidated=True)["surface"].to_numpy()[:,:4]
    upper = xr.open_zarr(f, group="upper", consolidated=True)["upper"].to_numpy()
    print("surf and upper:", surf.shape, upper.shape)
    return surf, upper


def load_era5(start, end, levels):
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
            f = f"/home/lianghongli/era5_china_src/202407/2024/era5.{var}.{t}.nc"
            ds = xr.open_dataset(f)

            # Take every 6th hour (0, 6, 12, 18)
            data = ds[surf_var_abbr[k]].isel(time=[0,6,12,18])
            surf_vars_data.append(data)
            
        # Process upper air variables
        high_vars_data = []
        for k, var in enumerate(high_vars):
            f = f"/home/lianghongli/era5_china_src/202407/2024/era5.{var}.{t}.nc"
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


@hydra.main(version_base=None, config_path="/home/lianghongli/caiyun-algo-misc/caiyun-era5-pretrain/conf", config_name="train_40yr_config.yaml")
def main(cfg):
    bucket_central1 = "weather-us-central1"  # Replace with your bucket name
    eval_class = "cncast"
    lat_stride = 60
    lon_stride = 70
    lat_weights = util.lat_weight()[None,None,:]
    lat_weights = lat_weights[..., lat_stride:241-lat_stride,lon_stride:281-lon_stride]
    variables = util.reset_era5_stat()
    era5_levels = variables["levels"]
    
    if eval_class == "cncast":
        levels = [1000, 950, 850, 700, 600, 500,450,400,300,250,200,150,100]
    elif eval_class == "ifs":
        levels = [1000, 925, 850, 700, 600, 500,400,300,250,200,150,100,50]
    elif eval_class == "aifs":
        levels = [1000, 925, 850, 700, 600, 500,400,300,250,200,150,100,50]

    era5_2024_levels = [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,925,950,1000]
    lev_idxs = [era5_levels.index(lev) for lev in levels]
    lev_idxs_2024 = [era5_2024_levels.index(lev) for lev in levels]
    avgs = util.static4eval(lev_idxs)
    avgs = [avgs[0][...,lat_stride:241-lat_stride,lon_stride:281-lon_stride], avgs[1][...,lat_stride:241-lat_stride,lon_stride:281-lon_stride]]
    start = "2024070100"
    end = "2024073100"
    start_ts = int(arrow.get(start, "YYYYMMDDHH").timestamp())
    end_ts = int(arrow.get(end, "YYYYMMDDHH").timestamp())  
    era5_surf, era5_upper = load_era5(start_ts, end_ts, lev_idxs_2024)
    print("era5:", era5_surf.shape, era5_upper.shape)
    # era5_upper = era5_upper[0]

    
    for month in range(7,8):
        start_ts = int(arrow.get(start, "YYYYMMDDHH").timestamp())
        end_ts = int(arrow.get(end, "YYYYMMDDHH").timestamp())
        scores = {"surf":{}, "high":{}}
        rmse_srf = 0
        acc_srf = 0
        rmse_high = 0
        acc_high = 0
        damaged = 0
        for k, ts in enumerate(range(start_ts, end_ts-5*3600*24, 3600*12)):
            t0 = time.time()
            t = arrow.get(ts).format("YYYYMMDDHH")
            # Load data
            # try:
            if eval_class == "cncast":
                cncast_surf, cncast_upper = load_ifs_cncast(t)
            elif eval_class == "ifs":
                cncast_surf, cncast_upper = load_ifs(t, use_ai=False)
            elif eval_class == "aifs":
                cncast_surf, cncast_upper = load_ifs(t, use_ai=True)
                cncast_upper[:,0] = cncast_upper[:,0]/9.80665
            plt.imshow(cncast_surf[0,0])
            plt.savefig(f"/home/lianghongli/caiyun-algo-misc/caiyun-era5-pretrain/aifs-init_month{month}_surf_{t}.png")
            plt.close()
            exit()
            ''' extract center region '''
            cncast_surf = cncast_surf[...,lat_stride:241-lat_stride,lon_stride:281-lon_stride]
            cncast_upper = cncast_upper[...,lat_stride:241-lat_stride,lon_stride:281-lon_stride]

            lead1 = timestamp_to_datetime(ts+3600*6)
            lead120 = timestamp_to_datetime(ts+120*3600)
            y_surf = era5_surf.sel(time=slice(lead1, lead120)).to_numpy()
            y_high = era5_upper.sel(time=slice(lead1, lead120)).to_numpy()
            y_surf = y_surf[...,lat_stride:241-lat_stride,lon_stride:281-lon_stride]
            y_high = y_high[...,lat_stride:241-lat_stride,lon_stride:281-lon_stride]
            # plt.imshow(y_surf[0,0])
            # plt.savefig(f"/home/lianghongli/caiyun-algo-misc/caiyun-era5-pretrain/era5_month{month}_surf_{t}.png")

            print("geopotential: ", y_high[:,0,0].max(), y_high[:,0,0].min(), cncast_upper[:,0,0].max(), cncast_upper[:,0,0].min())
            rmse_srf = util.RMSE(y_surf, cncast_surf, lat_weights) + rmse_srf
            acc_srf = util.ACC(y_surf, cncast_surf, avgs[0], lat_weights) + acc_srf
            rmse_high = util.RMSE(y_high, cncast_upper, lat_weights) + rmse_high
            acc_high = util.ACC(y_high, cncast_upper, avgs[1], lat_weights) + acc_high
            print("one update costs:",t, time.time()-t0)
            # except:
            #     print(f"error loading ifs data for {t}")
            #     continue

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
        if eval_class == "cncast":
            fname = f"ifs-init_aifs-bound_month{month}_scores_china_5d_center{241-lat_stride*2}.npz"
        elif eval_class == "ifs":
            fname = f"ifs_month{month}_scores_china_5d_center{241-lat_stride*2}.npz"
        elif eval_class == "aifs":
            fname = f"aifs_month{month}_scores_china_5d_center{241-lat_stride*2}.npz"

        np.savez_compressed(f"/home/lianghongli/weather-caiyun/lianghl/pretrain_results/pred_recurse_bc/cncast-v1_scores/{fname}", 
                            scores=final_score, mid_rmse=scores)


if __name__=="__main__":
    main()
    
