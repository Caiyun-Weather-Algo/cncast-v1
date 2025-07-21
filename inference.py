import os
import numpy as np
from omegaconf import DictConfig
import yaml, time, arrow
import cv2
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import argparse
import multiprocessing as mp
from PIL import Image

from src.data.datasets import IFSDataset, ERA5Dataset
from src.data.data2file import ds2grib, to_varsplit_dataset
from src.utils import util, mapplot_utils
from src.utils.model_utils import load_ckpt, iter_predict, iter_pred_tp
from src.utils.color_configs import ColorConfig
from src.utils.pylogger import RankedLogger

from src.data.feature_encoding import get_coords, get_time_features, load_static 


def get_static_data():
    z = load_static("z")
    lsm = load_static("lsm")
    slt = load_static("slt")
    z = (z-z.min())/(z.max()-z.min())
    slt = (slt-slt.min())/(slt.max()-slt.min())
    coords = get_coords(degree=False)
    static = np.concatenate((z, lsm, slt, coords), axis=0)
    return static


def convert2uint8(x, norm, type_lev, name, level_idx=None):
    x = norm(x, type_lev, name=name, v=level_idx, norm_method="minmax")
    x = (x*255).astype(np.uint8)
    return x


def show_comparison(data_cncast, data_ifs, cfg, lead, t, fig_savepath):
    surf, high, tp, tp_sr = data_cncast
    surf_ifs, high_ifs, tp_ifs = data_ifs
    title = ["CNCast", "IFS"]
    bbox = [0, 70, 60, 140]
    var2plot = cfg.visualization.var2plot
    '''------- plot non-precipitation variables ----------'''
    for i, v in enumerate(cfg.input["surface"]):
        if v in var2plot["surface"]:
            x, xifs = surf[i], surf_ifs[i]
            cbar_args = ColorConfig[v]
            figname = f"{fig_savepath}/{t}_{v}_lead{lead}_cncast-vs-ifs.png"
            mapplot_utils.map_proj_n([x, xifs], cbar_args, figname, title, fig_layout=[1,2], bbox_list=[bbox]*2, share_cbar=True, suptitle=v, color_discretized=False)
    for i, v in enumerate(cfg.input["high"]):
        if v in var2plot["high"]:
            for lk, l in enumerate(cfg.input["levels"]):
                if l in var2plot["level"]:
                    x, xifs = high[i, lk], high_ifs[i, lk]
                    cbar_args = ColorConfig[v][str(l)]
                    figname = f"{fig_savepath}/{t}_{v}-{l}_lead{lead}_cncast-vs-ifs.png"
                    mapplot_utils.map_proj_n([x, xifs], cbar_args, figname, title, fig_layout=[1,2], bbox_list=[bbox]*2, share_cbar=True, suptitle=v, color_discretized=False)

    ''' plot precipitation '''
    cbar_args = ColorConfig["precip"]
    x = [tp, tp_ifs]
    figname = f"{fig_savepath}/{t}_tp-0p25_lead{lead}_cncast-vs-ifs.png"
    mapplot_utils.map_proj_n(x, cbar_args, figname, title, fig_layout=[1,2], bbox_list=[bbox]*2, share_cbar=True, suptitle="total_precipitation", color_discretized=True)
    ## mapping super resolutionized
    x = [tp_sr, tp_ifs]  # Removed undefined tp_pred_5d_sr reference
    figname = f"{fig_savepath}/{t}_tp-0p06_lead{lead}_cncast-vs-ifs.png"
    title = ["cncast tp-0p06", "IFS"]
    mapplot_utils.map_proj_n(x, cbar_args, figname, title, fig_layout=[1,2], bbox_list=[bbox]*2, share_cbar=True, suptitle="total_precipitation", color_discretized=True)
    return


def plot_results(arg_list):
    pred, tp_pred, tp_sr_pred, lead, dataset, t, cfg, fig_savepath, version, ifs_compare = arg_list
    surf, high = pred
    x_webmerc = util.projection(surf, "EPSG:4326", "EPSG:3857")
    
    var2plot = cfg.visualization.var2plot
    for j, v in enumerate(cfg.input["surface"]):
        if v in var2plot["surface"]:
            figname = f"{fig_savepath}/{t}_{v}_lead{lead}_cncast_{version}_webmerc.png"
            if v == "2mt":
                img = Image.fromarray(mapplot_utils.apply_cmap_nafp(x_webmerc[j], v))
                img.save(figname)
            else:
                x = convert2uint8(x_webmerc[j], dataset.norm, "surface", v, level_idx=None)
                cv2.imwrite(figname, mapplot_utils.apply_colormap(x))
            
        
    ''' -------------- plot total precipitation ------------'''
    tp_pred_webmerc = util.projection(tp_pred[None,:], "EPSG:4326", "EPSG:3857")[0]
    img = Image.fromarray(mapplot_utils.color_convert(tp_pred_webmerc))
    figname = f"{fig_savepath}/{t}_tp-0p25-dit_lead{lead}_cncast_{version}_webmerc.png"
    img.save(figname)
    if cfg.hyper_params.tp_high_res:
        tp_pred_sr_webmerc = util.projection(tp_sr_pred[None,:], "EPSG:4326", "EPSG:3857")[0]
        figname = f"{fig_savepath}/{t}_tp-0p06-dit_lead{lead}_cncast_{version}_webmerc.png"
        img = Image.fromarray(mapplot_utils.color_convert(tp_pred_sr_webmerc))
        img.save(figname)
    
    for j, v in enumerate(cfg.input["high"]):
        if var2plot["high"] is None:
            continue
        if v not in var2plot["high"]:
            continue
        for lk, l in enumerate(cfg.input["levels"]):
            if l in var2plot["level"]:
                x = util.projection(high[j, lk:lk+1], "EPSG:4326", "EPSG:3857")[0]
                x = convert2uint8(x, dataset.norm, "high", v, level_idx=dataset.input_lev_idxs[lk])
                figname = f"{fig_savepath}/{t}_{v}-{l}_lead{lead}_cncast_{version}_webmerc.png"
                cv2.imwrite(figname, mapplot_utils.apply_colormap(x))
    
    '''------------ plot comparison between cncast and IFS if needed ------------'''
    if ifs_compare:
        if lead%3>0:
            lead_hr = lead + 3 - lead%3  # Fixed: lead_hr was undefined
        else:
            lead_hr = lead
        ifs_fc = dataset.process_ifs(init_time=t, lead_time=lead_hr, norm=False)
        ifs_fc_surf, ifs_fc_high = ifs_fc  # Fixed: ifs_fc_srf -> ifs_fc_surf
        ifs_tp = dataset.process_ifs(init_time=t, lead_time=lead_hr, norm=False, tp_only=True)
        show_comparison([surf, high, tp_pred, tp_sr_pred], [ifs_fc_surf, ifs_fc_high, ifs_tp], 
                        cfg, lead, t, fig_staging_path)  

    return


def runner(args):
    tstart = time.time()
    logger = RankedLogger(name="cncast_v1", log_dir=args.log_dir)
    ''' getting configs '''
    cfg = yaml.load(open(args.config), Loader=yaml.Loader) 
    cfg = DictConfig(cfg) 
    logger.log(f"configs: {cfg}")

    try:
        ''' loading checkpoints according to configs and device '''
        device = "cuda" 
        models = {}
        for k in [1,3,6,24]:
            models[f"lead_{k}h"] = load_ckpt(k, cfg, target="non-tp").to(device)
        models["tp"] = load_ckpt(1, cfg, target="tp").to(device)
        models["tp-sr"] = load_ckpt(1, cfg, target="tp-sr").to(device)
        logger.log("Model checkpoints loaded!")

        levels = cfg.input.levels
        cfg.hyper_params.batch_size = 1
        forecast_day = args.forecast_day

        ''' setting coordinates for static features and to save predicted results '''
        bbox = [0, 70, 60, 140]
        lats = np.arange(bbox[2], bbox[0]-0.1, -0.25)
        lons = np.arange(bbox[1], bbox[3]+0.1, 0.25)
        lats006 = np.linspace(bbox[2], bbox[0], 1024, dtype=np.float32)
        lons006 = np.linspace(bbox[1], bbox[3], 1024, dtype=np.float32)
        version = "v1"
        compare_ifs = cfg.visualization.compare_ifs
        
        current_time = int((time.time()+8*3600)//(3600*12) * (3600*12) - 12*3600) ## for online
        # current_time  = int(arrow.get("2024080100", "YYYYMMDDHH").timestamp()) ## for test
        current_tstr = arrow.get(current_time).format("YYYYMMDDHH")
        ''' loading dataset '''
        if args.data_source=="ifs":
            ''' real-time inference for online data '''
            dataset = IFSDataset(
                            input_vars=cfg.input, 
                            output_vars=cfg.input, 
                            bucket_cfg=cfg.buckets, 
                            use_dem=cfg.hyper_params.use_dem, 
                            norm_method=cfg.hyper_params.norm_method,
                            start_time=current_tstr, 
                            )
        else:
            ''' test model performance with historical ERA5 data '''
            dataset = ERA5Dataset(
                input_vars=cfg.input,
                output_vars=cfg.input,
                bucket_cfg=cfg.buckets, 
                use_dem=cfg.hyper_params.use_dem,
                norm_method=cfg.hyper_params.norm_method,
                mode="test", 
                input_hist_hr=0,
                sample_interval=1, 
                end_time="2021122700", 
                test_start_time="2021010100", 
                resize_data=False, 
            )
        logger.log(f"initiated at {current_tstr}, loading IFS dataset done! dataloader length {len(dataset)}")   
        dem = dataset.load_dem()
        dem = torch.broadcast_to(torch.tensor(dem[None,:], dtype=torch.float32, device=device), (cfg.hyper_params.batch_size, 1, 241, 281))
        loader = DataLoader(dataset, batch_size=1)
        logger.log(f"ifs dataloader done, dataloader length {len(ifs_loader)}!")
        
        static_args = (lats, lons, levels, dem, forecast_day, models, dataset, args.data_source)

        for k, data in enumerate(loader):
            if args.data_source=="ifs":
                ''' real-time inference for online data '''
                sample_start_time, x_surf, x_high = data
            else:
                sample_start_time, x_surf, x_high, _, _, _ = data
                logger.log(f"data prepared for {current_tstr}, start iterative prediction")
            
            ''' make directories '''
            current_tstr = arrow.get(sample_start_t).format("YYYYMMDDHH")
            tdir = arrow.get(sample_start_t).format("YYYY/MM/DD/HH")
            fig_savepath = f"{cfg.directory.save_path}/cncast_{version}/png/{tdir}"  
            data_savepath = f"{cfg.directory.save_path}/cncast_{version}/{args.data_format}/{tdir}" 
            os.makedirs(data_savepath, exist_ok=True)
            os.makedirs(fig_savepath, exist_ok=True)
            ''' prepare time stamps for xr.Dataset savery '''
            fcst_ts = range(3600*1+sample_start_time, 24*forecast_day*3600+sample_start_time+1, 3600)
            fcst_ts = [datetime.fromtimestamp(t) for t in fcst_ts]
            x_surf = x_surf.to(device)
            x_high = x_high.to(device)

            logger.log(f"data prepared for {current_tstr}, start iterative prediction")
            ''' iterative prediction and save data to xarray.Dataset'''
            t0 = time.time()
            fcst_surf, fcst_high = iter_predict(x_surf, x_high, current_time, cfg, static_args)

            ''' save the resumed data to grib file '''
            surf_fc, high_fc = dataset.resume([fcst_surf.to_numpy().copy(), fcst_high.to_numpy().copy()])
            surf_fc = np.squeeze(surf_fc)
            high_fc = np.squeeze(high_fc)
            
            if args.data_format=="zarr":
                surf = util.to_dataset(surf_fc, times, cfg.input.surface, lats, lons, levels=None)
                high = util.to_dataset(high_fc, times, cfg.input.high, lats, lons, levels=levels)
                filename = f"{data_savepath}/pred_china_{current_tstr}_cncast_{version}_5d.zarr"
                surf.to_zarr(filename, mode="w", group="surf", consolidated=True)
                high.to_zarr(filename, mode="a", group="upper", consolidated=True)
            elif args.data_format=="grib2":
                for i in range(surf_fc.shape[0]):
                    # Extract the data for the current step
                    surf_step = surf_fc[i:i+1]
                    high_step = high_fc[i:i+1]
                    fcst_ts_step = [fcst_ts[i]]
                    # Create datasets for the current step
                    ds_surf_step = to_varsplit_dataset(surf_step, fcst_ts_step, cfg.input.surface, lats, lons, levels=None)
                    ds_high_step = to_varsplit_dataset(high_step, fcst_ts_step, cfg.input.high, lats, lons, levels=levels)
                    # Generate the file name for the current step
                    f2save_step = f"{data_savepath}/{current_tstr}_{i+1:03d}h_cncast_{version}.grib2"
                    # Save the data to a grib2 file
                    ds2grib(ds_surf_step, ds_high_step, f2save_step, step=i+1)
                    # Create an index file for the current grib2 file, wgrib installation: https://github.com/NOAA-EMC/wgrib2
                    index_file = f2save_step.replace(".grib2", ".index")
                    os.system(f"wgrib2 {f2save_step} -s > {index_file}")
            else:
                raise ValueError(f"Invalid data format, data format {args.data_format} not implemented")
            
            t1 = time.time()
            logger.log(f"----------------- swin3d iterative prediction and data savery costs {t1-t0} --------------------------")

            ''' diagnose precipitation at resolution of 0.25 degrees '''
            logger.log("-------------------------------start predicting precip-------------------------------")
            static_args = (forecast_day, lats, lons, lats006, lons006, device)
            tp_models = (models["tp"], models["tp-sr"])
            ''' denorm(resume) the variables '''
            surf = fcst_surf.to_numpy()
            high = fcst_high.to_numpy()
            tp_pred, tp_pred_sr = iter_pred_tp(surf, high, current_time, dataset, tp_models, static_args, tp_high_res=cfg.hyper_params.tp_high_res) 
            t2 = time.time()
            logger.log(f"----------------- precipitation diagnosis costs {t2-t1} --------------------------")
            
            ''' Save tp-0p25 to grib2 file '''
            f2save_grib = f"{data_savepath}/{current_tstr}_{forecast_day}d_tp-0p25-dit_cncast_{version}.grib2"
            ds_tp = to_varsplit_dataset(tp_pred["surface"].to_numpy()*1e-3, fcst_ts, ["total_precipitation"], lats, lons, levels=None)
            ds2grib(ds_tp, None, f2save_grib, step=forecast_day*24)
            # Create index file for the grib2 file
            index_file = f2save_grib.replace(".grib2", ".index")
            os.system(f"wgrib2 {f2save_grib} -s > {index_file}")

            ''' Save tp-sr to grib2 file '''
            if cfg.hyper_params.tp_high_res:
                f2save_grib = f"{data_savepath}/{current_tstr}_{forecast_day}d_tp-0p06-dit_cncast_{version}.grib2"
                ds_tp_sr = to_varsplit_dataset(tp_pred_sr["surface"].to_numpy()*1e-3, fcst_ts, ["total_precipitation"], lats006, lons006, levels=None)
                ds2grib(ds_tp_sr, None, f2save_grib, step=forecast_day*24)
                # Create index file for the grib2 file
                index_file = f2save_grib.replace(".grib2", ".index")
                os.system(f"wgrib2 {f2save_grib} -s > {index_file}")

            t3 = time.time()
            logger.log(f"----------------- precipitation results saving to grib costs {t3-t2} --------------------------")
            
            ''' plot and project the variables for showing the results '''
            step = cfg.visualization.step
            start_lead = cfg.visualization.start_lead
            tp_pred = tp_pred["surface"].to_numpy()[:,0]
            if cfg.hyper_params.tp_high_res:
                tp_pred_sr = tp_pred_sr["surface"].to_numpy()[:,0]
            else:
                tp_pred_sr = [None]*24*forecast_day
            tasks = []
            for lead in range(start_lead, 24*forecast_day+1, step):
                arg_list = ([surf_fc[lead-1], high_fc[lead-1]], tp_pred[lead-1], tp_pred_sr[lead-1], 
                                lead, dataset, current_tstr, 
                                cfg, fig_savepath, version, compare_ifs)
                tasks.append(arg_list)
                # plot_results(arg_list)
            
            # Use multiprocessing Pool to parallelize plotting
            num_processes = 8
            with mp.Pool(processes=num_processes) as pool:
                pool.map(plot_results, tasks)
            logger.log(f"----------------- plotting pred results costs {time.time()-t3} --------------------------") 
                
        logger.log(f"------------------------------total time costs {time.time()-tstart}----------")
    except Exception as e:
        logger.log(f"making forecast error: {e}")
    return


def main():
    parser = argparse.ArgumentParser(description="predict iteratively")
    parser.add_argument("--config", type=str, default="./configs/online.yaml", help="model and hp settings", )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--forecast_day", type=int, default=5, help="max lead time of days")
    parser.add_argument("--log_dir", type=str, default="./logs", help="log directory")
    parser.add_argument("--data_source", type=str, default="ifs", help="data source, ifs or era5")
    parser.add_argument("--data_format", type=str, default="grib2", help="format of saving data, grib2 or zarr")
    args = parser.parse_args()
    runner(args)


if __name__=="__main__":
    main()

