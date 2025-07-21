import xarray as xr
import numpy as np
import pandas as pd
import os, glob
import time, arrow
import earthkit.data as ekd
from datetime import datetime, timedelta
from einops import rearrange

import src.utils.util as util


class DataSource():
    def __init__(self, home_path, bucket_cfg):
        self.bucket_base = bucket_cfg.bucket_base
        self.bucket_era5 = bucket_cfg.bucket_era5
        self.bucket_ifs = bucket_cfg.bucket_ifs
        self.project = bucket_cfg.project
        self.home_path = home_path
        self.china_bbox = [0, 70, 60, 140]  ## bottom, left, top, right
    
    def __call__(self, src: str = "era5", 
                 start_time: str = "2010010100", 
                 end_time: str = "2022123123", 
                 lead_time: int = 1,
                 **kwargs
                 ):
        assert src in ["era5", "era5_tp", "ifs", "era5_2025", "cmpa"], "Invalid src"
        
        if src == "era5":
            surf, upper, upper_925 = self.call_era5(start_time, end_time)
            return surf, upper, upper_925
        elif src == "era5_tp":
            tp = self.call_era5_tp(start_time, end_time)
            return tp
        elif src == "ifs":
            surf, upper = self.call_ifs(start_time, lead_time, kwargs=kwargs)
            return surf, upper
        elif src == "era5_2025":
            levels = [1000, 925,850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
            surf, upper = self.call_era5_2025(start_time, lead_time, levels)
            return surf, upper
        elif src == "cmpa":
            self.start_time = int(arrow.get(str(start_time), "YYYYMMDDHH").timestamp())
            self.end_time = int(arrow.get(str(end_time), "YYYYMMDDHH").timestamp())
            cmpa_data,cmpa_days = self.call_cmpa(self.start_time, self.end_time)
            return cmpa_data, cmpa_days
        else:
            raise ValueError(f"Invalid src, src {src} not implemented")
            
    
    def call_era5(self, start_time, end_time):
        start_yr = int(start_time[:4])
        end_yr = int(end_time[:4])
        surf_data = xr.open_dataset(f"gcs://{self.bucket_base}/era5_china_src/surface_1980-2022.zarr", 
                                        backend_kwargs={"storage_options": {"project": self.project, "token": None}},
                                        engine="zarr",)
        surf_data = surf_data.sel(time=slice(datetime(start_yr,1,1,0), datetime(end_yr+1,1,1,0)))
        high_data = [xr.open_dataset(f"gcs://{self.bucket_base}/era5_china_src/high_{yr}.zarr", 
                                        backend_kwargs={"storage_options": {"project": self.project, "token": None}},
                                        engine="zarr",) for yr in range(start_yr, end_yr+1)]
        high_925 = [xr.open_dataset(f"gcs://{self.bucket_base}/era5_china_src/high_{yr}_925-50hPa.zarr", 
                                        backend_kwargs={"storage_options": {"project": self.project, "token": None}},
                                        engine="zarr",) for yr in range(start_yr, end_yr+1)]
        return surf_data, high_data, high_925
    
    def call_era5_tp(self, start_time, end_time):
        tp = xr.open_dataset(f"gcs://{self.bucket_base}/era5_china_src/era5_tp_1980-2022.zarr", 
                                        backend_kwargs={"storage_options": {"project": "colorful-aia", "token": None}},
                                        engine="zarr",)
        start = datetime(int(start_time[:4]), 1, 1, 0)
        end = datetime(int(end_time[:4])+1, 1, 1, 0)
        tp = tp.sel(time=slice(start, end))
        return tp
    
    def call_ifs(self, start_time, lead_hr:int=1, **kwargs):
        init_hr = start_time[8:10]
        f = f"{start_time[:10]}0000-{lead_hr}h-oper-fc.grib2"
        # print("IFS file:", f)
        ec_files = glob.glob(f"{self.home_path}/{self.bucket_ifs}/{start_time[:8]}/{init_hr}z/ifs/0p25/oper/{f}")
        field = ekd.from_source("file", ec_files[0])
        latlons = field.to_latlon()
        lat_index = [list(latlons["lat"][:, 0]).index(self.china_bbox[2]), list(latlons["lat"][:, 0]).index(self.china_bbox[0]) + 1]
        lon_index = [list(latlons["lon"][0, :]).index(self.china_bbox[1]), list(latlons["lon"][0, :]).index(self.china_bbox[3]) + 1]
        ## variables
        def preprocess_ifs(fields, param, level=None):
            """选择并按顺序处理不同层次的气象数据""" ## from hess
            data = fields.sel(param=param)
            if level:
                data = data.sel(level=level)
            data = data.order_by(param=param, level=level)
            return data.to_numpy()

        if "surf_var" not in kwargs["kwargs"]["args"]:
            surf = ["msl", "10u", "10v", "2t", "tp"]
        else:
            surf = kwargs["kwargs"]["args"]["surf_var"]
        
        if "upper_var" not in kwargs["kwargs"]["args"]:
            upper_pl = (
                ["gh", "q", "t", "u", "v"],
                [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
            )
        else:
            upper_pl = kwargs["kwargs"]["args"]["upper_var"]

        upper_data = preprocess_ifs(field, *upper_pl)[:, lat_index[0]:lat_index[1], lon_index[0]:lon_index[1]]
        upper_data = rearrange(upper_data, '(v l) h w -> v l h w', v=5)
        upper_data[0] *= 9.80665 ## convert geopotential height to geopotential
        surf_data = preprocess_ifs(field, surf)[:, lat_index[0]:lat_index[1], lon_index[0]:lon_index[1]]
        if "tp" in surf:
            surf_data[-1] *= 1000 ## convert tp from m to mm
        return surf_data, upper_data
    
    def call_cmpa(self, start_timestamp, end_timestamp):
        cmpa_ds = [arrow.get(timestamp).format("YYYYMMDD") for timestamp in range(start_timestamp, end_timestamp, 24*3600)]
        cmpa_data = []
        cmpa_days = []
        for t in cmpa_ds:
            t0 = time.time()
            if os.path.exists(f"{self.home_path}/{self.bucket_base}/cmpa_5km/{t[:6]}/cmpa_{t}.zarr"):
                ds = xr.open_dataset(f"gcs://{self.bucket_base}/cmpa_5km/{t[:6]}/cmpa_{t}.zarr", 
                                        backend_kwargs={"storage_options": {"project": self.project, "token": None}},
                                        engine="zarr",)
                cmpa_days.append(t)
                cmpa_data.append(ds)
        return cmpa_data,cmpa_days

    
    def call_era5_2025(self, start_time, lead_time, levels):
        surf_vars = ["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation"]
        high_vars = ["geopotential", "temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"]
        surf_var_abbr = ["t2m", "msl", "u10", "v10", "tp"]
        high_var_abbr = ["z", "t", "q", "u", "v"]
        start_time = arrow.get(start_time, "YYYYMMDDHH").shift(hours=+lead_time).format("YYYYMMDDHH")
        # Initialize empty lists to store data for each variable
        start = np.datetime64(f"{start_time[:4]}-{start_time[4:6]}-{start_time[6:8]}T{start_time[8:]}")
        surf_vars_data = []
        for k, var in enumerate(surf_vars):
            f = f"{self.home_path}/{self.bucket_era5}/2025/era5.{var}.{start_time[:8]}.nc"
            ds = xr.open_dataset(f)
            data = ds[surf_var_abbr[k]].sel(valid_time=start)
            surf_vars_data.append(data)
            
        # Process upper air variables
        high_vars_data = []
        for k, var in enumerate(high_vars):
            f = f"{self.home_path}/{self.bucket_era5}/2025/era5.{var}.{start_time[:8]}.nc"
            ds = xr.open_dataset(f)
            data = ds[high_var_abbr[k]].sel(valid_time=start, pressure_level=levels)
            high_vars_data.append(data)
        
        # Concatenate variables for this time step
        surf_ds = np.stack(surf_vars_data, axis=0)
        surf_ds[-1] *= 1000 ## convert tp from m to mm
        high_ds = np.stack(high_vars_data, axis=0)
        return surf_ds, high_ds 


if __name__=="__main__":
    from typing import OrderedDict
    from omegaconf import OmegaConf
    start_time = "2025042012"
    cfg = OmegaConf.load("configs/train.yaml")
    source = DataSource(home_path="/home/lianghongli", bucket_cfg=cfg.buckets)
    ifs_surf, ifs_high = source("ifs", start_time, lead_time=0, args={"surf_var": ["2t", "msl", "10u", "10v", "tp"], 
                                            "upper_var": (["gh", "t", "q", "u", "v"], [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100])})
    # print(ifs_surf.shape, ifs_high.shape)
    
    start_time = "2020042012"
    era5_surf, era5_high, era5_high_925 = source("era5", start_time, end_time="2021042012")
    # print(era5_surf, era5_high, era5_high_925)

    cmpa_data,cmpa_days = source("cmpa", start_time, end_time="2021042012")
    print(cmpa_data[0],cmpa_days)