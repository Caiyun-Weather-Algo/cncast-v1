import numpy as np
import arrow
from copy import deepcopy
from datetime import datetime
import xarray as xr

from src.utils import util
from src.data.feature_encoding import load_static


class BaseDataset():
    '''
    shared functions for all datasets
    '''
    def __init__(self, project_dir="./caiyun-cncast-v1"):
        self.variables = util.era5_stat()
        ## reset meanstd for total precipitation of ERA5 precip
        self.variables["surface"]["total_precipitation"]["meanstd"] = [5.27158476833704, 8.765997488569014]
        self.era5_levels = self.variables["levels"]
        self.t2021 = int(arrow.get("2021", "YYYY").timestamp())


    def level_idxs(self, levels):
        input_lev_idxs = [self.era5_levels.index(lev) for lev in levels]
        return input_lev_idxs
    
    def load_dem(self):
        dem = load_static("dem")
        dem = (dem-dem.min())/(dem.max()-dem.min())
        return dem
    
    def norm(self, data, level, name, v=None, norm_method="minmax"):
        #print(level, name, v, data.max(), data.min())
        if norm_method=="minmax":
            minmax = self.variables[level][name]["minmax"]
        else:
            minmax = self.variables[level][name]["meanstd"]

        if v is not None:
            minmax = minmax[:,v]

        if norm_method=="minmax" and minmax[0]>minmax[1]:
            minmax = minmax[::-1]
        if norm_method=="minmax":
            data[data<minmax[0]] = minmax[0]
            data[data>minmax[1]] = minmax[1]
            normed = (data - minmax[0]) / (minmax[1] - minmax[0])
        else:
            minmax = np.array(minmax, dtype=np.float32)
            if minmax.ndim>1:
                normed = (data - minmax.transpose(1,0)[0][:,None,None,None]) / minmax.transpose(1,0)[1][:,None,None,None]
            else:
                normed = (data-minmax[0])/minmax[1]
        return normed
    
    
    def denorm(self, data, level, name, v=None, norm_method="minmax"):
        if norm_method=="minmax":
            minmax = self.variables[level][name]["minmax"]
        else:
            minmax = self.variables[level][name]["meanstd"]
        if v is not None:
            minmax = minmax[:,v]
        # print(level, name, v, minmax, data.min(), data.max())
        if norm_method=="minmax":
            denormed = data * (minmax[1] - minmax[0]) + minmax[0]
        else:
            minmax = np.array(minmax, dtype=np.float32)
            if minmax.ndim>1:
                denormed = data * minmax.transpose(1,0)[1][:,None,None,None] + minmax.transpose(1,0)[0][:,None,None,None]
            else:
                denormed = data * minmax[1] + minmax[0]
        return denormed
    
    def filter_cmpa_ts(self, sample_start_t, cmpa_data, cmpa_days, cmpa_frame):
        sample_start_ts = deepcopy(sample_start_t)
        for start_t in sample_start_ts:
            if start_t>=self.t2021:
                shift = 8
            else:
                shift = 0
            # t = arrow.get(start_t).shift(hours=shift).format("YYYYMMDDHH")
            ts = [arrow.get(start_t).shift(hours=shift).shift(hours=h).format("YYYYMMDDHH") for h in range(-cmpa_frame+1, 1)]
            lack_cnt = 0
            for t in ts:
                try:
                    idx_cmpa_day = cmpa_days.index(t[:8])
                except:
                    lack_cnt += 1
                try:
                    cmpa_tp = cmpa_data[idx_cmpa_day]["precipitation"].sel(time=datetime(int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:10]))).to_numpy()
                except:
                    # print(f"time {t} cmpa not found")
                    lack_cnt += 1
            
            if lack_cnt>0:
                sample_start_t.remove(start_t)
        return sample_start_t
    
    def get_cmpa_tp(self, sample_start_time, cmpa_data, cmpa_days, cmpa_frame):
        sample_times = [sample_start_time+h*3600 for h in range(-cmpa_frame+1, 1)]
        cmpa_tps = []
        for cmpa_t in sample_times:
            if cmpa_t>=self.t2021:
                shift = 8
            else:
                shift = 0
            t = arrow.get(cmpa_t).shift(hours=shift).format("YYYYMMDDHH")
            idx_cmpa_day = cmpa_days.index(t[:8])
            cmpa_tp = cmpa_data[idx_cmpa_day]["precipitation"]
            # print(f"getting data of time {t}, {sample_start_time}")
            cmpa_tp = cmpa_tp.sel(time=datetime(int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:10]))).to_numpy()
            if cmpa_tp.ndim>2:
                cmpa_tp = cmpa_tp[0] ## for duplicated data
            cmpa_tp = cmpa_tp[::-1]
            cmpa_tp[np.isnan(cmpa_tp)] = 0.0
            cmpa_tp = util.tp2dbz(cmpa_tp)
            cmpa_tps.append(cmpa_tp)
        cmpa_tps = np.stack(cmpa_tps)
        cmpa_tp = self.norm(cmpa_tps, level="surface", name="cmpa_precipitation", v=None, norm_method="meanstd")
        return cmpa_tp


if __name__ == "__main__":
    dataset = BaseDataset()
    print(dataset.variables)