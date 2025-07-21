import arrow
import time
from datetime import datetime
import numpy as np
import yaml, hydra
from einops import rearrange

from src.utils import util
from src.data.feature_encoding import load_static
from src.data.base_dataset import BaseDataset
from src.data.data_source import DataSource


class ERA5Dataset(BaseDataset):
    '''
    从原始ERA5数据文件加载不同变量
    输入输出所需变量在train.yaml文件中设置；
    args:
        input_vars: list of string, variables to use
        output_vars: list of string, variables to forecast
        input_hist_hr: int, 使用几个历史时次数据（默认历史时次连续，暂不支持间隔大于1h）
        forecast_step: int， 预报未来多少个时次
    return:
        x, y: array with 3 dims
    '''
    def __init__(self, 
                 input_vars, 
                 output_vars, 
                 bucket_cfg, 
                 target="era5", 
                 cmpa_frame=1, 
                 input_hist_hr=0,
                 forecast_step=6, 
                 sample_interval=1, 
                 norm_method="minmax", 
                 use_dem=False, 
                 start_time="1980010100", 
                 end_time="2021122700", 
                 cut_era5=False, 
                 with_era5_tp=False, 
                 resize_data=False, 
                 ):
        super().__init__()
        self.variables = util.era5_stat()
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.forecast_step = forecast_step
        self.input_hist_hr = input_hist_hr
        self.norm_method = norm_method
        self.use_dem = use_dem
        self.bucket_cfg = bucket_cfg
        self.target = target
        self.cmpa_frame = cmpa_frame
        self.cut_era5 = cut_era5
        self.with_era5_tp = with_era5_tp
        self.resize_data = resize_data
        if self.target=="cmpa":
            self.cut_era5 = True

        self.data_src = DataSource(home_path="/home/lianghongli", bucket_cfg=bucket_cfg)
        self.surf_data, self.high_data, self.high_925 = self.data_src(src="era5", start_time=start_time, end_time=end_time)
        self.tp_data = self.data_src(src="era5_tp", start_time=str(start_time), end_time=str(end_time))

        if target=="cmpa":
            self.cmpa_data, self.cmpa_days = self.data_src(src="cmpa", start_time=str(start_time), end_time=str(end_time))

        self.sample_interval = sample_interval
        era5_levels = self.variables["levels"]
        self.input_lev_idxs = [era5_levels.index(j) for j in self.input_vars["levels"]]
        self.t1980 = int(arrow.get("1980010100", "YYYYMMDDHH").timestamp())
        self.tstart_anchor_yr = int(start_time[:4])
        self.tstart_anchor = int(arrow.get(f"{start_time[:4]}010100", "YYYYMMDDHH").timestamp())
        
        self.start_time = int(arrow.get(start_time, "YYYYMMDDHH").timestamp())
        self.end_time = int(arrow.get(end_time, "YYYYMMDDHH").timestamp())
        self.sample_start_t = self.gen_sample_times()
        print(f"sample numer: {len(self.sample_start_t)}")
        if resize_data:
            self.interp_idxs = util.nearest_interp_idxs(src_shape=(241, 281), dst_shape=(256, 256))
        
        if use_dem:
            self.dem = self.load_dem()
           
    def gen_sample_times(self):
        sample_start_t = list(range(self.start_time, self.end_time, 1*3600))
        #sample_start_t = sample_start_t[::self.sample_interval]
        n = len(sample_start_t)
        if self.sample_interval>1:
            count = len(sample_start_t)//self.sample_interval
            np.random.seed(2023)
            sample_start_t = np.random.choice(sample_start_t, count)
        
        if self.target=="cmpa":
            sample_start_t = self.filter_cmpa_ts(sample_start_t, self.cmpa_data, self.cmpa_days, self.cmpa_frame)
        return sample_start_t  
        
    
    def get_idx_high(self, timestamp):
        yr = arrow.get(int(timestamp)).format("YYYY")
        yr_idx = int(eval(yr)-self.tstart_anchor_yr)
        idx = int((timestamp - (arrow.get(yr, "YYYY").timestamp()))//3600)
        return yr_idx, idx

    def get_bc(self, surf, high):
        surf_bc_bottom = surf[:,-4:,:]
        surf_bc_top = surf[:,:4,:]
        surf_bc_left = surf[:,:,:4]
        surf_bc_right = surf[:,:,-4:]
        high_bc_bottom = rearrange(high[...,-4:,:], '(v d) h w -> v d h w', v=5)
        high_bc_top = rearrange(high[...,:4,:], '(v d) h w -> v d h w', v=5)
        high_bc_left = rearrange(high[...,:,:4], '(v d) h w -> v d h w', v=5)
        high_bc_right = rearrange(high[...,:,-4:], '(v d) h w -> v d h w', v=5)
        xbc = [[surf_bc_bottom, surf_bc_left, surf_bc_top, surf_bc_right], [high_bc_bottom, high_bc_left, high_bc_top, high_bc_right]]
        return xbc

    def get_bc_from_file(self, tfcst: str = "2021070100"):
        fbc_surf = f"gcs://{self.bucket_cfg.bucket_era5}/bc_2021/bc_surf_{tfcst}.zarr"
        fbc_high = f"gcs://{self.bucket_cfg.bucket_era5}/bc_2021/bc_high_{tfcst}.zarr"
        ds_bc_surf = xr.open_zarr(fbc_surf)
        ds_bc_high = xr.open_zarr(fbc_high)
        bc_surf, bc_high = [], []
        for j,bcname in enumerate(["bottom", "left", "top", "right"]):
            var_name = f"bc_surf_{bcname}"
            data = ds_bc_surf[f"bc_{bcname}"].sel(lat=lats_bc[j],lon=lons_bc[j]).values
            bc_surf.append(data)
            var_name = f"bc_high_{bcname}"
            data = ds_bc_high[f"bc_{bcname}"].sel(lat=lats_bc[j],lon=lons_bc[j]).values
            bc_high.append(data)
        
        xbc = [bc_surf, bc_high]
        return xbc
    
    def get_cmpa_tp(self, sample_start_time):
        sample_times = [sample_start_time+h*3600 for h in range(-self.cmpa_frame+1, 1)]
        cmpa_tps = []
        for cmpa_t in sample_times:
            if cmpa_t>=self.t2021:
                shift = 8
            else:
                shift = 0
            t = arrow.get(cmpa_t).shift(hours=shift).format("YYYYMMDDHH")
            idx_cmpa_day = self.cmpa_days.index(t[:8])
            cmpa_tp = self.cmpa_data[idx_cmpa_day]["precipitation"]
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
    
    def get_era5_tp(self, indices):
        tp = self.tp_data["total_precipitation"][indices].to_numpy()
        tp = util.tp2dbz(tp*1000)
        if tp.ndim==2:
            tp = tp[None,:]
        tp = self.norm(tp, level="surface", name="total_precipitation", v=None, norm_method="meanstd")
        
        if self.resize_data:
            tp = util.resize(tp, self.interp_idxs)
            
        if self.cut_era5:
            tp = tp[:,:181]
        return tp
    
    def get_era5_surf(self, indices):
        x_surf = self.surf_data['2m_temperature'][indices].to_numpy()
        if x_surf.ndim==3:
            x_surf = x_surf[None,:]
        
        x_surf = np.concatenate([self.norm(x_surf[:,k:k+1], "surface", self.input_vars["surface"][k], v=None, norm_method=self.norm_method) for k in range(x_surf.shape[1])], axis=1)
        x_surf = x_surf.astype(np.float32)
        x_surf = rearrange(x_surf, 't v h w -> (t v) h w')
        if self.cut_era5:
            x_surf = x_surf[:,:181]
        return x_surf
    
    def get_era5_high(self, sample_start_time):
        x_high_all = []
        for start_time in sample_start_time:
            yr_idx, t_idx = self.get_idx_high(start_time)
            x_high = self.high_data[yr_idx]['geopotential'][t_idx:(t_idx+1),:].to_numpy()  #
            x_high_all.append(x_high)
        x_high = np.concatenate(x_high_all, axis=0)
        
        for j,v in enumerate(self.input_lev_idxs):
            x_high[:,:,j] = np.concatenate([self.norm(x_high[:, k:k+1, j], "high", self.input_vars["high"][k], v, norm_method=self.norm_method) for k in range(x_high.shape[1])], axis=1)
        
        x_high = x_high.astype(np.float32)
        x_high = rearrange(x_high, 't v d h w -> (t v d) h w')
        if self.cut_era5:
            x_high = x_high[:,:181]
        return x_high
        
    def __len__(self):
        return len(self.sample_start_t)
    
    def __getitem__(self, idx):
        sample_start_time = self.sample_start_t[idx]
        # t = arrow.get(sample_start_time).format("YYYY-MM-DDTHH:mm:ss")
        idx_surf = int((sample_start_time-self.tstart_anchor)//3600)
        yr_idx, t_idx = self.get_idx_high(sample_start_time)
        x_surf = self.get_era5_surf(idx_surf)
        x_high = self.get_era5_high([sample_start_time])
        
        ### y ====
        y_surf = self.get_era5_surf(idx_surf+1+self.input_hist_hr)
        y_high = self.get_era5_high([sample_start_time+3600*self.forecast_step])
        
        ## get boundary condition
        xbc = self.get_bc(y_surf, y_high)
        
        if self.use_dem:
            x_surf = np.concatenate((x_surf, self.dem[None,:].astype(np.float32)), axis=0)
        
        x = np.concatenate((x_surf, x_high), axis=0)
        y = np.concatenate((y_surf, y_high), axis=0)
        ## CMPA precipitaion
        if self.target=="cmpa":
            cmpa_tp = self.get_cmpa_tp(sample_start_time)
        ## 
        if self.target=="cmpa":
            if self.with_era5_tp:
                x_tp = self.get_era5_tp(idx_surf)
                return sample_start_time, x, cmpa_tp, x_tp
            else:
                return sample_start_time, x, cmpa_tp
        else:
            if self.forecast_step>0: ## for prediction model
                return sample_start_time, x, y, xbc
            else: ## for VAE model
                return sample_start_time, x

    
    def resume(self, output):
        if isinstance(output, tuple) or isinstance(output, list):
            surf = rearrange(output[0], 'b (n l) h w -> b n l h w', l=1)
            high = output[1]
        else:
            output = rearrange(output, 'b (n l) h w -> b n l h w', n=5)
            l = output.shape[2]
            surf, high = np.split(output, 1, dim=2)
        surf = rearrange(surf, 'b (t v) l h w -> b t v l h w', 
                            v=len(self.input_vars["surface"]))
        high = rearrange(high, 'b (t v) l h w -> b t v l h w', 
                            v=len(self.input_vars["high"]))
        surf = np.concatenate([self.denorm(surf[:,:,k:k+1,0], "surface", self.input_vars["surface"][k], v=None, norm_method=self.norm_method) for k in range(surf.shape[2])], axis=2)
        for j,v in enumerate(self.input_lev_idxs):
            high[:,:,:,j] = np.concatenate([self.denorm(high[:,:,k:k+1,j], "high", self.input_vars["high"][k], v, self.norm_method) for k in range(high.shape[2])], axis=2)
        return surf, high


class IFSDataset(BaseDataset):
    '''
    从实时IFS数据文件加载不同变量，所有可能用到的变量和气压层设置在era5_variables.yaml文件，
    输入输出所需变量在train_config.yaml文件中设置；
    args:
        input_vars: list of string, variables to use
        output_vars: list of string, variables to forecast
    return:
        x_surf, x_high: array with 3 dims
    '''
    def __init__(self, 
                 input_vars, 
                 output_vars, 
                 bucket_cfg, 
                 norm_method="meanstd", 
                 use_dem=True, 
                 start_time="1980010100", 
                 ):
        super().__init__()
        self.variables = util.era5_stat()
        self.variables["surface"]["total_precipitation"]["meanstd"] = [5.27158476833704, 8.765997488569014]
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.norm_method = norm_method
        self.use_dem = use_dem
        self.data_source = DataSource(home_path="/home/lianghongli", bucket_cfg=bucket_cfg)
        era5_levels = self.variables["levels"]
        
        current_levels = [1000, 950, 850, 700, 600, 500, 450, 400, 300, 250, 200, 150,100]
        self.idx_950 = current_levels.index(950)
        self.idx_450 = current_levels.index(450)
        self.ifs_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
        self.start_time = [int(arrow.get(start_time, "YYYYMMDDHH").timestamp())]
        self.input_lev_idxs = [era5_levels.index(lv) for lv in current_levels]
        ## for interpolation of 925 and 50hPa
        self.alpha_950 = util.vertical_interp_coeff(1000, 925, 950)
        self.alpha_450 = util.vertical_interp_coeff(500, 400, 450)
        self.idx_1000 = self.ifs_levels.index(1000)
        self.idx_925 = self.ifs_levels.index(925)
        self.idx_500 = self.ifs_levels.index(500)
        self.idx_400 = self.ifs_levels.index(400)

        if use_dem:
            self.dem = self.load_dem()
            
    
    def level_interp(self, data, levels):
        if levels == (1000, 925):
            dinterp = data[..., self.idx_1000, :, :] + self.alpha_950 * (data[..., self.idx_925, :, :] - data[..., self.idx_1000, :, :])
        elif levels == (500, 400):
            dinterp = data[..., self.idx_500, :, :] + self.alpha_450 * (data[..., self.idx_400, :, :] - data[..., self.idx_500, :, :])
        else:
            raise ValueError(f"Unsupported level combination: {levels}")
        return dinterp 
    

    def reaorganize_upper(self, ifs_high):
        ifs_950 = self.level_interp(ifs_high, (1000, 925))
        ifs_high[...,1,:,:] = ifs_950 
        ifs_450 = self.level_interp(ifs_high, (500, 400))
        ifs_high = np.concatenate((ifs_high[..., :self.idx_450,:,:], ifs_450[:,None], ifs_high[...,self.idx_450:,:,:]), axis=-3)
        return ifs_high
    

    def get_bc(self, data: np.array):
        bc_bottom = data[..., -4:, :]
        bc_top = data[..., :4, :]
        bc_left = data[..., :4]
        bc_right = data[..., -4:]
        return bc_bottom, bc_left, bc_top, bc_right

        
    def __len__(self):
        return len(self.start_time)
    
    def __getitem__(self, idx):
        sample_start_time = self.start_time[idx]
        t = arrow.get(sample_start_time).format("YYYYMMDDHH")
        x_surf, x_high = self.process_ifs(t, lead_time=0)
        x_high = rearrange(x_high, 'v d h w -> (v d) h w')
        return (sample_start_time, x_surf, x_high)

    
    def process_ifs(self, init_time: str, 
                        lead_time: int=0, 
                        bc_only: bool=False, 
                        norm: bool=True, 
                        tp_only: bool=False, 
                        ):
        ifs_surf, ifs_high = self.data_source(src="ifs", start_time=init_time, 
                                    lead_time=lead_time, 
                                    args={"surf_var": ["2t", "msl", "10u", "10v", "tp"], 
                                            "upper_var": (["gh", "t", "q", "u", "v"], self.ifs_levels)})
        '''  interpolate pressure level: 
                                        1000, 925 --> 950
                                        400, 500 --> 450
        '''
        tp = ifs_surf[4]
        if tp_only:
            return tp
        ifs_surf = ifs_surf[:4]
        ifs_high = self.reaorganize_upper(ifs_high.copy())
        ''' normalization '''
        if norm:
            ifs_surf = np.concatenate([self.norm(ifs_surf[k:k+1], "surface", self.input_vars["surface"][k], v=None, norm_method=self.norm_method) for k in range(ifs_surf.shape[0])], axis=0)
            for j,v in enumerate(self.input_lev_idxs):
                ifs_high[:,j] = np.concatenate([self.norm(ifs_high[k:k+1, j], "high", self.input_vars["high"][k], v, norm_method=self.norm_method) for k in range(ifs_high.shape[0])], axis=0)

        x_surf = ifs_surf.astype(np.float32)
        x_high = ifs_high.astype(np.float32)
        
        if bc_only:
            bc_surf = self.get_bc(x_surf)
            bc_high = self.get_bc(x_high)
            return [bc_surf, bc_high]
        return [x_surf, x_high]
    
    def resume(self, output):
        if isinstance(output, tuple) or isinstance(output, list):
            surf = rearrange(output[0], 'b (n l) h w -> b n l h w', l=1)
            high = output[1]
        else:
            output = rearrange(output, 'b (n l) h w -> b n l h w', n=5)
            l = output.shape[2]
            surf, high = np.split(output, 1, dim=2)
        surf = rearrange(surf, 'b (t v) l h w -> b t v l h w', 
                            v=len(self.input_vars["surface"]))
        high = rearrange(high, 'b (t v) l h w -> b t v l h w', 
                            v=len(self.input_vars["high"]))
        surf = np.concatenate([self.denorm(surf[:,:,k:k+1,0], "surface", self.input_vars["surface"][k], v=None, norm_method=self.norm_method) for k in range(surf.shape[2])], axis=2)
        for j,v in enumerate(self.input_lev_idxs):
            high[:,:,:,j] = np.concatenate([self.denorm(high[:,:,k:k+1,j], "high", self.input_vars["high"][k], v, self.norm_method) for k in range(high.shape[2])], axis=2)
        return surf, high


@hydra.main(version_base=None, config_path="/home/lianghongli/projects/cncast-v1/configs", config_name="train.yaml")
def main(cfg):
    # ifs_dataset = IFSDataset(cfg.input, 
    #              cfg.output, 
    #              norm_method="meanstd", 
    #              use_dem=True, 
    #              bucket_cfg=cfg.buckets, 
    #              start_time="2024070100", )

    # print("datataset length:", len(ifs_dataset))
    # for i in range(len(ifs_dataset)):
    #     sample = ifs_dataset[i]
    #     print(len(sample))
    #     break
    cfg.dataload.start_time = "2021070100"
    era5_dataset = ERA5Dataset(
                input_vars=cfg.input, 
                output_vars=cfg.input, 
                bucket_cfg=cfg.buckets, 
                input_hist_hr=cfg.dataload.hist4in,
                forecast_step=cfg.dataload.fcst4out, 
                sample_interval=cfg.dataload.sample_interval, 
                norm_method=cfg.dataload.norm_method, 
                use_dem=cfg.dataload.use_dem, 
                start_time=cfg.dataload.start_time, 
                end_time=cfg.dataload.end_time, 
                cut_era5=cfg.dataload.cut_era5,
                target=cfg.dataload.target, 
                cmpa_frame=cfg.dataload.cmpa_frame,
                with_era5_tp=cfg.dataload.with_era5_tp, 
                resize_data=cfg.dataload.resize_data,
    )
    print("dataset length: ", len(era5_dataset))
    for i in range(len(era5_dataset)):
        data = era5_dataset[i]
        print(data[1].shape, len(data))
        break

if __name__=="__main__":
    main()

