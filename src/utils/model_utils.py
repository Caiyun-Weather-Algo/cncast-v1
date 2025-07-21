import torch
import arrow, time
import xarray as xr
import numpy as np
from einops import rearrange

from src.utils import util
from src.models.create_model import load_model
from src.utils.util import totensor
from src.models.DiT.diffusion import create_diffusion
from src.utils.metric_utils import pm_enmax


def load_ckpt(lead_time, cfg, target: str = "non-tp"):
    model_path = cfg.directory.model_path

    if target=="non-tp":
        model = load_model(cfg.model.name, cfg)
        ckpt_path = f"{model_path}/swin_transformer_3d_cncast_v1_{lead_time}h.pth.tar"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
    elif target=="tp":
        model = load_model("dit", cfg) 
        ckpt_path = f"{model_path}/precip_diagnosis_dit_ckpt.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["ema"])
    elif target=="tp-sr":
        model = load_model("swinir", cfg) 
        ckpt_path = f"{model_path}/swinir_tp_sr.pth.tar"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def bc_att_ifs(ts, lead_hr, dataset, device):
    ''' time resolution of ifs is 3h, so we need to convert lead_hr to 3h '''
    if lead_hr%3>0:
        lead_hr = lead_hr + 3 - lead_hr%3
    t = arrow.get(ts).format("YYYYMMDDHH")
    xbc = dataset.process_ifs(init_time=t, lead_time=lead_hr, bc_only=True)
    xbc = [[totensor(x[None,:], device) for x in bc_list] for bc_list in xbc]
    return xbc


def bc_att_era5(timestamp, j, dataset, device):
    t = arrow.get(ts).shift(hours=+j).format("YYYYMMDDHH")
    xbc = dataset.get_bc_from_file(t)
    xbc = [[totensor(x[None,:], device) for x in bc_list] for bc_list in xbc]
    return xbc


def affine_model_idx(models, j, leads=[24,6,3,1]):
    for x in leads:
        if j%x==0:
            model = models[f"lead_{x}h"]
            x_idx = j-x
            break
    return model, x_idx


def iter_predict(x_surf, x_high, ts, cfg, static):
    device = x_surf.device
    lats, lons, levels, dem, forecast_day, models, dataset, data_source = static
            
    with torch.no_grad():
        fcst_surf = util.to_dataset(x_surf.cpu().numpy(), [ts], cfg.input.surface, lats, lons, levels=None)
        fcst_high = util.to_dataset(rearrange(x_high.cpu().numpy(), 'b (c v) h w -> b c v h w', c=5), [ts], cfg.input.high, lats, lons, levels=levels)
        for j in range(1, forecast_day*24+1):
            print(f"step {j}, start predicting")
            model, x_idx = affine_model_idx(models, j)
            ## get bc data
            if data_source=="ifs":
                xbc = bc_att_ifs(ts, j, dataset, device)
            else:
                xbc = bc_att_era5(ts, j, dataset, device)

            x_surf = torch.cat((totensor(fcst_surf['surface'][x_idx:(x_idx+1)].values, device), dem), dim=1)
            x_high = totensor(fcst_high['upper'][x_idx:(x_idx+1)].values, device)
            output = model([x_surf, x_high], xbc)            
            output = [output[0].cpu().numpy(), output[1].cpu().numpy()]
            output[1] = rearrange(output[1], 'b (c v) h w -> b c v h w', c=5)

            surf = util.to_dataset(output[0], [ts+3600*j], cfg.input.surface, lats, lons, levels=None)
            high = util.to_dataset(output[1], [ts+3600*j], cfg.input.high, lats, lons, levels=levels)
            fcst_surf = xr.concat([fcst_surf, surf], dim="time")
            fcst_high = xr.concat([fcst_high, high], dim="time")

    fcst_surf = fcst_surf['surface'][1:]
    fcst_high = fcst_high['upper'][1:]
    return fcst_surf, fcst_high


def iter_pred_tp(tp_x_surf, tp_x_high, ts, dataset, models, static_args, tp_high_res=False):
    tp_model, tp_sr_model = models
    forecast_day, lats, lons, lats0p6, lons0p6, device = static_args
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    diffusion = create_diffusion(str(200))
    def gen_samples(diff, model, x, model_kwargs, device):
        samples = diff.p_sample_loop(
                model.forward, x.shape, x, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        return samples[:,:1]
    
    interp_idxs = util.nearest_interp_idxs(src_shape=(241, 281), dst_shape=(256, 256))
    inverse_interp_idxs = util.nearest_interp_idxs(src_shape=(256, 256), dst_shape=(241, 281))
    if tp_high_res:
        batch_size = 2
    else:
        batch_size = 24
    
    for p in range(1, 24*forecast_day+1, batch_size):
        xin = [tp_x_surf[p-1:p+batch_size-1], rearrange(tp_x_high[p-1:p+batch_size-1], 't v l h w -> t (v l) h w')]
        xin = np.concatenate(xin, axis=1)
        xin = torch.tensor(util.resize(xin, interp_idxs), dtype=torch.float32).to(device)
        z = torch.randn(xin.shape[0], 1, *xin.shape[-2:]).to(device)
        
        tp_pred = gen_samples(diffusion, tp_model, z, dict(cond=xin), device) #.detach().cpu().numpy() 

        tp_pred = dataset.denorm(tp_pred, "surface", "total_precipitation", v=None, norm_method="meanstd").detach().cpu().numpy()
        
        if tp_high_res:
            sr_in = torch.tensor(tp_pred, dtype=torch.float32).to(device)
        ## convert from dbz to precipitation with unit mm/h
        tp_pred = util.dbz2tp(tp_pred)
        tp_pred = tp_pred*(tp_pred>=0.1)
        tp_pred = util.resize(tp_pred, inverse_interp_idxs)
        tp_pred = util.to_dataset(tp_pred, [ts+p_*3600 for p_ in range(p,p+batch_size)], ["tp"], lats, lons, levels=None) 
        ## super resolution
        if tp_high_res:
            with torch.no_grad():
                tp_pred_06 = tp_sr_model(sr_in/55).detach().cpu().numpy()
            ## convert from dbz to precipitation with unit mm/h
            tp_pred_06 = util.dbz2tp(tp_pred_06*55)
            tp_pred_06 = tp_pred_06*(tp_pred_06>=0.1)
            tp_pred_06 = util.to_dataset(tp_pred_06, [ts+p_*3600 for p_ in range(p,p+batch_size)], ["tp"], lats0p6, lons0p6, levels=None)
        
        if p==1:
            tp_pred_5d = tp_pred
            if tp_high_res:
                tp_pred_5d_sr = tp_pred_06
            else:
                tp_pred_5d_sr = None
        else:
            tp_pred_5d = xr.concat([tp_pred_5d, tp_pred], dim="time")
            if tp_high_res:
                tp_pred_5d_sr = xr.concat([tp_pred_5d_sr, tp_pred_06], dim="time")
            else:
                tp_pred_5d_sr = None
    return tp_pred_5d, tp_pred_5d_sr
