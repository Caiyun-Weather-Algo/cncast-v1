import numpy as np
import pickle
import xarray as xr 
import torch
import earthkit.data as ekt
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def era5_stat():
    variables = pickle.load(open("./share/era5_stats_1980-2018.pkl", "rb"))
    return variables


def nearest_interp_idxs(src_shape=(241, 281), dst_shape=(256, 256)):
    facs = np.array(dst_shape, dtype=np.float16)/np.array(src_shape, dtype=np.float16)
    idxs = np.indices(dst_shape)
    row_idxs = np.round(idxs[0][:,0]/facs[0]).astype(int)
    col_idxs = np.round(idxs[1][0,:]/facs[1]).astype(int)
    return (row_idxs, col_idxs)


def resize(data, idxs):
    interped = data[..., idxs[0], :][..., idxs[1]]
    return interped


def ACC(y, pred, yavg, lat_weight):
    '''
    y, pred are long-term-mean-subtracted values
    '''
    # Calculate anomaly correlation coefficient (ACC)
    # ACC is calculated as the correlation between predicted and observed anomalies
    # from their respective climatological means, weighted by latitude
    x = np.nansum((y-yavg)*(pred-yavg)*lat_weight, axis=(-2,-1))
    d = np.nansum((y-yavg)**2*lat_weight, axis=(-2,-1)) * np.nansum((pred-yavg)**2*lat_weight, axis=(-2,-1))
    return x/np.sqrt(d)


def RMSE(y, pred, lat_weight):
    x = np.nansum((y-pred)**2 * lat_weight, axis=(-2,-1))
    size = (~np.isnan(y)).sum(axis=(-2,-1))
    rmse = np.sqrt(x/size)
    return rmse

def static4eval(lev_idxs):
    surf_avg = np.load("./share/surf_var_mean_pix-lev_38ys.npz")["data"]  # for ACC calculation
    high_avg = np.load("./share/high_var_mean_pix-lev_38ys.npz")["data"]  # for ACC calculation
    high_avg = np.concatenate((high_avg[2:3], high_avg[:1], high_avg[1:2], high_avg[3:]))
    high_avg = high_avg[:, lev_idxs]
    return surf_avg, high_avg


def tp2dbz(x, dmin=2, dmax=55):
    x = np.where(x<0.04, 0.04, x)
    y = 10 * np.log10(200 * np.power(x, 1.6))
    y = np.where(y<dmin, dmin, y)
    y = np.where(y>dmax, dmax, y)
    return y


def dbz2tp(x):
    y = np.power(10**(x/10)/200, 1/1.6)
    y = np.where(y<0.04, 0, y)
    return y


def to_dataset(data, times, variables, lats, lons, levels=None):
    if levels is None:
        ds = xr.Dataset(
        {
            'surface': (['time', 'variable', 'latitude', 'longitude'], data), 
        },
        coords={
            'time': times,
            'variable': variables, 
            'latitude': lats,
            'longitude': lons,
        })
    else:
        ds = xr.Dataset(
        {
            'upper': (['time', 'variable', 'level', 'latitude', 'longitude'], data), 
        },
        coords={
            'time': times,
            'variable': variables,
            'level': levels,
            'latitude': lats,
            'longitude': lons,
        })
    return ds


def totensor(data, device):
    return torch.tensor(data, dtype=torch.float32, device=device)


def projection(data: np.array, src_proj: str = "EPSG:4326", ds_proj: str = "EPSG:3857"):
    """
    Reproject data from source projection to destination projection.
    
    Args:
        src_proj (str): Source projection (e.g., 'EPSG:4326')
        ds_proj (str): Destination projection (e.g., 'EPSG:3857')
        data (numpy.ndarray): Input data with shape [variables, height, width]
        
    Returns:
        numpy.ndarray: Reprojected data with the same number of variables but potentially different dimensions
    """
    # Get the number of variables and dimensions
    n_vars, height, width = data.shape
    # Create a list to store reprojected data for each variable
    reprojected_data = []
    # Define the source metadata
    src_meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,  # Process one band at a time
        'dtype': data.dtype,
        'crs': src_proj
    }
    
    # Calculate the transform for the source data
    # Set transform based on the data bounds [bottom_lat, left_lon, top_lat, right_lon]
    if src_proj == 'EPSG:4326':
        src_meta['transform'] = rasterio.transform.from_bounds(
            70, 0, 140, 60, width, height
        )
    else:
        # For other projections, you might need to adjust this
        raise ValueError(f"Source projection {src_proj} not supported for automatic bounds calculation")
    
    # Calculate the destination transform and dimensions, project to the same size
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_meta['crs'], ds_proj, width, height, **{
            'left': src_meta['transform'].c,
            'bottom': src_meta['transform'].f + src_meta['transform'].e * height,
            'right': src_meta['transform'].c + src_meta['transform'].a * width,
            'top': src_meta['transform'].f
        }
    )
    # Update metadata for destination
    dst_meta = src_meta.copy()
    dst_meta.update({
        'crs': ds_proj,
        'transform': dst_transform,
        'width': dst_width,
        'height': dst_height
    })
    
    # Process each variable separately
    for i in range(n_vars):
        # Create destination array
        dst_array = np.zeros((dst_height, dst_width), dtype=data.dtype)
        # Reproject the data
        reproject(
            source=data[i],
            destination=dst_array,
            src_transform=src_meta['transform'],
            src_crs=src_meta['crs'],
            dst_transform=dst_transform,
            dst_crs=ds_proj,
            resampling=Resampling.bilinear
        )
        reprojected_data.append(dst_array)
    
    # Stack the reprojected data back into a single array
    return np.stack(reprojected_data, axis=0)


def vertical_interp_coeff(p1, p2, p_target):
    """
    在气压对数坐标下，通过线性插值计算目标气压层的温度和位势高度。
    
    参数：
        p1, p2    : 已知层的气压值（单位：hPa），如 925 和 1000
        p_target  : 目标气压层（单位：hPa），如 950
        
    返回：
        alpha: 插值权重
    """
    # 转换为自然对数坐标
    ln_p1 = np.log(p1)
    ln_p2 = np.log(p2)
    ln_p_target = np.log(p_target)
    
    # 计算插值权重
    alpha = (ln_p_target - ln_p1) / (ln_p2 - ln_p1)
    return alpha

