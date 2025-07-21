import matplotlib.pyplot as plt
import numpy as np

from src.utils import util

var_stats = util.era5_stat()
era5_levels = var_stats["levels"]
cncast_levels = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100,50)
lev_idxs = [era5_levels.index(j) for j in cncast_levels]

lev_idx_dict = {i:j for i,j in zip(cncast_levels, lev_idxs)}
var_stats['surface']["mslp"]["unit"] = "Pa"

ColorConfig = {
    "precip": 
        {"label": "precipitation(mm)", 
            # "levels": [0, 0.1, 1, 2, 5, 10, 15, 20],
            # "colors": ['#FFFFFF', '#A6F28F','#3DBA3D','#61BBFF','#0000FF','#FA00FA','#800040']
            "levels": [0, 0.1, 1, 2, 5, 10, 15, 20,25,30],
            "colors": ['#FFFFFF']+[plt.cm.RdYlGn(i) for i in np.linspace(0.85, 0, 11)], 
            "cmap": plt.cm.RdYlGn.reversed(), 
            "vmin": 0, 
            "vmax": 30,
            }, 
            }

for var in ["2mt", "mslp", "10m_u_component_of_wind", "10m_v_component_of_wind"]:
    ColorConfig[var] = {"label": f"{var}({var_stats['surface'][var]['unit']})", 
                        "levels": np.linspace(np.ceil(var_stats["surface"][var]["minmax"][1]), np.floor(var_stats["surface"][var]["minmax"][0]), 20),
                        "colors": [plt.cm.RdBu(i) for i in np.linspace(1, 0, 20)], 
                        "cmap": plt.cm.RdBu.reversed(), 
                        "vmin": var_stats["surface"][var]["minmax"][1], 
                        "vmax": var_stats["surface"][var]["minmax"][0],
                        }


for var in ["geopotential", "temperature", "specific_humidity","u_component_of_wind", "v_component_of_wind"]:
    ColorConfig[var] = {}
    for lev,idx in lev_idx_dict.items():
        if var=="specific_humidity":
            levels = np.linspace(var_stats["high"][var]["minmax"][1,idx]*0.9, var_stats["high"][var]["minmax"][0,idx]*0.9, 20)
        else:
            levels = np.linspace(np.ceil(var_stats["high"][var]["minmax"][1,idx]), np.floor(var_stats["high"][var]["minmax"][0,idx]), 20)
        ColorConfig[var][f"{lev}"] = {"label": f"{var}({var_stats['high'][var]['unit']})", 
                                          "levels": levels,
                                          "colors": [plt.cm.RdBu(i) for i in np.linspace(1, 0, 20)], 
                                          "cmap": plt.cm.RdBu.reversed(), 
                                          "vmin": var_stats["high"][var]["minmax"][1,idx], 
                                          "vmax": var_stats["high"][var]["minmax"][0,idx],
                                          }
