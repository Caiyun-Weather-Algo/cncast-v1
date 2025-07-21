import xarray as xr
import numpy as np
import pandas as pd
import eccodes as ecc

# Define GRIB parameter mappings
SURF_PARAMS = {
    "2mt": {"shortName": "2t", "paramId": 167, "level": 2, "typeOfLevel": "surface", 
            "discipline": 0, "parameterCategory": 0, "parameterNumber": 0, 
            "typeOfFirstFixedSurface": 103, "scaledValueOfFirstFixedSurface": 2, "scaleFactorOfFirstFixedSurface": 0}, #, "units": "K"},              # 2m temperature
    "mslp": {"shortName": "msl", "paramId": 151, "level": 0, "typeOfLevel": "meanSea", #"units": "Pa", 
             "discipline": 0, "parameterCategory": 3, "parameterNumber": 0, "typeOfFirstFixedSurface": 101},           # Mean sea level pressure
    "10m_u_component_of_wind": {"shortName": "10u", "paramId": 165, "level": 10, "typeOfLevel": "surface", 
                                "discipline": 0, "parameterCategory": 2, "parameterNumber": 2, #"units": "ms-1", 
                                "typeOfFirstFixedSurface": 103, "scaledValueOfFirstFixedSurface": 10, "scaleFactorOfFirstFixedSurface": 0},  # 10m u-component of wind
    "10m_v_component_of_wind": {"shortName": "10v", "paramId": 166, "level": 10, "typeOfLevel": "surface", 
                                "discipline": 0, "parameterCategory": 2, "parameterNumber": 3, #"units": "ms-1", 
                                "typeOfFirstFixedSurface": 103, "scaledValueOfFirstFixedSurface": 10, "scaleFactorOfFirstFixedSurface": 0},  # 10m v-component of wind
    "total_precipitation": {"shortName": "prate", "paramId": 228, "level": 0, "typeOfLevel": "surface", 
                            "discipline": 0, "parameterCategory": 1, "parameterNumber": 7, #"units": "m", 
                            },  # Total precipitation (#! instant, instead of accumulated)
}

HIGH_PARAMS = {
    "geopotential": {"shortName": "z", "paramId": 129, #"units": "m2s-2", 
                     "discipline": 0, "parameterCategory": 3, "parameterNumber": 4},      # Geopotential
    "temperature": {"shortName": "t", "paramId": 130, #"units": "K", 
                    "discipline": 0, "parameterCategory": 0, "parameterNumber": 0},       # Temperature
    "specific_humidity": {"shortName": "q", "paramId": 133, #"units": "kgkg-1", 
                          "discipline": 0, "parameterCategory": 1, "parameterNumber": 0}, # Specific humidity
    "u_component_of_wind": {"shortName": "u", "paramId": 131, #"units": "ms-1", 
                            "discipline": 0, "parameterCategory": 2, "parameterNumber": 2},  # U component of wind
    "v_component_of_wind": {"shortName": "v", "paramId": 132, #"units": "ms-1", 
                            "discipline": 0, "parameterCategory": 2, "parameterNumber": 3},  # V component of wind
}

def to_varsplit_dataset(data, times, variables, lats, lons, levels=None):
    if levels is None:
        ds = xr.Dataset(
            {var: (['time', 'latitude', 'longitude'], data[:,i]) for i, var in enumerate(variables)},
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons,
        })
    else:
        ds = xr.Dataset(
            {var: (['time', 'level', 'latitude', 'longitude'], data[:,i]) for i, var in enumerate(variables)},
        coords={
            'time': times,
            'level': levels,
            'latitude': lats,
            'longitude': lons,
        })
    return ds


def ds2grib(surf_ds, high_ds, output_file, step):
    """
    Save two xarray.Dataset objects (surf_ds and high_ds) to a GRIB file.

    Parameters:
        surf_ds (xarray.Dataset): Dataset with shape [time, latitude, longitude].
        high_ds (xarray.Dataset): Dataset with shape [time, level, latitude, longitude].
        output_file (str): Path to the output GRIB file.
        ----GRIB file variable shortname lookup table: https://codes.ecmwf.int/grib/param-db/ --------
    """
    # Add required GRIB attributes to variables in surface dataset
    lats = surf_ds.latitude.values
    lons = surf_ds.longitude.values
    tstr = surf_ds.time.values[0].astype("datetime64[s]").item().strftime("%Y%m%d%H%M")
    ymd = int(tstr[:8])
    hr = int(tstr[8:10])
    with open(output_file, "wb") as f:
        for var in surf_ds.data_vars:
            if var in SURF_PARAMS:
                gid = ecc.codes_grib_new_from_samples("GRIB2")
                attrs = {
                    'gridType': 'regular_ll',
                    **SURF_PARAMS[var], 
                    #'values': surf_ds[var].values,
                    'stepType': 'instant',
                    'Ni': len(lons),
                    'Nj': len(lats),
                    'missingValue': 9999,
                    'dataType': 'fc',  # 2: forecast
                    'stepRange': step, 
                    'latitudeOfFirstGridPointInDegrees': lats[0],
                    'longitudeOfFirstGridPointInDegrees': lons[0],
                    'latitudeOfLastGridPointInDegrees': lats[-1],
                    'longitudeOfLastGridPointInDegrees': lons[-1],
                    'iDirectionIncrementInDegrees': np.float32(np.abs(lons[1]-lons[0])),
                    'jDirectionIncrementInDegrees': np.float32(np.abs(lats[1]-lats[0])),
                    'scanningMode': 0,
                    'dataDate': ymd, 
                    'dataTime': hr, 
                }
                att_arrays = {"values": surf_ds[var].values.astype(np.float32).flatten()}
                for key, value in attrs.items():
                    ecc.codes_set(gid, key, value)
                for key, value in att_arrays.items():
                    ecc.codes_set_array(gid, key, value)
                ecc.codes_write(gid, f)
                ecc.codes_release(gid)
                
        # Add required GRIB attributes to variables in upper-level dataset
        if high_ds is not None:
            for var in high_ds.data_vars:
                if var in HIGH_PARAMS:
                    # Create a new variable for each pressure level
                    for level in high_ds.level.values:
                        gid = ecc.codes_grib_new_from_samples("GRIB2")
                        level_data = high_ds[var].sel(level=level)
                        attrs = {
                            'gridType': 'regular_ll',
                            'typeOfLevel': 'isobaricInhPa',
                            'level': int(level),  # pressure level in hPa
                            'stepType': 'instant',
                            **HIGH_PARAMS[var],
                            'Ni': len(lons),
                            'Nj': len(lats),
                            'missingValue': 9999,
                            'dataType': 'fc',  # 2: forecast
                            'stepRange': step, 
                            'latitudeOfFirstGridPointInDegrees': lats[0],
                            'longitudeOfFirstGridPointInDegrees': lons[0],
                            'latitudeOfLastGridPointInDegrees': lats[-1],
                            'longitudeOfLastGridPointInDegrees': lons[-1],
                            'iDirectionIncrementInDegrees': np.float32(np.abs(lons[1]-lons[0])),
                            'jDirectionIncrementInDegrees': np.float32(np.abs(lats[1]-lats[0])),
                            'scanningMode': 0,
                            'dataDate': ymd, 
                            'dataTime': hr, 
                        }
                        for key, value in attrs.items():
                            ecc.codes_set(gid, key, value)
                        
                        ecc.codes_set_array(gid, "values", level_data.values.astype(np.float32).flatten()) 
                        ecc.codes_write(gid, f)
                        ecc.codes_release(gid)
        
    print(f"Datasets saved to {output_file}")
    return

