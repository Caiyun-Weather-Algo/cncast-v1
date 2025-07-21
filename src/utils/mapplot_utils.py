import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cv2
import json

COLOR_CONFIG = json.load(open("configs/colormap.json"))


def color_convert(data):
    cmap = matplotlib.colormaps.get_cmap('gist_ncar')
    cmp_min, cmp_max = [-5.22, 214.06]
    data_min, data_max = [0, 100]
    data[data<data_min] = data_min
    data[data>data_max] = data_max

    data = (data - cmp_min)/(cmp_max - cmp_min)
    data = np.log10(data * 90 + 1) / np.log10(91)
    thres = (0.1 - cmp_min) / (cmp_max - cmp_min)
    thres = np.log10(thres * 90 + 1) / np.log10(91)
    show_area = data > thres

    # Convert to color
    rgba = cmap(data)
    rgba[:, :, -1] = rgba[:, :, -1] * 0.4
    rgba *= show_area[:, :, None].astype(float)
    rgba = (rgba * 255).astype(np.uint8)
    return rgba


def apply_colormap(data, color=cv2.COLORMAP_JET):
    """
    Apply a colormap to float32 data.
    
    Parameters:
        data (np.ndarray): Input data (float32)
        colormap: cv2 colormap (e.g., cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS)
        cv2 colormap list:
        #COLORMAPS = {
                'AUTUMN': cv2.COLORMAP_AUTUMN,     # Red-Yellow
                'BONE': cv2.COLORMAP_BONE,         # Black-White, a variant of gray
                'JET': cv2.COLORMAP_JET,           # Blue-Cyan-Yellow-Red
                'WINTER': cv2.COLORMAP_WINTER,     # Blue-Green
                'RAINBOW': cv2.COLORMAP_RAINBOW,   # Violet-Blue-Green-Yellow-Red
                'OCEAN': cv2.COLORMAP_OCEAN,       # Deep blue-white
                'SUMMER': cv2.COLORMAP_SUMMER,     # Green-Yellow
                'SPRING': cv2.COLORMAP_SPRING,     # Pink-Yellow
                'COOL': cv2.COLORMAP_COOL,         # Cyan-Magenta
                'HSV': cv2.COLORMAP_HSV,           # Red-Yellow-Green-Cyan-Blue-Magenta
                'PINK': cv2.COLORMAP_PINK,         # Black-Pink-White
                'HOT': cv2.COLORMAP_HOT,           # Black-Red-Yellow-White
                'PARULA': cv2.COLORMAP_PARULA,     # Blue-Green-Yellow
                'MAGMA': cv2.COLORMAP_MAGMA,       # Black-Purple-Yellow
                'INFERNO': cv2.COLORMAP_INFERNO,   # Black-Red-Yellow
                'PLASMA': cv2.COLORMAP_PLASMA,     # Purple-Red-Yellow
                'VIRIDIS': cv2.COLORMAP_VIRIDIS,   # Purple-Green-Yellow
                'CIVIDIS': cv2.COLORMAP_CIVIDIS,   # Blue-Yellow
                'TWILIGHT': cv2.COLORMAP_TWILIGHT, # Purple-Blue-Green-Yellow
                'TWILIGHT_SHIFTED': cv2.COLORMAP_TWILIGHT_SHIFTED,  # Shifted version of TWILIGHT
                'TURBO': cv2.COLORMAP_TURBO,       # Improved rainbow colormap
                'DEEPGREEN': cv2.COLORMAP_DEEPGREEN # Variations of green
            }
    Returns:
        np.ndarray: BGR image with colormap applied
    """
    return cv2.applyColorMap(data, color)


def apply_cmap_nafp(data, variable):
    """
    Apply a colormap to float32 data.
    
    Parameters:
        data (np.ndarray): normalized input data (float32)
        variable (str): variable name
    Returns:
        np.ndarray: BGR image with colormap applied
    """
    data = data - 273.15  ## convert to Celsius degree
    value_area = COLOR_CONFIG[variable]["value area"]
    data[data<np.min([value_area[0][0],value_area[0][1]])] = np.min([value_area[0][0],value_area[0][1]])
    data[data>np.max([value_area[0][0],value_area[0][1]])] = np.max([value_area[0][0],value_area[0][1]])
    data = (data - value_area[1][0])/(value_area[1][1] - value_area[1][0])
    show_area = np.ones(data.shape, dtype = np.uint8)

    cmap = cm.get_cmap(COLOR_CONFIG[variable]["cmap"], 1024)
    rgba = cmap(data)
    rgba[:,:,-1] = rgba[:,:,-1]*0.4  ## 0.4 is the opacity of the color
    rgba = np.uint8(np.round(rgba*255))
    return rgba*show_area[:,:,None]


def map_proj_n(x, cbar_args, figname, title, fig_layout, bbox_list, share_cbar=True,suptitle=None, color_discretized=True):
    projection = ccrs.PlateCarree()
    lons, lats = [], []
    for i in range(len(bbox_list)):
        lons.append(np.linspace(bbox_list[i][1], bbox_list[i][3], x[i].shape[1]))
        lats.append(np.linspace(bbox_list[i][2], bbox_list[i][0], x[i].shape[0]))

    # 创建地图和子图
    figsize = (8*fig_layout[1], 8*fig_layout[0])
    fig, axs = plt.subplots(fig_layout[0], fig_layout[1], figsize=figsize, facecolor="white",
                        subplot_kw={'projection': projection}, dpi=400)
    axs = axs.flatten()
    # Plot data on each subplot
    ims = []
    if cbar_args["vmin"] is None:
        cbar_args["vmin"] = np.min(x)
    if cbar_args["vmax"] is None:
        cbar_args["vmax"] = np.max(x)
    for i in range(len(bbox_list)):
        ax = axs[i]
        if color_discretized:
            im = ax.contourf(lons[i], lats[i], x[i], levels=cbar_args["levels"], colors=cbar_args["colors"], extend='both', alpha=0.8)
        else:
            im = ax.pcolormesh(lons[i], lats[i], x[i], cmap=cbar_args["cmap"], vmin=cbar_args["vmin"], vmax=cbar_args["vmax"], alpha=0.8, shading="auto")
        ax.set_title(title[i])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.coastlines()
        ims.append(im)
        if not share_cbar:
            cbar = fig.colorbar(im, extend="both", label=cbar_args["label"][i], orientation="horizontal")
            cbar.set_label(cbar_args["label"][i], loc="right")

    for i,ax in enumerate(axs):
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                        linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        if i==0:
            gl.right_labels = False
        if i>0:
            gl.left_labels = False
        gl.xlocator = plt.FixedLocator(list(np.arange(lons[0][0], lons[0][-1]+1, 10)))  # Customized longitude locations
        gl.ylocator = plt.FixedLocator(list(np.arange(lats[0][0], lats[0][-1]-1, -10)))    # Customized latitude locations
        # gl.xformatter = plt.FuncFormatter(lambda x, pos: '{:.0f}°E'.format(x))  # Customized longitude labels
        # gl.yformatter = plt.FuncFormatter(lambda y, pos: '{:.0f}°N'.format(y))  # Customized latitude labels
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
    # 添加颜色条
    plt.tight_layout()
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=14)
    if share_cbar:
        fig.subplots_adjust(bottom=0.12)
        cbar_ax = fig.add_axes([0.15, 0.07, 0.7, 0.03]) # left, bottom, width, height
        cbar = fig.colorbar(im, cax=cbar_ax, extend="both", label=cbar_args["label"], orientation="horizontal")
        # cbar.set_label(cbar_args["label"], loc="right")
        cbar.set_label(cbar_args["label"], loc="right")

    axs[0].set_xlabel('Longitude')
    axs[1].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    plt.savefig(figname)
    plt.close()
    return 0
