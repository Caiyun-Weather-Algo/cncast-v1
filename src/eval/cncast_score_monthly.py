import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from matplotlib.gridspec import GridSpec
from src.eval.eval_utils import plot_subfigure


@hydra.main(version_base=None, config_path="./configs", config_name="online.yaml")
def main(cfg):
    # Load all monthly scores
    base_path = Path(cfg.directory.cncast_v1_scores_path)
    figpath = Path(cfg.directory.save_path)
    surf_scores = {}
    high_scores = {}
    level = 500
    levels = cfg.input.levels
    idx = levels.index(level)
    
    # Surface variables and their units
    surf_vars = cfg.input.surface
    high_vars = cfg.input.high
    titles = {"mslp": "mslp", "10m_u_component_of_wind": "10u", "10m_v_component_of_wind": "10v", "2mt": "2mt", "geopotential": "z", "temperature": "t", "u_component_of_wind": "u", "v_component_of_wind": "v", "specific_humidity": "q"}
    surf_units = {"mslp": "$Pa$", "10m_u_component_of_wind": "$m/s$", "10m_v_component_of_wind": "$m/s$", "2mt": "$K$"}
    high_units = {"geopotential": "$m^2/s^2$", "temperature": "$K$", "u_component_of_wind": "$m/s$", "v_component_of_wind": "$m/s$", "specific_humidity": "$g/kg$"}

    # Load scores for each month
    step = [0,11,23,47]
    for month in range(1, 13):
        print(month)
        fname = base_path / f"month{month}_scores_china_5d_averaged.npz"
        data = np.load(fname, allow_pickle=True)
        scores = data['scores'].tolist()
        
        # Process surface variables
        for var in surf_vars:
            if month == 1:
                surf_scores[var] = [scores['surf'][var][0][step]]  # RMSE
                surf_scores[f"{var}_acc"] = [scores['surf'][var][1][step]]  # ACC
            else:
                surf_scores[var].append(scores['surf'][var][0][step])
                surf_scores[f"{var}_acc"].append(scores['surf'][var][1][step])
            
        # Process upper air variables
        for var in high_vars:
            if month == 1:
                high_scores[var] = [scores['high'][var][0][:,idx][step]]  # RMSE
                high_scores[f"{var}_acc"] = [scores['high'][var][1][:,idx][step]]  # ACC
            else:
                high_scores[var].append(scores['high'][var][0][:,idx][step])
                high_scores[f"{var}_acc"].append(scores['high'][var][1][:,idx][step])

    x = list(range(1, 13))  # Lead times from January to December
    # Plot surface variables
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes = axes.flatten()
    labels = "abcdefgh"
    
    for idx, var in enumerate(surf_vars):
        # Plot RMSE
        plot_subfigure(surf_scores[var], x, axes[idx], 
                      "month", f"RMSE ({surf_units[var]})", 
                      titles[var], labels[idx])
        
    
    plt.subplots_adjust(left=0.07, right=0.95, bottom=0.06, top=0.95)
    plt.savefig(figpath/"surface_scores_seasonal_all-step_rmse.pdf")
    
    # Plot upper air variables at 500hPa
    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(2, 6, figure=fig)  # 2 rows, 6 columns (to center the bottom row)
    # Top row: 3 subplots (span columns 1-2, 3-4, 5-6)
    ax1 = fig.add_subplot(gs[0, :2])  # Row 0, Columns 0-1
    ax2 = fig.add_subplot(gs[0, 2:4]) # Row 0, Columns 2-3
    ax3 = fig.add_subplot(gs[0, 4:])  # Row 0, Columns 4-5

    # Bottom row: 2 subplots (centered by spanning columns 2-3 and 4-5)
    ax4 = fig.add_subplot(gs[1, 1:3]) # Row 1, Columns 1-2 (shifted right)
    ax5 = fig.add_subplot(gs[1, 3:5]) # Row 1, Columns 3-4 (shifted left)
    axes = [ax1, ax2, ax3, ax4, ax5]
    labels = "abcdefghij"
    
    for idx, var in enumerate(high_vars):
        if "specific" in var:
            high_scores[var] = [high_scores[var][j]*1000 for j in range(len(high_scores[var]))]
        
        # Plot RMSE
        plot_subfigure(high_scores[var], x, axes[idx], 
                      "month", f"RMSE ({high_units[var]})", 
                      f"{titles[var]}{level}", labels[idx])
        

    plt.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.95)
    plt.savefig(figpath/"upper_air_scores_seasonal_all-step_rmse.pdf")

if __name__ == "__main__":
    main()
