import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# from evaluation import final_score, utils
from utils import util
import hydra
from matplotlib.gridspec import GridSpec


def plot_subfigure(scores, x, ax, xlabel, ylabel, title, text, level="surf"):
    if "q" in title:
        scores = [scores[0]*1000, scores[1]*1000]
    ax.plot(x, scores[0], "r-", linewidth=1, label="pangu")
    ax.plot(x, scores[1], "b-", linewidth=1, label="cncast")
    # if text in ["a", "c", "e", "g", "i"]:
    ax.set_title(title, fontsize=14)
    if (level=="surf" and text in ["c", "d"]) or level=="high":
        ax.set_xlabel("Lead time(h)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(linestyle="--", color="gray", alpha=0.7)
    if level=="surf":
        legend_subfigs = ["b", "d"]
    else:
        legend_subfigs = ["c", "e"]
    if text in legend_subfigs:
        if "ACC" in ylabel:
            ax.legend(loc="lower left", fontsize=12)
        else:
            ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim([1, x[-1]])
    ax.set_xticks(list(range(12,121,12)))
    ax.set_xticklabels(list(range(12,121,12)))
    # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)
    plt.text(0.02, 0.98, text, transform=ax.transAxes, 
            fontsize=12, fontweight="bold", va="top", ha="left")


@hydra.main(version_base=None, config_path="/home/lianghongli/caiyun-algo-misc/caiyun-era5-pretrain/conf", config_name="train_40yr_config.yaml")
def main(cfg):
    fpath = Path("/home/lianghongli/weather-caiyun/lianghl/pretrain_results/pred_recurse/pangu_july_score/")
    fs_pangu_score = [fpath/f"score_pangu_step{k}.npz" for k in range(1,121)]
    # print(fs_pangu_score)
    # exit()
    surf_scores = {}
    high_scores = {}
    # units = {"mean_sea_level_pressure_0": "$Pa$", "10m_u_component_of_wind_1": "$m/s$", "10m_v_component_of_wind_2": "$m/s$", "2m_temperature_3": "$K$"}
    units = {"mean_sea_level_pressure": "$Pa$", "10m_u_component_of_wind": "$m/s$", "10m_v_component_of_wind": "$m/s$", "2m_temperature": "$K$"}
    titles = {"mean_sea_level_pressure": "mslp", "10m_u_component_of_wind": "10u", "10m_v_component_of_wind": "10v", "2m_temperature": "2mt", "geopotential": "z", "temperature": "t", "u_component_of_wind": "u", "v_component_of_wind": "v", "specific_humidity": "q"}
    for i,f in enumerate(fs_pangu_score):
        data = np.load(f, allow_pickle=True)
        score = data['scores'].tolist()

        for k, v in score["surf"].items():
            if i==0:
                surf_scores[k] = np.array(v).astype(np.float32).reshape(-1,1)
            else:
                surf_scores[k] = np.concatenate((surf_scores[k], np.array(v).astype(np.float32).reshape(-1,1)), axis=1)
                # surf_scores[k] = np.concatenate((surf_scores[k], np.array(v).astype(np.float32).reshape(-1,1)), axis=1)
        for k, v in score["high"].items():
            if i==0:
                high_scores[k] = np.array(v).astype(np.float32).reshape(-1,1)
                # high_scores[k] = np.stack(v, axis=0).astype(np.float32)[:,:,None]
            else:
                high_scores[k] = np.concatenate((high_scores[k], np.array(v).astype(np.float32).reshape(-1,1)), axis=1)
                # high_scores[k] = np.concatenate((high_scores[k], np.stack(v, axis=0).astype(np.float32)[:,:,None]), axis=2)
    
    fig_prefix = Path("/home/lianghongli/weather-caiyun/lianghl/pretrain_results/pred_recurse_bc/score_comparison_pangu-cncast")
    fig_prefix.mkdir(exist_ok=True)
    month = 7
    cn_score_july = np.load("/home/lianghongli/weather-caiyun/lianghl/pretrain_results/pred_recurse_bc/cncast-v1_scores/month7_scores_china_5d.npz", allow_pickle=True)["scores"].tolist()
    cn_score_surf = cn_score_july['surf']
    cn_score_high = cn_score_july['high']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes = axes.flatten()
    labels = "abcdefgh"
    x = list(range(1,121))
    # x = list(range(6, 121, 6))
    print(surf_scores.keys(), cn_score_surf.keys())
    idx = 0
    for k, v in surf_scores.items():
        print(k)
        if k=="mean_sea_level_pressure_0":
            kk = "mslp_1"
        elif k=="10m_u_component_of_wind_1":
            kk = "10m_u_component_of_wind_2"
        elif k=="10m_v_component_of_wind_2":
            kk = "10m_v_component_of_wind_3"
        elif k=="2m_temperature_3":
            kk = "2mt_0"
        
        # kk = k
        rmses = [surf_scores[k][0], cn_score_surf[kk][0]]
        # accs = [surf_scores[k][1], cn_score_surf[k][1]]
        accs = [surf_scores[k][1]*100, cn_score_surf[kk][1]*100]
        var_name = "_".join(k.split("_")[:-1])
        
        plot_subfigure(rmses, x, axes[idx], "Lead time(hs)", f"RMSE({units[var_name]})", titles[var_name], labels[idx],level="surf")
        # plot_subfigure(accs, x, axes[idx+4], "Lead time(hs)", "ACC(%)", var_name, labels[idx*2+1],level="surf")
        idx += 1
    
    plt.subplots_adjust(left=0.07, right=0.96, bottom=0.07, top=0.95)
    plt.savefig(f"{fig_prefix}/pangu_vs_cn_month{month}_surface_rmseonly.pdf")
    ## upper-air scores
    units = {"geopotential": "$m^2/s^2$", "temperature": "$K$", "u_component_of_wind": "$m/s$", "v_component_of_wind": "$m/s$", "specific_humidity": "$g/kg$"}
    fig = plt.figure(figsize=(20, 10), dpi=300)
    gs = GridSpec(2, 6, figure=fig)  # 2 rows, 6 columns (to center the bottom row)
    # Top row: 3 subplots (span columns 1-2, 3-4, 5-6)
    ax1 = fig.add_subplot(gs[0, :2])  # Row 0, Columns 0-1
    ax2 = fig.add_subplot(gs[0, 2:4]) # Row 0, Columns 2-3
    ax3 = fig.add_subplot(gs[0, 4:])  # Row 0, Columns 4-5

    # Bottom row: 2 subplots (centered by spanning columns 2-3 and 4-5)
    ax4 = fig.add_subplot(gs[1, 1:3]) # Row 1, Columns 1-2 (shifted right)
    ax5 = fig.add_subplot(gs[1, 3:5]) # Row 1, Columns 3-4 (shifted left)
    # axes[1,2].remove()
    # axes = axes.flatten()
    axes = [ax1, ax2, ax3, ax4, ax5]
    labels = "abcdefghij"
    # fig.suptitle("RMSE Scores by Lead Time", fontsize=16)
    idx = 0
    level = "500"
    idx_level = cfg.input.levels.index(int(level))
    idx_level_pangu = [1000,850,700,600,500,400,300,250,200,150,100].index(int(level))
    # print(high_scores.keys(), cn_score_high.keys())
    for k, v in high_scores.items():
        if "500" not in k:
            continue
        if "temperature" not in k and "geopotential" not in k:
            name, level = "_".join(k.split("_")[:-1]), k.split("_")[-1]
        else:
            name, level = k.split("_")[0], int(k.split("_")[-1])
        # name = k
        print(name, level)
        # if level in k:
        rmses = [high_scores[k][0], cn_score_high[k][0]]
        # accs = [high_scores[k][1]*100, cn_score_high[name][1][:,idx_level]*100]
        ##### IFS inited
        # rmses = [high_scores[k][0,idx_level_pangu], cn_score_high[name][0][:,idx_level]]
        # accs = [high_scores[k][1,idx_level_pangu]*100, cn_score_high[name][1][:,idx_level]*100]
        # var_name = "_".join(k.split("_")[:-1])
        plot_subfigure(rmses, x, axes[idx], "Lead time(hs)", f"RMSE({units[name]})", f"{titles[name]}{level}", labels[idx], level="high")
        # plot_subfigure(accs, x, axes[idx+5], "Lead time(hs)", "ACC(%)", k, labels[idx*2+1], level="high")
        idx += 1
    
    plt.subplots_adjust(left=0.04, right=0.985, bottom=0.07, top=0.95)
    plt.savefig(f"{fig_prefix}/pangu_vs_cn_month{month}_upper_{level}hPa_rmseonly.pdf")

if __name__ == '__main__':
    main()
