import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import hydra
from matplotlib.gridspec import GridSpec


def plot_subfigure(scores, x, ax, xlabel, ylabel, title, text, level="surf", labels=["pangu", "cncast"]):
    for i, score in enumerate(scores):
        ax.plot(x, score, f"-o", linewidth=1, label=labels[i])

    # if text in ["a", "c", "e", "g", "i"]:
    ax.set_title(title, fontsize=14)
    if (level=="surf" and text in ["c", "d"]) or level=="high":
        ax.set_xlabel("Lead time(h)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(linestyle="--", color="gray", alpha=0.7)
    # ax.set_xticks([24,48,72,96,120])
    ax.set_xticks([6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120])
    ax.set_xticklabels(["6", "12", "18", "24", "30", "36", "42", "48", "54", "60", "66", "72", "78", "84", "90", "96", "102", "108", "114", "120"])
    # ax.set_xticklabels(["24", "48", "72", "96", "120"])
    if level=="surf":
        legend_subfigs = ["b", "d"]
    else:
        legend_subfigs = ["c", "e"]
    if text in legend_subfigs:
        if "ACC" in ylabel:
            ax.legend(loc="lower left", fontsize=12)
        else:
            ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim([6, x[-1]])
    # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)
    plt.text(0.02, 0.98, text, transform=ax.transAxes, 
            fontsize=12, fontweight="bold", va="top", ha="left")


@hydra.main(version_base=None, config_path="./configs", config_name="online.yaml")
def main(cfg):
    center = 241-60*2
    fpath = Path(f"{cfg.directory.base_path}/pangu_july_score")
    fs_pangu_score = [fpath/f"ifs-init_score_pangu_step{k}_center{center}.npz" for k in range(6,121,6)]
    
    use_ifs = True
    use_aifs = True
    use_aifs_bound = True

    surf_scores = {}
    high_scores = {}
    # units = {"mean_sea_level_pressure_0": "$Pa$", "10m_u_component_of_wind_1": "$m/s$", "10m_v_component_of_wind_2": "$m/s$", "2m_temperature_3": "$K$"}
    units = {"mean_sea_level_pressure": "$Pa$", "10m_u_component_of_wind": "$m/s$", "10m_v_component_of_wind": "$m/s$", "2m_temperature": "$K$"}
    titles = {"mean_sea_level_pressure": "mslp", "10m_u_component_of_wind": "10u", "10m_v_component_of_wind": "10v", "2m_temperature": "2mt", 
                "geopotential": "z", "temperature": "t", "u_component_of_wind": "u", "v_component_of_wind": "v", "specific_humidity": "q"}
    for i,f in enumerate(fs_pangu_score):
        data = np.load(f, allow_pickle=True)
        score = data['scores'].tolist()

        for k, v in score["surf"].items():
            if i==0:
                surf_scores[k] = np.array(v).astype(np.float32).reshape(-1,1)
            else:
                surf_scores[k] = np.concatenate((surf_scores[k], np.array(v).astype(np.float32).reshape(-1,1)), axis=1)
        for k, v in score["high"].items():
            if i==0:
                high_scores[k] = np.stack(v, axis=0).astype(np.float32)[:,:,None]
            else:
                high_scores[k] = np.concatenate((high_scores[k], np.stack(v, axis=0).astype(np.float32)[:,:,None]), axis=2)
    
    fig_prefix = Path(f"{cfg.directory.base_path}/ifs-init_score_comparison_pangu-cncast")
    fig_prefix.mkdir(exist_ok=True)
    month = 7
    cn_score_july = np.load(f"{cfg.directory.cncast_v1_scores_path}/ifs-init_month7_scores_china_5d_center{center}.npz", allow_pickle=True)["scores"].tolist()
    cn_score_surf = cn_score_july['surf']
    cn_score_high = cn_score_july['high']
    if use_aifs_bound:
        cn_aifs_score = np.load(f"{cfg.directory.cncast_v1_scores_path}/ifs-init_aifs-bound_month7_scores_china_5d_center{center}.npz", allow_pickle=True)["scores"].tolist()
        cn_aifs_score_surf = cn_aifs_score['surf']
        cn_aifs_score_high = cn_aifs_score['high']
    print(cn_score_surf.keys())
    if use_ifs:
        ifs_score_july = np.load(f"{cfg.directory.cncast_v1_scores_path}/ifs_month7_scores_china_5d_center{center}.npz", allow_pickle=True)["scores"].tolist()
        ifs_score_surf = ifs_score_july['surf']
        ifs_score_high = ifs_score_july['high']
    if use_aifs:
        aifs_score_july = np.load(f"{cfg.directory.cncast_v1_scores_path}/aifs_month7_scores_china_5d_center{center}.npz", allow_pickle=True)["scores"].tolist()
        aifs_score_surf = aifs_score_july['surf']
        aifs_score_high = aifs_score_july['high']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    labels = "abcdefgh"
    # x = list(range(1,len(surf_scores[k][0])+1))
    x = list(range(6, 121, 6))
    # fig.suptitle("RMSE Scores by Lead Time", fontsize=16)
    idx = 0
    for k, v in surf_scores.items():
        print(k)
        if k=="mean_sea_level_pressure":
            kk = "mslp"
        elif k=="10m_u_component_of_wind":
            kk = "10m_u_component_of_wind"
        elif k=="10m_v_component_of_wind":
            kk = "10m_v_component_of_wind"
        elif k=="2m_temperature":
            kk = "2mt"
        
        # kk = k
        rmses = [surf_scores[k][0], cn_score_surf[kk][0]]
        # accs = [surf_scores[k][1], cn_score_surf[k][1]]
        accs = [surf_scores[k][1]*100, cn_score_surf[kk][1]*100]
        var_name = "_".join(k.split("_"))
        if use_ifs:
            rmses = rmses + [ifs_score_surf[kk][0]]
            accs = accs + [ifs_score_surf[kk][1]*100]
        if use_aifs:
            rmses = rmses + [aifs_score_surf[kk][0]]
            accs = accs + [aifs_score_surf[kk][1]*100]
        if use_aifs_bound:
            rmses = rmses + [cn_aifs_score_surf[kk][0]]
            accs = accs + [cn_aifs_score_surf[kk][1]*100]
        plot_subfigure(rmses, x, axes[idx], "Lead time(hs)", f"RMSE({units[k]})", titles[k], labels[idx],level="surf", labels=["pangu", "cncast", "ifs", "aifs", "cncast_aifs-bc"])
        idx += 1
    
    plt.subplots_adjust(left=0.05, right=0.97, bottom=0.07, top=0.95)
    plt.savefig(f"{fig_prefix}/comparison_month{month}_surface_center{center}_rmse_with_aifs-bc_120h.pdf") 
    ## upper-air scores
    units = {"geopotential": "$m^2/s^2$", "temperature": "$K$", "u_component_of_wind": "$m/s$", "v_component_of_wind": "$m/s$", "specific_humidity": "$g/kg$"}

    fig = plt.figure(figsize=(22, 10), dpi=300)
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
    # fig.suptitle("RMSE Scores by Lead Time", fontsize=16)
    idx = 0
    level = "850"
    idx_level = cfg.input.levels.index(int(level))
    idx_level_pangu = [1000,850,700,600,500,400,300,250,200,150,100].index(int(level))
    idx_level_ifs = [1000,925,850,700,600,500,400,300,250,200,150,100,50].index(int(level))
    print(high_scores.keys(), cn_score_high["geopotential"][0].shape)

    for k, v in high_scores.items():
        name = k
        print(name, level)
        ##### IFS inited
        rmses = [high_scores[k][0,idx_level_pangu], cn_score_high[name][0][:,idx_level]]
        accs = [high_scores[k][1,idx_level_pangu]*100, cn_score_high[name][1][:,idx_level]*100]
        if use_ifs:
            rmses = rmses + [ifs_score_high[name][0][:,idx_level_ifs]]
            accs = accs + [ifs_score_high[name][1][:,idx_level_ifs]*100]
        if use_aifs:
            rmses = rmses + [aifs_score_high[name][0][:,idx_level_ifs]]
            accs = accs + [aifs_score_high[name][1][:,idx_level_ifs]*100]
        if use_aifs_bound:
            rmses = rmses + [cn_aifs_score_high[name][0][:,idx_level_ifs]]
            accs = accs + [cn_aifs_score_high[name][1][:,idx_level_ifs]*100]
        # var_name = "_".join(k.split("_")[:-1])
        if "specific" in k:
            rmses = [rmses[m]*1000 for m in range(len(rmses))]
        plot_subfigure(rmses, x, axes[idx], "Lead time(hs)", f"RMSE({units[k]})", f"{titles[k]}{level}", labels[idx], level="high", labels=["pangu", "cncast", "ifs", "aifs", "cncast_aifs-bc"])
        idx += 1
    
    plt.subplots_adjust(left=0.036, right=0.985, bottom=0.07, top=0.95)
    plt.savefig(f"{fig_prefix}/comparison_month{month}_upper_{level}hPa_center{center}_rmse_with_aifs-bc_120h.pdf")

if __name__ == '__main__':
    main()
