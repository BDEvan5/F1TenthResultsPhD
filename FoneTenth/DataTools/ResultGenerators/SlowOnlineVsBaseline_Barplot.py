
from TrajectoryAidedLearning.DataTools.plotting_utils import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.rcParams['pdf.use14corefonts'] = True



   
def SlowOnlineVsBaseline_maps_Barplot():
    p = "Data/Vehicles/SlowOnlineVsBaseline/"
    
    # fig, axs = plt.subplots(1, 2, figsize=(7, 1.8))
    fig, axs = plt.subplots(1, 2, figsize=(6, 1.4))
    xs = np.arange(4)
    
    barWidth = 0.32
    w = 0.05
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]
    
    keys = ["time", "success"]
    ylabels = "Time (s), Success (%)".split(", ")

    # for z in range(2):
    key = "time"
    ylabel = "Normalised Time"
    plt.sca(axs[0])
    baseline_mins, baseline_maxes, baseline_means = load_time_data(p, "Progress")
    online_mins, online_maxes, online_means = load_time_data(p, "Zero")
    
    mean_times = np.mean(np.array([baseline_means["time"], online_means["time"]]), axis=0)
    print(mean_times)
    
    plt.bar(br1, baseline_means[key]/mean_times, color=light_blue, width=barWidth, label="Baseline")
    plot_error_bars(br1, baseline_mins[key]/mean_times, baseline_maxes[key]/mean_times, dark_blue, w)
    
    plt.bar(br2, online_means[key]/mean_times, color=light_red, width=barWidth, label="Supervisor")
    plot_error_bars(br2, online_mins[key]/mean_times, online_maxes[key]/mean_times, dark_red, w)
        
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(1))
    plt.xticks([0, 1, 2, 3], ["AUT", "ESP", "GBR", "MCO"])
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.ylim(0.95, 1.05, 100)
    
    
    
    
    key = "success"
    ylabel = "Success (%)"
    
    plt.sca(axs[1])
    mins, maxes, means = load_time_data(p, "Progress")
    
    plt.bar(br1, means[key], color=light_blue, width=barWidth, label="Baseline")
    plot_error_bars(br1, mins[key], maxes[key], dark_blue, w)
    
    mins, maxes, means = load_time_data(p, "Zero")
    plt.bar(br2, means[key], color=light_red, width=barWidth, label="Supervisor")
    plot_error_bars(br2, mins[key], maxes[key], dark_red, w)
        
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(1))
    plt.xticks([0, 1, 2, 3], ["AUT", "ESP", "GBR", "MCO"])
    plt.ylabel(ylabel)
    plt.grid(True)
    
    fig.subplots_adjust(wspace=.38)
    
    axs[0].yaxis.set_major_locator(MultipleLocator(0.04))
    axs[1].yaxis.set_major_locator(MultipleLocator(25))
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="center", bbox_to_anchor=(0.55, -0.16))
        
    name = p + f"SlowOnlineVsBaseline_barplot"
    plt.savefig(name + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)

    
    
   
SlowOnlineVsBaseline_maps_Barplot()
   