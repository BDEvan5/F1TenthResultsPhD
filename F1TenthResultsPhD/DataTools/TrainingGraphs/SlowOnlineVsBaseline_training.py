import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.use14corefonts'] = True

from matplotlib.ticker import MultipleLocator, PercentFormatter

from F1TenthResultsPhD.Utils.utils import *
from F1TenthResultsPhD.DataTools.TrainingGraphs.TrainingUtils import *



def SlowOnlineVsBaseline_TrainingGraphMaps():
    p = "Data/Vehicles/SlowOnlineVsBaseline/"

    map_names = ["f1_esp", "f1_gbr", "f1_aut", "f1_mco"]
    print_names = ["ESP", "GBR", "AUT", "MCO"]
    repeats = 3
    moving_avg = 10
    # vehicle_names = ["Std"]
    vehicle_names = ["Std", "Online"]
    paths = ["slow_Std_Std_Progress", "slow_Online_Std_Zero"]
    
    xs = np.arange(400)
        
    fig, axs = plt.subplots(1, 2, sharey=False, figsize=(6, 1.4))
    axs = axs.reshape(-1)
        
    # for ax, vehicle_name in enumerate(vehicle_names):
    
    step_list = [[] for _ in range(len(map_names))]
    progresses_list = [[] for _ in range(len(map_names))]
        
    for i in range(repeats):
        for j in range(len(map_names)):
            path = p + paths[0] + f"_{map_names[j]}_2_1_{i}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            
            steps = np.cumsum(lengths)/100
            progresses = true_moving_average(rewards, moving_avg)
            # progresses = true_moving_average(progresses, moving_avg)*100
            step_list[j].append(steps)
            progresses_list[j].append(progresses)


    for j in range(len(map_names)):
        mins, maxes, means = convert_to_min_max_avg(step_list[j], progresses_list[j], xs)
        axs[0].plot(xs/10, means, '-', color=pp[j], linewidth=1.6, label=print_names[j])
            
        # axs[ax].fill_between(xs, mins, maxes, color=pp[j], alpha=0.3)
        
    axs[0].grid(True)
        
    step_list = [[] for _ in range(len(map_names))]
    rewards_list = [[] for _ in range(len(map_names))]
    moving_avg = 2
    xs = np.arange(60)
        
    for i in range(repeats):
        for j in range(len(map_names)):
            path = p + paths[1] + f"_{map_names[j]}_2_1_{i}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            
            steps = np.cumsum(lengths)/100
            # progresses = true_moving_average(rewards, moving_avg)
            step_list[j].append(steps)
            rewards_list[j].append(rewards)


    for j in range(len(map_names)):
        mins, maxes, means = convert_to_min_max_avg(step_list[j], rewards_list[j], xs)
        axs[1].plot(xs/10, means, '-', color=pp[j], linewidth=1.6)
            
        # axs[1].fill_between(xs, mins, maxes, color=pp[j], alpha=0.3)
        
        axs[1].grid(True)

    axs[0].set_xlabel("Training Steps (x1000)")
    axs[1].set_xlabel("Training Steps (x1000)")
    # axs[0].set_title("Avg. Ep. Reward")
    axs[0].set_ylabel("Avg. Ep. Reward")
    axs[1].set_ylabel("Avg. Ep. Reward")
    axs[0].set_title("Conventional")
    axs[1].set_title("Supervisor")
    # axs[0].set_title("Avg. Progress %")
    # axs[1].set_title("Reward")
    axs[0].get_yaxis().set_major_locator(MultipleLocator(1))
    axs[0].get_xaxis().set_major_locator(MultipleLocator(10))
    axs[1].get_yaxis().set_major_locator(MultipleLocator(50))
    axs[1].get_xaxis().set_major_locator(MultipleLocator(2))
    fig.subplots_adjust(wspace=.45)
    # axs[1].set_ylabel("Avg. Progress %")
    fig.legend(loc='center', ncol=4, bbox_to_anchor=(0.5, -0.3))
    # axs[0].legend(loc='center', ncol=4, bbox_to_anchor=(1.1, 1.1))

    name = p + f"SlowOnlineVsBaseline_TrainingGraphMaps"
    plt.savefig(name + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)


SlowOnlineVsBaseline_TrainingGraphMaps()

