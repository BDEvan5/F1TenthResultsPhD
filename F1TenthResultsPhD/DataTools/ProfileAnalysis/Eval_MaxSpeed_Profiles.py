import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
from RacingRewards.DataTools.MapData import MapData
from RacingRewards.RewardSignals.StdTrack import StdTrack 

from F1TenthResultsPhD.Utils.utils import *

class TestLapData:
    def __init__(self, path, lap_n=0):
        self.path = path
        self.vehicle_name = self.path.split("/")[-2]
        self.map_name = self.vehicle_name.split("_")[2]
        if self.map_name == "f1":
            self.map_name += "_" + self.vehicle_name.split("_")[3]
        self.map_data = MapData(self.map_name)
        self.race_track = StdTrack(self.map_name)

        self.states = None
        self.actions = None
        self.lap_n = lap_n

        self.load_lap_data()

    def load_lap_data(self):
        try:
            data = np.load(self.path + f"Testing/Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
        except Exception as e:
            print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            return 0
        self.states = data[:, :7]
        self.actions = data[:, 7:]

        return 1 # to say success

    def generate_state_progress_list(self):
        pts = self.states[:, 0:2]
        progresses = [0]
        for pt in pts:
            p = self.race_track.calculate_progress_percent(pt)
            # if p < progresses[-1]: continue
            progresses.append(p)
            
        return np.array(progresses[:-1])



def make_slip_compare_graph():
    # map_name = "f1_gbr"
    map_name = "f1_esp"
    # pp_path = f"Data/Vehicles/RacingResultsWeekend/PP_Std_{map_name}_1_0/"
    # agent_path = f"Data/Vehicles/RacingResultsWeekend/Agent_Cth_{map_name}_2_1/"

    pp_path = f"Data/Vehicles/PerformanceSpeed/PP_Std5_{map_name}_1_0/"
    agent_path = f"Data/Vehicles/PerformanceSpeed/Agent_Cth_{map_name}_3_0/"


    pp_data = TestLapData(pp_path)
    agent_data = TestLapData(agent_path, 2)

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 1.7), sharex=True)
    ax1.plot(agent_data.states[:, 6], color=pp[1], label="Agent")
    ax1.plot(pp_data.states[:, 6], color=pp[0], label="PP")


    ax1.set_ylabel("Slip angle")
    ax1.set_xlabel("Time steps")
    ax1.legend(ncol=2)

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"Data/HighSpeedEval/SlipCompare_{map_name}.pdf", bbox_inches='tight')

    plt.show()
    
    
    

def make_velocity_compare_graph():
    # map_name = "f1_gbr"
    map_name = "f1_esp"
    # pp_path = f"Data/Vehicles/RacingResultsWeekend/PP_Std_{map_name}_1_0/"
    # agent_path = f"Data/Vehicles/RacingResultsWeekend/Agent_Cth_{map_name}_2_1/"
    pp_path = f"Data/Vehicles/PerformanceSpeed/PP_Std5_{map_name}_1_0/"
    agent_path = f"Data/Vehicles/PerformanceSpeed/Agent_Cth_{map_name}_3_0/"

    pp_data = TestLapData(pp_path)
    agent_data = TestLapData(agent_path, 2)

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 1.7), sharex=True)
    ax1.plot(agent_data.states[:, 3], color=pp[1], label="Agent")
    ax1.plot(pp_data.states[:, 3], color=pp[0], label="PP")

    # ax2.plot(pp_data.states[:, 6], 'r-')
    # ax2.plot(agent_data.states[:, 6], 'b-')

    ax1.set_ylabel("Velocity m/s")
    ax1.set_xlabel("Time steps")
    ax1.legend(ncol=2)
    # ax2.set_ylabel("Slip Angle")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"Data/HighSpeedEval/VelocityCompare_{map_name}.pdf", bbox_inches='tight')

    plt.show()


