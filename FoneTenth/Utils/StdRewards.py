from FoneTenth.Utils.RewardUtils import *
from matplotlib import pyplot as plt

from RacingRewards.Utils.utils import *
from FoneTenth.Utils.StdTrack import StdTrack

# rewards functions
class DistanceReward():
    def __init__(self, race_track: StdTrack) -> None:
        self.race_track = race_track

    def __call__(self, observation, prev_obs, pre_action):
        if prev_obs is None: return 0

        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash
        
        
        position = observation['state'][0:2]
        prev_position = prev_obs['state'][0:2]
        theta = observation['state'][2]

        s = self.race_track.calculate_progress(prev_position)
        ss = self.race_track.calculate_progress(position)
        reward = (ss - s) / self.race_track.total_s
        if abs(reward) > 0.5: # happens at end of eps
            return 0.001 # assume positive progress near end

        # self.race_track.plot_vehicle(position, theta)


        reward *= 10 # remove all reward
        return reward 

class CrossTrackHeadReward:
    def __init__(self, race_track: StdTrack, conf):
        self.race_track = race_track
        self.r_veloctiy = conf.r_velocity
        self.r_distance = conf.r_distance
        self.max_v = conf.max_v

    def __call__(self, observation, prev_obs, pre_action):
        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash

        position = observation['state'][0:2]
        theta = observation['state'][2]
        heading, distance = self.race_track.get_cross_track_heading(position)
        # self.race_track.plot_vehicle(position, theta)

        d_heading = abs(robust_angle_difference_rad(heading, theta))
        r_heading  = np.cos(d_heading)  * self.r_veloctiy # velocity
        r_heading *= (observation['state'][3] / self.max_v)

        r_distance = distance * self.r_distance 

        reward = r_heading - r_distance
        reward = max(reward, 0)
        # reward *= 0.1
        return reward
        # return 0 # test super #!1!!!!!!!!!!!!

class NoneReward:
    def __call__(self, observation, prev_obs, pre_action):
        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash
        return 0


class TimeReward:
    def __init__(self, conf):
        self.rk = conf.r_time

    def __call__(self, observation, prev_obs, pre_action):
        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash
        return - self.rk 


class CthLinkReward:
    def __init__(self, race_track: StdTrack, conf):
        self.race_track = race_track
        self.r_veloctiy = conf.r_velocity
        self.r_distance = conf.r_distance
        self.max_v = conf.max_v

    def __call__(self, observation, prev_obs, pre_action):
        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash

        position = observation['state'][0:2]
        theta = observation['state'][2]
        heading, distance = self.race_track.get_cross_track_heading(position)
        # self.race_track.plot_vehicle(position, theta)

        d_heading = abs(robust_angle_difference_rad(heading, theta))
        r_heading  = np.cos(d_heading)  * self.r_veloctiy # velocity
        r_heading *= (observation['state'][3] / self.max_v)

        r_distance = distance * self.r_distance
        reward = r_heading - r_distance

        # link reward
        if prev_obs is not None:
            v = prev_obs['state'][3]
            steer = pre_action[0]
            if calculate_steering(v) < abs(steer):
                reward -= 0.2

        return reward


from FoneTenth.Planners.PurePursuit import PurePursuit

class VelocityReward:
    def __init__(self, conf, run):
        self.speed_cap = run.speed_cap

    def __call__(self, observation, prev_obs, pre_action):
        if pre_action is None: return 0
        if observation['lap_done']:
            return 1
        if observation['colision_done']:
            return -1

        # v = observation['state'][3]
        v = pre_action[1]
        reward = v / self.speed_cap
        # reward = v / self.speed_cap - 0.2 # upbias


        # reward *= 0.5
        # reward *= 2
        reward ** 2

        return reward


class PppsReward:
    def __init__(self, conf, run):
        # self.pp = PurePursuit(conf, run, False, False)
        self.pp = PurePursuit(conf, run, False, True)

        self.beta_c = 0.4
        self.beta_steer_weight = 0.4
        if run.racing_mode =="Fast":
            self.beta_velocity_weight = 0.4
        else:
            self.beta_velocity_weight = 0.0

        self.max_steer_diff = 0.8
        self.max_velocity_diff = 2.0
        # self.max_velocity_diff = 4.0

    def __call__(self, observation, prev_obs, action):
        if prev_obs is None: return 0

        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash
        
        pp_act = self.pp.plan(prev_obs)

        steer_reward =  (abs(pp_act[0] - action[0]) / self.max_steer_diff)  * self.beta_steer_weight

        throttle_reward =   (abs(pp_act[1] - action[1]) / self.max_velocity_diff) * self.beta_velocity_weight

        # reward = self.beta_c - steer_reward
        reward = self.beta_c - steer_reward - throttle_reward
        reward = max(reward, 0) # limit at 0

        reward *= 0.5

        return reward


#TODO: njit these function
def robust_angle_difference_degree(x, y):
    """Returns the difference between two angles in DEGREES
    r = x - y"""
    x = np.deg2rad(x)
    y = np.deg2rad(y)
    r = np.arctan2(np.sin(x-y), np.cos(x-y))
    return np.rad2deg(r)

def robust_angle_difference_rad(x, y):
    """Returns the difference between two angles in RADIANS
    r = x - y"""
    return np.arctan2(np.sin(x-y), np.cos(x-y))


if __name__ == '__main__':
    test_angle_diff()
    pass
