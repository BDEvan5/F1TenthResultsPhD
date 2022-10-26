import numpy as np 
from FoneTenth.Utils.TD3 import TD3
from FoneTenth.Utils.HistoryStructs import TrainHistory
import torch
from numba import njit

from FoneTenth.Utils.utils import init_file_struct, calculate_speed, calculate_steering

from matplotlib import pyplot as plt

from FoneTenth.Planners.PurePursuit import PurePursuit



class E2eSlowArch:
    def __init__(self, run, conf):
        self.range_finder_scale = conf.range_finder_scale
        self.n_beams = conf.n_beams
        self.max_v = conf.max_v
        self.max_steer = conf.max_steer
        self.vehicle_speed = run.vehicle_speed

        self.state_space = conf.n_beams 
        self.action_space = 1

        self.n_scans = run.n_scans
        self.scan_buffer = np.zeros((self.n_scans, self.n_beams))
        self.state_space *= self.n_scans

    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """
            
        scan = np.array(obs['scan']) 

        scaled_scan = scan/self.range_finder_scale
        scan = np.clip(scaled_scan, 0, 1)


        if self.scan_buffer.all() ==0: # first reading
            for i in range(self.n_scans):
                self.scan_buffer[i, :] = scan 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        nn_obs = np.reshape(self.scan_buffer, (self.n_beams * self.n_scans))

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        speed = self.vehicle_speed

        action = np.array([steering_angle, speed])

        return action


class E2eRaceArch:
    def __init__(self, run, conf):
        self.state_space = conf.n_beams 
        self.range_finder_scale = conf.range_finder_scale
        self.n_beams = conf.n_beams
        self.max_v = run.speed_cap
        # self.max_v = conf.max_v 
        self.max_steer = conf.max_steer

        self.action_space = 2

        self.n_scans = run.n_scans
        self.scan_buffer = np.zeros((self.n_scans, self.n_beams))
        self.state_space *= self.n_scans
        self.speed_cap = run.speed_cap

    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """
            
        scan = np.array(obs['scan']) 

        scaled_scan = scan/self.range_finder_scale
        scan = np.clip(scaled_scan, 0, 1)

        if self.scan_buffer.all() ==0: # first reading
            for i in range(self.n_scans):
                self.scan_buffer[i, :] = scan 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        nn_obs = np.reshape(self.scan_buffer, (self.n_beams * self.n_scans))

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        speed = (nn_action[1] + 1) * (self.max_v  / 2 - 0.5) + 1
        speed = min(speed, self.speed_cap) # cap the speed

        action = np.array([steering_angle, speed])

        return action

