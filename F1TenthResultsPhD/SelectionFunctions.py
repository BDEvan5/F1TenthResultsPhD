from F1TenthResultsPhD.Planners.Architectures import *
from F1TenthResultsPhD.Utils.StdRewards import *
from F1TenthResultsPhD.Utils.RacingLineRewards import *


def select_architecture(architecture: str): 
    if architecture == "slow": 
        architecture_type = SlowArchitecture
    elif architecture == "fast":
        architecture_type = FastArchitecture
    elif architecture == "link":
        architecture_type = LinkArchitecture
    else: raise Exception("Unknown architecture")

    return architecture_type


def select_reward_function(run, conf, std_track, race_track):
    reward = run.reward
    if reward == "Zero":
        reward_function = ZeroReward()
    elif reward == "Progress":
        reward_function = ProgressReward(std_track)
    elif reward == "Cth": 
        reward_function = CrossTrackHeadReward(std_track, conf)
    elif reward == "CthRace": 
        reward_function = CrossTrackHeadReward(race_track, conf)
    elif reward == "PPPS":
        reward_function = PppsReward(conf, run)
    elif reward == "Velocity":
        reward_function = VelocityReward(conf, run)
    elif reward == "v1":
        reward_function = RewardV1(race_track, conf)
    elif reward == "v2":
        reward_function = RewardV2(race_track, conf)
    elif reward == "v3":
        reward_function = RewardV3(race_track, conf)
    else: raise Exception("Unknown reward function: " + reward)
        
    return reward_function