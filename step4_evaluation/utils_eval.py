import numpy as np


def compute_alignment_score(ratings_arr, conf_arr):
    reward_diffs = ratings_arr.reshape(-1, 1) - ratings_arr.reshape(1, -1)
    confidence_diffs = conf_arr.reshape(-1, 1) - conf_arr.reshape(1, -1)
    rewards = np.where(
        reward_diffs > 0, confidence_diffs, np.zeros_like(confidence_diffs)
    )
    punishments = np.where(
        reward_diffs < 0, -confidence_diffs, np.zeros_like(confidence_diffs)
    )
    total_rewards = rewards.sum(axis=1) + punishments.sum(axis=1)
    return total_rewards
