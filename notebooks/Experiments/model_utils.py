import pandas as pd
from collections import defaultdict
import numpy as np
from typing import Tuple

def build_ground_truth_at_least_once(pivot_matrix: pd.DataFrame):
    ground_truth = defaultdict(list)
    for user_id, row in pivot_matrix.iterrows():
        ground_truth[user_id] = row[row != 0].index.tolist()
    return dict(ground_truth)

def build_ground_truth_at_least_once_entirely(pivot_matrix: pd.DataFrame):
    ground_truth = defaultdict(list)
    for user_id, row in pivot_matrix.iterrows():
        ground_truth[user_id] = row[row > 1].index.tolist()
    return dict(ground_truth)

def build_ground_truth_mean(pivot_matrix: pd.DataFrame):
    ground_truth = defaultdict(list)
    for user_id, row in pivot_matrix.iterrows():
        non_zero_values = row[row != 0] # removing older NaN
        if non_zero_values.empty:
            continue
        mean_watch_ratio = non_zero_values.mean()
        watched_videos = non_zero_values[non_zero_values > mean_watch_ratio].index.tolist()
        ground_truth[user_id] = watched_videos
    return dict(ground_truth)

def normalize_ratings(Y: np.ndarray, R: np.ndarray, axis: int = 1) -> Tuple:
    Ymean = (np.sum(Y * R, axis=axis) / (np.sum(R, axis=axis) + 1e-12))
    if axis == 0:
        Ymean = Ymean.reshape(1, -1)
        Ynorm = Y - np.multiply(R, Ymean)
    else:
        Ymean = Ymean.reshape(-1, 1)
        Ynorm = Y - np.multiply(R, Ymean)
    return (Ynorm, Ymean)
