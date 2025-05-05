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

def normalize_ratings(Y, R):
    Ymean = np.zeros(Y.shape[0])
    for i in range(Y.shape[0]):
        if np.sum(R[i, :]) > 0:
            Ymean[i] = np.sum(Y[i, :] * R[i, :]) / np.sum(R[i, :])

    Ynorm = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        Ynorm[i, :] = (Y[i, :] - Ymean[i]) * R[i, :]

    return Ynorm, Ymean