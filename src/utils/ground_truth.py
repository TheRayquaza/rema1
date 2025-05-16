from collections import defaultdict
import pandas as pd

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

def build_ground_truth_last_quartile(pivot_matrix: pd.DataFrame):
    ground_truth = defaultdict(list)
    for user_id, row in pivot_matrix.iterrows():
        non_zero_values = row[row != 0 & row.isna()]
        if non_zero_values.empty:
            continue
        q3 = non_zero_values.quantile(0.75)
        top_items = non_zero_values[non_zero_values >= q3].index.tolist()
        ground_truth[user_id] = top_items
    return dict(ground_truth)

def build_ground_truth_top_10_percent(pivot_matrix: pd.DataFrame):
    ground_truth = defaultdict(list)
    for user_id, row in pivot_matrix.iterrows():
        non_zero_values = row[(row != 0) & (~row.isna())]
        if non_zero_values.empty:
            continue

        sorted_values = non_zero_values.sort_values(ascending=False)

        top_n = max(1, int(len(sorted_values) * 0.10))
        top_items = sorted_values.iloc[:top_n].index.tolist()

        ground_truth[user_id] = top_items

    return dict(ground_truth)
