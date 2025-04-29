from collections import defaultdict
import pandas as pd
from typing import Dict, Any, List

def build_ground_truth(data: pd.DataFrame, user_id_col: str, video_id_col: str) -> Dict[Any, List[Any]]:
    ground_truth = defaultdict(list)
    for _, row in data.iterrows():
        ground_truth[row[user_id_col]].append(row[video_id_col])
    return dict(ground_truth)

