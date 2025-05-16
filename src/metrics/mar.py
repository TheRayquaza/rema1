import numpy as np
from typing import Dict, List

def mean_average_recall(recommendations: Dict[int, List[int]], ground_truth: Dict[int, List[int]], k: int) -> float:
    if ground_truth is None:
        return 0.0
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    recalls = []

    for user, recs in recommendations.items():
        relevant_items = set(ground_truth.get(user, []))
        recommended_top_k = recs[:k]

        if not recommended_top_k:
            continue

        hits = sum(1 for item in recommended_top_k if item in relevant_items)
        recall = hits / len(relevant_items)
        recalls.append(recall)

    if not recalls:
        return 0.0

    return float(np.mean(recalls))