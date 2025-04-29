import pandas as pd
import numpy as np
from typing import Dict, List, Any
import math
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def ndcg(recommendations: Dict[int, List[int]], ground_truth: Dict[int, List[int]], k: int) -> float:
    if ground_truth is None:
        return 0.0
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    def dcg(recommended: Any, relevant: Any, k: int):
        score = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                score += 1 / math.log2(i + 2)
        return score

    def idcg(relevant: List, k: int):
        return sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))

    total_ndcg = 0.0
    user_count = 0

    for user, recs in recommendations.items():
        relevant = ground_truth.get(user, [])
        if not relevant:
            continue
        user_dcg = dcg(recs, relevant, k)
        user_idcg = idcg(relevant, k)
        total_ndcg += user_dcg / user_idcg if user_idcg > 0 else 0
        user_count += 1

    return total_ndcg / user_count if user_count > 0 else 0.0

def precision_at_k(recommendations: Dict[int, List[int]], ground_truth: Dict[int, List[int]], k: int) -> float:
    if ground_truth is None:
        return 0.0
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    precisions = []

    for user, recs in recommendations.items():
        relevant_items = set(ground_truth.get(user, []))
        recommended_top_k = recs[:k]

        if not recommended_top_k:
            continue

        hits = sum(1 for item in recommended_top_k if item in relevant_items)
        precision = hits / k
        precisions.append(precision)

    if not precisions:
        return 0.0

    return float(np.mean(precisions))

def novelty(recommendations: Dict[int, List[int]], item_popularity: Dict[int, float]) -> float:
    if len(item_popularity) == 0:
        print("Item popularity not provided.")
        return 0.0
    total_novelty = 0.0
    count = 0

    for recs in recommendations.values():
        for item in recs:
            popularity = item_popularity.get(item, 1e-6)
            total_novelty += -np.log2(popularity)
            count += 1

    return total_novelty / count if count > 0 else 0.0
