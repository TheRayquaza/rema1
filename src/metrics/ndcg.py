from metrics.base import RecommendationMetric
from typing import Dict, List, Any, Self
import math

class NDCG(RecommendationMetric):
    """NDCG metric for evaluating recommendation systems."""

    def __init__(self, name: str = "ndcg") -> Self:
        super().__init__(name)

    def compute(self, recommendations: Dict[Any, List[Any]], **kwargs) -> float:
        ground_truth = kwargs.get("ground_truth", {})
        k = kwargs.get("k", 10)
        if not ground_truth:
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
