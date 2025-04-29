from typing import Any, Dict, List, Self, Callable
from src.metrics.base import RecommendationMetric

class Diversity(RecommendationMetric):
    """Diversity metric for evaluating recommendation systems."""

    def __init__(self, similarity_func: Callable, name: str = "diversity") -> Self:
        super().__init__(name)
        self.similarity_func = similarity_func

    def compute(self, recommendations: Dict[Any, List[Any]], item_similarity) -> float:
        if not item_similarity:
            self.logger.warning("Item similarity not provided.")
            return 0.0
        total_diversity = 0.0
        pair_count = 0

        for recs in recommendations.values():
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    item_i = recs[i]
                    item_j = recs[j]
                    sim = item_similarity.get((item_i, item_j), item_similarity.get((item_j, item_i), 0))
                    diversity = 1 - sim
                    total_diversity += diversity
                    pair_count += 1

        return total_diversity / pair_count if pair_count > 0 else 0.0

def diversity(recommendations: Dict[Any, List[Any]], item_similarity) -> float:
    if not item_similarity:
        print("Item similarity not provided.")
        return 0.0
    total_diversity = 0.0
    pair_count = 0

    for recs in recommendations.values():
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                item_i = recs[i]
                item_j = recs[j]
                sim = item_similarity.get((item_i, item_j), item_similarity.get((item_j, item_i), 0))
                diversity = 1 - sim
                total_diversity += diversity
                pair_count += 1

    return total_diversity / pair_count if pair_count > 0 else 0.0
