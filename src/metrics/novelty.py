from typing import Any, Dict, List
import numpy as np
from src.metrics.base import RecommendationMetric

class Novelty(RecommendationMetric):
    """Novelty metric for evaluating recommendation systems."""

    def __init__(self, name: str = "novelty") -> None:
        super().__init__(name)

    def compute(self, recommendations: Dict[Any, List[Any]], **kwargs) -> float:
        item_popularity = kwargs.get("item_popularity", {})
        if not item_popularity:
            self.logger.warning("Item popularity not provided.")
            return 0.0
        total_novelty = 0.0
        count = 0

        for recs in recommendations.values():
            for item in recs:
                popularity = item_popularity.get(item, 1e-6)
                total_novelty += -np.log2(popularity)
                count += 1

        return total_novelty / count if count > 0 else 0.0
