from typing import Any, Dict, List, Self
from src.metrics.base import RecommendationMetric

class Coverage(RecommendationMetric):
    """Coverage metric for evaluating recommendation systems."""

    def __init__(self, name: str = "coverage") -> Self:
        super().__init__(name)

    def compute(self, recommendations: Dict[Any, List[Any]], **kwargs) -> float:
        items = kwargs.get("items", set())
        recommended_items = set(item for recs in recommendations.values() for item in recs)
        if not items:
            return 0.0
        return len(recommended_items) / len(items)
