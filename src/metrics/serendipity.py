from typing import Any, Dict, List
from src.metrics.base import RecommendationMetric

class Serendipity(RecommendationMetric):
    """Serendipity metric for evaluating recommendation systems."""

    def __init__(self, name: str = "serendipity") -> None:
        super().__init__(name)

    def compute(self, recommendations: Dict[Any, List[Any]], **kwargs) -> float:
        ground_truth = kwargs.get("ground_truth", {})
        baseline_recommendations = kwargs.get("baseline", {})
        if not ground_truth or not baseline_recommendations:
            self.logger.warning("Ground truth or baseline recommendations not provided.")
            return 0.0
        total_serendipity = 0.0
        user_count = 0

        for user, recs in recommendations.items():
            relevant = set(ground_truth.get(user, []))
            baseline = set(baseline_recommendations.get(user, []))
            if not relevant:
                continue

            surprising_recs = [item for item in recs if item not in baseline and item in relevant]
            total_serendipity += len(surprising_recs) / len(recs) if recs else 0
            user_count += 1

        return total_serendipity / user_count if user_count > 0 else 0.0
