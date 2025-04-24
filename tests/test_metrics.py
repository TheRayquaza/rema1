from src.metrics.ndcg import NDCG
from src.metrics.coverage import Coverage

def test_ndcg_basic(sample_recommendations, sample_ground_truth):
    metric = NDCG()
    score = metric(sample_recommendations, ground_truth=sample_ground_truth, k=3)
    assert 0.0 <= score <= 1.0, "nDCG should be between 0 and 1"

def test_coverage_all_items(sample_recommendations, all_items):
    metric = Coverage()
    score = metric(sample_recommendations, all_items=all_items)
    assert 0.0 <= score <= 1.0
    assert round(score, 2) == 0.83  # optional check, depending on expected value
