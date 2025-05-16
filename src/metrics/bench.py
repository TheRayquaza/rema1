from typing import Dict, List
from src.metrics.map import mean_average_precision
from src.metrics.mar import mean_average_recall
from src.metrics.ndcg import ndcg
from src.metrics.maf1 import mean_average_f1

def bench_model(recommendations: Dict[int, List[int]], ground_truth: Dict[int, List[int]], k: int) -> None:
    mapk = mean_average_precision(recommendations, ground_truth, k)
    mark = mean_average_recall(recommendations, ground_truth, k)
    ndcg_k = ndcg(recommendations, ground_truth, k)
    f1 = mean_average_f1(mapk, mark)

    print(f"NDCG@{k} = {ndcg_k:.4f}")
    print(f"MAP@{k} = {mapk:.4f}")
    print(f"MAR@{k} = {mark:.4f}")
    print(f"F1@{k} = {f1:.4f}")
