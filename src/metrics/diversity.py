from typing import Any, Dict, List

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
