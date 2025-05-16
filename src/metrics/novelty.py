from typing import Dict, List
import numpy as np

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
