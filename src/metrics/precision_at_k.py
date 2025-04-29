
def precision_at_k(recommendations: Dict[Any, List[Any]], ground_truth: Dict[Any, List[Any]], k: int) -> float:
    if not ground_truth:
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