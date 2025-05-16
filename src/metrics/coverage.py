from typing import Any, Dict, List

def coverage(recommendations: Dict[Any, List[Any]]) -> float:
    items = kwargs.get("items", set())
    recommended_items = set(item for recs in recommendations.values() for item in recs)
    if not items:
        return 0.0
    return len(recommended_items) / len(items)
