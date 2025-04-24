from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.utils.logger import get_logger

class RecommendationMetric(ABC):
    """
    Abstract base class for recommendation system metrics.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = get_logger(name)

    def __call__(self, recommendations: Dict[Any, List[Any]], **kwargs) -> float:
        return self.compute(recommendations, **kwargs)

    @abstractmethod
    def compute(self, recommendations: Dict[Any, List[Any]], **kwargs) -> float:
        pass
