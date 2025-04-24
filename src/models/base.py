from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Self
from src.utils.logger import get_logger

class AbstractModel(ABC):
    """Abstract class for machine learning models."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = get_logger(self.config.name)
        self.model = None

    @abstractmethod
    def fit(self: Self, X: np.ndarray, y: np.ndarray) -> Self:
        """Train the model using provided data."""
        pass

    @abstractmethod
    def predict(self: Self, X: np.ndarray) -> np.ndarray:
        """Predict output based on input data."""
        pass

    @abstractmethod
    def load(self: Self, path: str) -> Self:
        """Load the parameters of the model."""
        pass

    @abstractmethod
    def save(self: Self, path: str) -> Self:
        """Save the parameters of the model."""
        pass
