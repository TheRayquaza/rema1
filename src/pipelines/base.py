from abc import ABC, abstractmethod
from typing import Any, Self
from src.utils.logger import get_logger

class AbstractPipeline(ABC):
    """Abstract class for a data processing pipeline."""

    def __init__(self, config: Any = None) -> None:
        self.config = config
        if self.config != None:
            self.logger = get_logger(self.config.name)
        else:
            self.logger = get_logger(__name__)

    @abstractmethod
    def fit(self : Self, X: Any, y: Any) -> Self:
        """Fit the pipeline to the data."""
        pass

    @abstractmethod
    def transform(self : Self, X: Any, y: Any) -> Any:
        """Apply transformations to the data."""
        pass

    def fit_transform(self : Self, X: Any, y: Any) -> tuple:
        """Fit the pipeline and transform the data."""
        self.fit(X, y)
        return self.transform(X, y)
