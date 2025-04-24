import pandas as pd
import numpy as np
from typing import Self
from src.pipelines.base import AbstractPipeline
from sklearn.model_selection import train_test_split

class SimplePipelineParams:
    """Configuration for the simple pipeline."""
    name: str = "SimplePipeline"
    type: str = "simple"
    file_path : str = "../data/dataset.csv"
    train_size: float = 0.8

class SimplePipeline(AbstractPipeline):
    """A simple pipeline implementing data loading, transformation, and splitting."""

    def __init__(self: Self, config: SimplePipelineParams) -> None:
        super().__init__(config)

    def load(self: Self) -> pd.DataFrame:
        """Simulate loading a dataset."""
        self.logger.info(f"loading data  {self.config.file_path}")
        if self.config.file_path.endswith(".csv"):
            return pd.read_csv(self.config.file_path)    
        raise Exception(f"Unable to load file {self.config.file_path}")

    def transform(self: Self, data: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """Apply simple transformations (e.g., encoding labels)."""
        self.logger.info("Transforming data")
        
        X = data.drop(columns=['Difficulty', 'Category', 'Serving_Size'])
        y = data[['Difficulty']].replace({'Easy': 0, 'Medium': 1, 'Hard': 2})

        X = pd.get_dummies(X, drop_first=False)

        return X, y

    def split(self: Self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Split data into train/test sets."""
        self.logger.info(f"split data")
        return train_test_split(X, y, train_size=self.config.train_size)
