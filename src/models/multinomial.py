import joblib
import os
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from pydantic import BaseModel, Field
from typing import Self, Any
from src.models.model import AbstractModel

class MultinomialParams(BaseModel):
    """Configuration for the multinomial logistic regression model."""
    type: str = "multinomial_nb"
    name: str = "Complexity"
    alpha: float =  Field(default=1.0, gt=0)

class Multinomial(AbstractModel):
    def __init__(self, config: MultinomialParams):
        super().__init__(config)
        #self.logger.info(f"Initializing {self.config.name} model with configuration: {self.config.model_dump()}")
        self.model = MultinomialNB(alpha=self.config.alpha)

    def fit(self: Self, X: np.ndarray, y: np.ndarray) -> Self:
        #self.logger.info(f"Starting training with config: {self.config.model_dump()}")
        self.model.fit(X, y)
        self.logger.info("Training completed successfully")
        return self

    def predict(self: Self, X: np.ndarray) -> np.ndarray:
        self.logger.info(f"Making predictions with model: {self.config.name}")
        predictions = self.model.predict(X)
        self.logger.info(f"Predictions made for {len(X)} samples")
        return predictions

    def save(self: Self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}")

    def load(self: Self, path: str) -> Self:
        if not os.path.exists(path):
            self.logger.error(f"Model file not found at path: {path}")
            raise FileNotFoundError(f"Model file not found at path: {path}")

        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}")
        return self
