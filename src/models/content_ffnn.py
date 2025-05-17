from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib
import os

from src.utils.model import timeit
from src.models.base import AbstractModel
from src.utils.random import set_seed


class ContentBasedFilteringNN(AbstractModel):
    def __init__(self, verbose: bool = True):
        super().__init__()
        self.models = []
        self.best_model = None
        self.feature_importances = None
        self.verbose = verbose
        if not verbose:
            tf.get_logger().setLevel('ERROR')

    def _build_model(self, input_dim: int, params: Dict[str, Any]) -> Model:
        hidden_layers = params.get('hidden_layers', [128, 64, 32])
        dropout_rate = params.get('dropout_rate', 0.2)
        learning_rate = params.get('learning_rate', 0.001)
        l2_reg = params.get('l2_reg', 0.001)
        activation = params.get('activation', 'relu')
        
        model = Sequential()
        
        model.add(Dense(hidden_layers[0], input_dim=input_dim, 
                  activation=activation, kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(1, activation='linear'))
        
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        model.summary()

        return model
    
    def _compute_feature_importance(self, model: Model, X: np.ndarray) -> Dict[str, float]:
        base_pred = model.predict(X, verbose=0).flatten()
        importances = {}
        
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            perm_pred = model.predict(X_permuted, verbose=0).flatten()
            orig_mse = np.mean((base_pred - X[:, i])**2)
            perm_mse = np.mean((perm_pred - X[:, i])**2)
            importances[f"f{i}"] = abs(perm_mse - orig_mse)

        total = sum(importances.values())
        if total > 0:
            importances = {k: v/total for k, v in importances.items()}
        
        return importances

    @timeit
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            params: Optional[Dict[str, Any]] = None):
        if params is None:
            raise ValueError("Parameters must be provided for training.")

        self.logger.info(f"Starting training with parameters: {params}")
        set_seed(params.get('random_state', 42))

        batch_size = params.get('batch_size', 256)
        epochs = params.get('epochs', 100)

        model = self._build_model(X.shape[1], params)

        model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1 if self.verbose else 0,
            callbacks=[
                EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
                ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6)
            ]
        )

        self.best_model = model

        if len(X) > 0:
            sample_size = min(1000, X.shape[0])
            X_sample = X[:sample_size]
            self.feature_importances = self._compute_feature_importance(model, X_sample)

        self.logger.info("Training complete.")

        if self.feature_importances:
            self.logger.info("Top 10 feature importances:")
            sorted_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, importance in sorted_features:
                self.logger.info(f"  {feature}: {importance:.4f}")

        return self

    @timeit
    def predict(self, X: np.ndarray, use_all_models: bool = False) -> np.ndarray:
        self.logger.info(f"Generating predictions for {X.shape[0]} samples...")

        if use_all_models and len(self.models) > 0:
            self.logger.info("Using ensemble prediction from all folds")
            preds = np.zeros(X.shape[0])
            for model in self.models:
                preds += model.predict(X, verbose=0).flatten()
            preds /= len(self.models)
        elif self.best_model is not None:
            self.logger.info("Using best model for prediction")
            preds = self.best_model.predict(X, verbose=0).flatten()
        else:
            raise ValueError("No trained models available. Please fit the model first.")

        return preds

    @timeit
    def recommend(self, 
                X: np.ndarray, 
                user_ids: List[int], 
                all_user_item_ids: np.ndarray,
                top_n: int = 10) -> Dict[int, List[Tuple[int, float]]]:
        predictions = self.predict(X)
        df = pd.DataFrame({
            "user_id": all_user_item_ids[:, 1],
            "item_id": all_user_item_ids[:, 0],
            "score": predictions
        })

        recommendations = {}
        for user_id in user_ids:
            user_df = df[df["user_id"] == user_id]
            top_items = user_df.nlargest(top_n, "score")[["item_id", "score"]]
            recommendations[user_id] = list(top_items.itertuples(index=False, name=None))

        return recommendations

    def get_feature_importances(self) -> Dict[str, float]:
        if self.feature_importances is None:
            self.logger.warning("Feature importances not available. Model may not be trained.")
            return {}
        return self.feature_importances

    def save(self, filepath: str) -> None:
        if self.best_model is not None:
            best_model_path = f"{filepath}"
            self.best_model.save(best_model_path)
            self.logger.info(f"Best model saved to {best_model_path}")
            
            if self.feature_importances:
                joblib.dump(self.feature_importances, f"{filepath}_importances.pkl")

    def load(self, filepath: str):
        self.best_model = load_model(filepath)
        self.logger.info(f"Model loaded from {filepath}")

        importances_path = f"{filepath}_importances.pkl"
        if os.path.exists(importances_path):
            self.feature_importances = joblib.load(importances_path)
        else:
            self.logger.warning("Feature importances file not found")

        return self
