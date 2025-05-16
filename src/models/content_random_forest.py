from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import ndcg_score
import joblib
from tqdm import tqdm

from src.utils.model import timeit
from src.models.base import AbstractModel

class ContentBasedFilteringRF(AbstractModel):
    def __init__(self, verbose: bool = True):
        super().__init__()
        self.models = []
        self.best_model = None
        self.feature_importances = None
        self.verbose = verbose

    @timeit
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            params: Optional[Dict[str, Any]] = None, n_folds: int = 3):
        if params is None:
            raise ValueError("Parameters must be provided for training.")

        self.logger.info(f"Starting training with parameters: {params}")
        self.logger.info(f"Using {n_folds}-fold cross-validation")

        unique_groups = np.unique(groups)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=params.get('random_state', 42))

        fold_scores = []
        self.models = []

        best_score = -np.inf
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(unique_groups),
                                                          total=n_folds,
                                                          desc="Cross-validation")):
            self.logger.info(f"\n{'='*50}\nFold {fold+1}/{n_folds}\n{'='*50}")

            train_groups = unique_groups[train_idx]
            val_groups = unique_groups[val_idx]

            train_mask = np.isin(groups, train_groups)
            val_mask = np.isin(groups, val_groups)

            X_train, y_train, g_train = X[train_mask], y[train_mask], groups[train_mask]
            X_val, y_val, g_val = X[val_mask], y[val_mask], groups[val_mask]

            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_val)

            score = self._evaluate_ndcg(g_val, y_val, preds)
            fold_scores.append(score)

            self.models.append(model)

            self.logger.info(f"Fold {fold+1} NDCG: {score:.6f}")

            if score > best_score:
                best_score = score
                best_model = model

        self.best_model = best_model

        if best_model is not None:
            self.feature_importances = {
                f"f{i}": imp for i, imp in enumerate(best_model.feature_importances_)
            }

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Cross-validation NDCG scores: {fold_scores}")
        self.logger.info(f"Mean CV NDCG: {mean_score:.6f} ± {std_score:.6f}")
        self.logger.info(f"Best model NDCG: {best_score:.6f}")

        if self.feature_importances:
            self.logger.info("Top 10 feature importances:")
            sorted_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, importance in sorted_features:
                self.logger.info(f"  {feature}: {importance:.4f}")

        return self

    def _evaluate_ndcg(self, groups, y_true, y_pred, k=100) -> float:
        ndcg_scores = []
        for g in np.unique(groups):
            mask = groups == g
            if np.sum(mask) > 1:
                ndcg_scores.append(ndcg_score([y_true[mask]], [y_pred[mask]], k=k))
        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    @timeit
    def predict(self, X: np.ndarray, use_all_models: bool = False) -> np.ndarray:
        self.logger.info(f"Generating predictions for {X.shape[0]} samples...")

        if use_all_models and len(self.models) > 0:
            self.logger.info("Using ensemble prediction from all folds")
            preds = np.zeros(X.shape[0])
            for model in self.models:
                preds += model.predict(X)
            preds /= len(self.models)
        elif self.best_model is not None:
            self.logger.info("Using best model for prediction")
            preds = self.best_model.predict(X)
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

    def save(self, filepath: str, save_all: bool = False) -> None:
        if self.best_model is not None:
            best_model_path = f"{filepath}_best.pkl"
            joblib.dump(self.best_model, best_model_path)
            self.logger.info(f"Best model saved to {best_model_path}")

        if save_all and len(self.models) > 0:
            for i, model in enumerate(self.models):
                fold_path = f"{filepath}_fold_{i}.pkl"
                joblib.dump(model, fold_path)
            self.logger.info(f"All {len(self.models)} fold models saved")

    def load(self, filepath: str):
        self.best_model = joblib.load(filepath)
        self.logger.info(f"Model loaded from {filepath}")

        try:
            self.feature_importances = {
                f"f{i}": imp for i, imp in enumerate(self.best_model.feature_importances_)
            }
        except:
            self.logger.warning("Could not extract feature importances from loaded model")

        return self
