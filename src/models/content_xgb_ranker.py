from typing import Dict, Any, Optional
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from tqdm import tqdm
from src.utils.model import timeit
from src.models.base import AbstractModel

class ContentBasedFilteringXGRanker(AbstractModel):    
    def __init__(self, verbose: bool = True):
        super().__init__()
        self.models = []
        self.best_model = None
        self.feature_importances = None
        self.verbose = verbose

    def _create_dmatrix(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                        groups: Optional[np.ndarray] = None) -> xgb.DMatrix:
        dmatrix = xgb.DMatrix(X, y)
        if groups is not None:
            unique_groups, group_counts = np.unique(groups, return_counts=True)
            dmatrix.set_group(group_counts)
        return dmatrix

    @timeit
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
            params: Optional[Dict[str, Any]] = None, n_folds: int = 3, 
            early_stopping_rounds: int = 50):
        if params is None:
            raise ValueError("Parameters must be provided for training.")

        self.logger.info(f"Starting training with parameters: {params}")
        self.logger.info(f"Using {n_folds}-fold cross-validation")
        
        unique_groups = np.unique(groups)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=params.get('random_state', 42))
        
        fold_scores = []
        self.models = []
        
        # For tracking best model
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
            
            X_train, y_train, groups_train = X[train_mask], y[train_mask], groups[train_mask]
            X_val, y_val, groups_val = X[val_mask], y[val_mask], groups[val_mask]
            
            dtrain = self._create_dmatrix(X_train, y_train, groups_train)
            dval = self._create_dmatrix(X_val, y_val, groups_val)
            
            watchlist = [(dtrain, 'train'), (dval, 'validation')]
            
            evals_result = {}
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params.get('n_estimators', 300),
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
                evals_result=evals_result,
                verbose_eval=10 if self.verbose else False
            )
            
            self.models.append(model)
            best_iteration = model.best_iteration
            best_score_fold = max(evals_result['validation'][params['eval_metric']])
            fold_scores.append(best_score_fold)
            
            self.logger.info(f"Fold {fold+1} best score: {best_score_fold:.6f} at iteration {best_iteration}")
            
            if best_score_fold > best_score:
                best_score = best_score_fold
                best_model = model
                
        self.best_model = best_model
        
        if best_model is not None:
            self.feature_importances = best_model.get_score(importance_type='gain')
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Cross-validation scores: {fold_scores}")
        self.logger.info(f"Mean CV score: {mean_score:.6f} ± {std_score:.6f}")
        self.logger.info(f"Best model score: {best_score:.6f}")

        if self.feature_importances:
            self.logger.info("Top 10 feature importances:")
            sorted_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, importance in sorted_features:
                self.logger.info(f"  {feature}: {importance:.4f}")

        return self

    @timeit
    def predict(self, X: np.ndarray, use_all_models: bool = False) -> np.ndarray:
        self.logger.info(f"Generating predictions for {X.shape[0]} samples...")
        
        dtest = xgb.DMatrix(X)
        
        if use_all_models and len(self.models) > 0:
            self.logger.info("Using ensemble prediction from all folds")
            preds = np.zeros(X.shape[0])
            for model in self.models:
                preds += model.predict(dtest)
            preds /= len(self.models)
        elif self.best_model is not None:
            self.logger.info("Using best model for prediction")
            preds = self.best_model.predict(dtest)
        else:
            raise ValueError("No trained models available. Please fit the model first.")
        
        return preds
    
    def get_feature_importances(self) -> Dict[str, float]:
        if self.feature_importances is None:
            self.logger.warning("Feature importances not available. Model may not be trained.")
            return {}
        return self.feature_importances
    
    def save(self, filepath: str, save_all: bool = False) -> None:
        if self.best_model is not None:
            best_model_path = f"{filepath}_best.json"
            self.best_model.save_model(best_model_path)
            self.logger.info(f"Best model saved to {best_model_path}")
            
        if save_all and len(self.models) > 0:
            for i, model in enumerate(self.models):
                fold_path = f"{filepath}_fold_{i}.json"
                model.save_model(fold_path)
            self.logger.info(f"All {len(self.models)} fold models saved")

    def load(self, filepath: str):
        self.best_model = xgb.Booster()
        self.best_model.load_model(filepath)
        self.logger.info(f"Model loaded from {filepath}")

        try:
            self.feature_importances = self.best_model.get_score(importance_type='gain')
        except:
            self.logger.warning("Could not extract feature importances from loaded model")
            
        return self
