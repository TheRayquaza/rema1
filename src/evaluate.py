from omegaconf import DictConfig
from src.models.multinomial import Multinomial
from src.pipelines.simple import SimplePipeline
from src.pipelines.base import AbstractPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.metrics.coverage import Coverage
from src.metrics.ndcg import NDCG
import hydra
from hydra.utils import to_absolute_path

models = {
    "multinomial": Multinomial,
}
pipelines = {
    "simple": SimplePipeline,
}

metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "ndcg": NDCG(),
    "coverage": Coverage(),
}

@hydra.main(config_path="../configs", config_name="mutlinomial_l2_lbfgs")
def evaluate(cfg: DictConfig):
    """Evaluate a model using the configuration provided."""
    # Load model
    model = models[cfg.model.type](cfg.model)
    model_path = to_absolute_path(f"models/{cfg.model.name}/{cfg.model.version}")
    model = model.load(model_path)

    # Load & preprocess data
    pipeline = pipelines[cfg.pipeline.type](cfg.preprocess)
    X_train, y_train, X_test, y_test = pipeline.load()
    X_train, y_train = pipeline.fit(X_train, y_train)

    # Evaluate model
    X_test, y_test = pipeline.transform(X_test, y_test)
    y_pred = model.predict(X_test)

    for metric in cfg.metrics:
        metric_func = metrics[metric.type]
        if metric.type in ["accuracy", "precision", "recall", "f1"]:
            score = metric_func(y_test, y_pred)
        else:
            score = metric_func(y_test, y_pred, **metric.params)

        print(f"{metric.name}: {score}")

if __name__ == "__main__":
    evaluate()
