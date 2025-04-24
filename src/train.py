from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
import os

from src.models.multinomial import Multinomial
from src.pipelines.simple import SimplePipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.metrics.coverage import Coverage
from src.metrics.ndcg import NDCG

models = {
    "multinomial_nb": Multinomial,
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

@hydra.main(config_path="../configs", config_name="multinomial_nb.yaml")
def run(cfg: DictConfig):
    """Train and/or evaluate a model using the provided configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_params = OmegaConf.to_container(cfg.model)
    model_specific_params = {}
    if isinstance(model_params, dict):
        model_specific_params = {f"model.{k}": v for k, v in model_params.items() 
                                if k not in ['version']}

    pipeline_params = OmegaConf.to_container(cfg.pipeline)
    pipeline_specific_params = {}
    if isinstance(pipeline_params, dict):
        pipeline_specific_params = {f"pipeline.{k}": v for k, v in pipeline_params.items()}

    experiment_name = f"{cfg.model.name}_Experiment"

    pipeline = pipelines[cfg.pipeline.type](cfg.pipeline)
    data = pipeline.load()
    X_trans, y_trans = pipeline.transform(data)
    X_train, X_test, y_train, y_test = pipeline.split(X_trans, y_trans)

    model = models[cfg.model.type](cfg.model)
    model.fit(X_train, y_train)

    model_path = to_absolute_path(f"../models/{cfg.model.name}/{timestamp}")
    model.save(model_path)

    y_pred = model.predict(X_test)

    for metric_name in cfg.metrics:
        metric_func = metrics.get(metric_name)
        if metric_func:
            if metric_name in ["precision", "recall", "f1"]:
                score = metric_func(y_test, y_pred, average="weighted")
            else:
                score = metric_func(y_test, y_pred)

            print(f"{metric_name.capitalize()}: {score:.4f}")

    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    print(f"Experiment: {experiment_name}")
    print(f"Model: Example")
    print(f"Version: {timestamp}")
    print("Run `mlflow ui` to view results and access the model in the Model Registry.")

if __name__ == "__main__":
    run()
