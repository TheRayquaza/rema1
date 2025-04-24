# Recommendation System - TikTok video

## TL;DR

```bash
docker compose up
```

## Structure

```
.
├── configs/                    # Configs to define each hyperparams / pipeline / metrics
├── data/                       # Data directory
├── dev-docker-compose.yml
├── docker-compose.yml          # "Production" docker compose
├── Dockerfile                  # Dockerfile for "production" usage
├── notebooks/                  # notebooks with each steps: EDA, Model, Fine Tune, Comparison
├── README.md
├── report.pdf                  # Report
├── requirements.txt
├── scripts/                    # Misc scripts
├── src/
│   ├── config.yaml             # best config used for "production"
│   ├── evaluate.py             # evaluate a given model
│   ├── metrics/                # custom metrics
│   ├── models/                 # custom models
│   ├── pipelines/              # custom pieplines
│   ├── train.py                # train a model
│   └── utils/                  # Misc utils
├── tests/
├── .pre-commit-config.yaml
├── .gitignore
└── .github/workflows/ci.yml
```

## Deploy with docker

```bash
docker build -t rema .
docker run -d -p 8080:8080 rema
```

| Variable         | Default Value            | Description                                 |
|-----------------|---------------------------|---------------------------------------------|
| `DEBUG`         | `false`                   | Enables debug mode (`true` or `false`).     |
| `DEFAULT_MODEL` | `mutlinomial_l2_lbfgs`    | The default model used for training.        |
| `DEFAULT_MODEL_DIR` | `../configs`          | Directory containing model configurations.  |
| `APP_PORT`      | `8080`                    | Application port.                           |
