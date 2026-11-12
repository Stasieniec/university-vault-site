---
type: concept
aliases: [MLFlow, mlflow, experiment tracking]
course: [EPRM]
tags: [mlops, tools]
status: complete
---

# MLflow

## Definition

> [!definition] MLflow
> **MLflow** is an open-source tool for **experiment tracking** (recording each training run's parameters, metrics and files), **model registry and management** (versioning trained models) and **model serving** (exposing a model behind an API).

## Intuition

> [!intuition] A lab notebook that fills itself in
> After twenty training runs you no longer remember which hyperparameters gave the best F1 or where that model file went. MLflow records every run in one place, with its metrics and outputs, and a web UI lets you compare runs and download the model that won.

## Core concepts

- An **experiment** groups related runs (for example "Alpha").
- A **run** is one execution of training code; MLflow names it automatically (for example "industrious-fish-222").
- Per run you log **parameters**, **metrics** (numbers such as `test_f1_mean`) and **artifacts** (any files: a JSON of cross-validation scores, a pickled model).
- The **tracking server** stores runs; code points to it with `mlflow.set_tracking_uri(...)` or the `MLFLOW_TRACKING_URI` environment variable.

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Alpha")
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("test_f1_mean", 0.68)
    mlflow.log_artifact("cv_scores.json")
    mlflow.log_artifact("alpha.pkl")        # the model as a pickle
```

`mlflow server --host 0.0.0.0 --port 5000` starts a tracking server, which is often run in its own container. Flavour-specific functions such as `mlflow.sklearn.log_model` log a model in a format the registry and serving tools understand.

## Key Properties / Variants

- **Optional.** Yesilbek calls it popular but definitely optional for more than 90% of projects.
- **Another dependency.** A tracking server is an external system the application talks to, so it belongs in integration tests and in the environment setup.

## Connections

- Part of: [[MLOps]] (experiment tracking feeds model evaluation and release)
- Often runs in: [[Docker]] [[Containerization|containers]]
- Tested with: [[Integration Testing]]

## Appears In

- [[EPRM-L03 - Containerization and Software Engineering Processes]]
