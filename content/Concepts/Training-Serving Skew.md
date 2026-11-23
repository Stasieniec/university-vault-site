---
type: concept
aliases: [training-serving skew, training/serving skew, feature encoding consistency, feature state]
course: [EPRM]
tags: [mlops, deployment]
status: complete
---

# Training-Serving Skew

## Definition

> [!definition] Training-serving skew
> A mismatch between how inputs are turned into features during training and during inference (serving). The model then receives runtime data that does not look like its training data, and its predictions are wrong, often without any error being raised.

## Intuition

> [!intuition] Same recipe, different oven
> A model learned that "amount z-score 2.0" means unusually large. If serving computes the z-score with a different mean, or scales the column differently, 2.0 now means something else, and the model happily answers anyway.

## How it happens and how to prevent it

- **Different code paths.** Training encodes features in a notebook or Spark job, serving re-implements them in the API. Fix: write the encoding once as a function or module and import it from both training and serving.
- **Different state.** Encodings such as a z-score, a category index or a one-hot vocabulary depend on numbers learned from the training data (means, standard deviations, category lists). Serving must use those saved numbers, never ones computed from the request, which for a single row cannot even give a standard deviation. Fix: save the state as artifacts next to the model (for example in the same [[MLflow]] run) and load model and state together.
- **Version drift.** Model and encoding code are updated separately. Fix: version the encoding code together with the model.
- Kästner: the inconsistency can be as small as one feature scaled differently or as large as scrambled inputs (colour channels in a different order).

## Connections

- Shared encodings at company scale: [[Feature Store]]
- Arises in every pattern of [[ML Deployment Patterns]], most visibly behind an [[HTTP API]]
- Part of: [[MLOps]]

## Appears In

- [[EPRM-L04 - Deployment Patterns]]
- [[EPRM-Book Ch10 - Deploying a Model]]
