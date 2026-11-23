---
type: concept
aliases: [feature stores, feature store pattern]
course: [EPRM]
tags: [mlops, data-engineering]
status: complete
---

# Feature Store

## Definition

> [!definition] Feature store
> A central system that **catalogues reusable feature-engineering code** and **computes, precomputes and caches feature vectors**, so several models and teams can use the same features without reimplementing them. Examples named by Kästner: Feast, Tecton, AWS SageMaker Feature Store.

## Intuition

> [!intuition] A shared pantry
> If five teams each need "language likely spoken at this GPS location", each writing its own version wastes work and produces five slightly different features. A feature store keeps one version, documented and findable, and can keep the computed values ready.

## Key Properties / Variants

- Solves two problems: **reuse and discovery** of feature code, and **cost** of recomputing expensive features (heavy queries, features from other models) again and again.
- Because training and serving read features from the same place, it also guards against [[Training-Serving Skew]].
- Worth it for large organisations with many models and much data; a single project can share a Python module instead.

## Connections

- Prevents: [[Training-Serving Skew]]
- Part of: [[MLOps]]

## Appears In

- [[EPRM-Book Ch10 - Deploying a Model]]
- [[EPRM-L04 - Deployment Patterns]]
