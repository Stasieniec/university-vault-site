---
type: concept
aliases: [Weight Decay, Overfitting Prevention]
course: [RL, IR]
tags: [foundations]
status: complete
---

# Regularization

## Definition

> [!definition] Regularization
> **Regularization** refers to a set of techniques used to prevent **overfitting** in machine learning models by adding a penalty to the loss function or modifying the learning process. The goal is to constrain the model's complexity so that it generalizes better to unseen data.

## Key Techniques

### 1. Norm Penalties (Weight Decay)
Add a term to the loss function $L$ that penalizes large weights:
- **L2 Regularization (Ridge)**: $L_{reg} = L + \lambda \sum w^2$. Encourages small weights across all features.
- **L1 Regularization (Lasso)**: $L_{reg} = L + \lambda \sum |w|$. Encourages **sparsity** (some weights become exactly zero).

### 2. Dropout
Randomly "dropping out" (setting to zero) a fraction of neurons during each training step.
- **Purpose**: Prevents neurons from co-adapting too much; forces the network to learn redundant representations.

### 3. Early Stopping
Monitoring performance on a validation set and stopping training once validation error starts to increase, even if training error is still decreasing.

### 4. Data Augmentation
Increasing the diversity of the training set by applying transformations (rotation, noise, cropping) to the data.

## Why it works

> [!intuition] Occam's Razor
> In machine learning, a simpler model that explains the data is usually better than a complex one that fits the noise. Regularization effectively "pushes" the model towards simpler solutions (smaller weights, fewer active neurons) unless the data provides overwhelming evidence that complexity is necessary.

## Connections

- Prevents: Overfitting
- Used in: [[Neural Networks]], Linear Regression, SVMs
- Interaction with: [[Gradient Descent]]

## Appears In

- [[RL-L08 - Deep Reinforcement Learning]]
- [[IR-L05 - Neural Information Retrieval]]
- Machine Learning Foundations
