---
type: concept
aliases: [SGD, Mini-batch Gradient Descent]
course: [RL, IR]
tags: [optimization]
status: complete
---

# Stochastic Gradient Descent

> [!definition] Stochastic Gradient Descent (SGD)
> **Stochastic Gradient Descent** is a variation of [[Gradient Descent]] that replaces the actual gradient (calculated from the entire dataset) with an estimate thereof (calculated from a randomly selected subset or a single sample). This reduces the computational burden, allowing for faster iterations and online learning.

> [!formula] SGD Update Rule
> For a single sample $(x_i, y_i)$:
> $$w \leftarrow w - \alpha \nabla L(w; x_i, y_i)$$
> 
> For a mini-batch $\mathcal{B}$:
> $$w \leftarrow w - \alpha \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla L(w; x_i, y_i)$$

> [!intuition] Efficiency through Estimation
> Regular gradient descent requires a full pass over the dataset for every single update. In large-scale machine learning, this is prohibitively slow. SGD assumes that the gradient from a small, representative sample is "good enough" to point in the general direction of the minimum. The noise introduced by the sampling can actually help the optimizer "jump" out of shallow local minima.

## Properties

- **Speed**: Much faster iterations than batch gradient descent.
- **Noise**: The path to the minimum is "noisy" and zig-zags, but it eventually converges (given a decreasing learning rate).
- **Online Learning**: Naturally supports learning from a continuous stream of data without needing to store the whole dataset.
- **Regularization Effect**: The inherent noise in SGD can provide a form of implicit regularization, often leading to better generalization.

## Connections

- Variant of: [[Gradient Descent]]
- Optimization backbone for: [[Neural Networks]], [[Deep Reinforcement Learning]]
- RL context: Essential for [[On-Policy Distribution|On-policy distribution]] updates in large state spaces.

## Appears In

- [[RL-L06 - Value Function Approximation]]
- [[IR-L08 - Neural Information Retrieval]]
