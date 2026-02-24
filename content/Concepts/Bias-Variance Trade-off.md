---
type: concept
aliases: [Bias-Variance Decomposition]
course: [RL, IR]
tags: [foundations]
status: complete
---

# Bias-Variance Trade-off

> [!definition] Bias-Variance Trade-off
> The **Bias-Variance Trade-off** is the property of a model that the error in its predictions can be decomposed into three components: bias, variance, and irreducible noise. It describes the conflict in trying to simultaneously minimize these two sources of error.

> [!formula] Error Decomposition
> $$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$
> 
> where:
> - $\text{Bias} = \mathbb{E}[\hat{f}(x)] - f(x)$ (how far the average prediction is from the truth)
> - $\text{Variance} = \mathbb{E}[\hat{f}(x)^2] - (\mathbb{E}[\hat{f}(x)])^2$ (how much the prediction varies between different training sets)
> - $\text{Noise} = \sigma^2$ (intrinsic error in the data)

> [!intuition] Underfitting vs. Overfitting
> - **High Bias (**[[Underfitting]]**)**: The model is too simple to capture the underlying patterns (e.g., using a straight line for quadratic data). It consistently misses the mark.
> - **High Variance (**[[Overfitting]]**)**: The model is too complex and fits the noise in the training data (e.g., using a high-degree polynomial that wiggles through every point). It changes drastically with different training samples.
> - **Trade-off**: Increasing model complexity decreases bias but increases variance.

## Relevance to AI Coursework

- **RL**: Function approximation (FA) in RL involves balancing the bias of the bootstrap targets with the variance of the sampled trajectories.
- **IR**: Model selection for ranking functions (like tuning parameters for [[BM25]]) involves finding the right level of complexity for relevance estimation.

## Connections

- Related to: [[Overfitting]], [[Underfitting]], [[Regularization]]
- Key concept in: [[Machine Learning]], [[Value Function Approximation]]

## Appears In

- [[RL-L06 - Value Function Approximation]]
- [[IR-L04 - Evaluation]]
