---
type: concept
aliases: [OLS, Linear Least Squares]
course: [RL]
tags: [optimization]
status: complete
---

# Ordinary Least Squares

> [!definition] Ordinary Least Squares (OLS)
> **Ordinary Least Squares** is a linear regression method for estimating the unknown parameters in a linear model by minimizing the sum of the squares of the vertical deviations (residuals) between each data point and the fitted line.

> [!formula] The Objective and Solution
> **Objective**: Minimize the Sum of Squared Residuals (SSR):
> $$L(w) = \sum_{i=1}^n (y_i - x_i^T w)^2 = \|y - Xw\|^2$$
> 
> **Closed-form Solution**:
> $$w = (X^T X)^{-1} X^T y$$
> 
> where:
> - $X$ — the feature matrix (design matrix)
> - $y$ — the vector of target values
> - $w$ — the weight vector we want to find

> [!intuition] Best Fit Line
> Geometrically, OLS finds the hyperplane that is "closest" to all data points simultaneously. The squared penalty means that large outliers have a disproportionately large effect on the final fit, as the model will work harder to reduce a large error than multiple small ones.

## Applications in RL

- **Linear Function Approximation**: Used to find the weights $w$ such that $\hat{v}(s, w) \approx v_\pi(s)$.
- **LSTD (Least-Squares Temporal Difference)**: An efficient offline RL algorithm that uses the OLS closed-form solution to directly compute the value function parameters without iterative gradient descent.

## Connections

- Solved iteratively via: [[Gradient Descent]]
- Compared with: [[Weighted Least Squares]], [[Recursive Least Squares]]
- Foundational for: [[LSTD]], [[Linear Regression]]

## Appears In

- [[RL-L07 - Eligibility Traces]]
- [[RL-L06 - Value Function Approximation]]
