---
type: concept
aliases: [GD, Steepest Descent]
course: [RL, IR]
tags: [optimization]
status: complete
---

# Gradient Descent

> [!definition] Gradient Descent
> **Gradient Descent** is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. The algorithm takes steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point.

> [!formula] Update Rule
> $$w_{t+1} = w_t - \alpha \nabla L(w_t)$$
> 
> where:
> - $w_t$ — parameters at iteration $t$
> - $\alpha$ — learning rate (step size), $\alpha > 0$
> - $\nabla L(w_t)$ — gradient of the loss function $L$ with respect to $w$ at $w_t$

> [!intuition] Moving Downhill
> Imagine standing on a hilly landscape in a fog. To find the bottom, you check the slope under your feet and take a step in the steepest downward direction. The learning rate $\alpha$ determines the size of your steps—too large and you might overstep the valley; too small and it will take forever to reach the bottom.

## Key Considerations

- **Learning Rate $\alpha$**: Crucial hyperparameter. Small steps ensure convergence but are slow. Large steps can cause oscillation or divergence.
- **Local Minima**: In non-convex functions, GD can get stuck in local minima rather than the global minimum.
- **Saddle Points**: Points where the gradient is zero but are not local extrema. These can significantly slow down GD.
- **Vanishing/Exploding Gradients**: Issues in deep networks where gradients become extremely small or large, preventing effective updates.

## Connections

- Foundation for: [[Stochastic Gradient Descent]], [[Adam]], [[RMSProp]]
- Used in: [[Neural Networks]], [[Logistic Regression]], [[Ordinary Least Squares]] (iterative solution)
- RL context: Used for [[Value Function Approximation]] and [[Policy Gradient Methods]]

## Appears In

- [[RL-L06 - Value Function Approximation]]
- [[IR-L03 - Retrieval Models]]
