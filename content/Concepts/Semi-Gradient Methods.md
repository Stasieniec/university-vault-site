---
type: concept
aliases: [semi-gradient, semi-gradient methods, semi-gradient TD]
course: [RL]
tags: [approximation, key-formula, exam-topic]
status: complete
---

# Semi-Gradient Methods

## Definition

> [!definition] Semi-Gradient Methods
> **Semi-gradient methods** are [[Function Approximation|function approximation]] methods where the gradient is computed only with respect to the weight vector in the estimate being updated, **ignoring** the effect of $\mathbf{w}$ on the target. They are called "semi-gradient" because they don't follow the true gradient of any objective function.

## Why "Semi"?

> [!intuition] The Missing Half of the Gradient
> Consider the TD(0) update target: $R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w})$.
> 
> The true gradient of the loss $[v_\pi(S_t) - \hat{v}(S_t, \mathbf{w})]^2$ would require differentiating through both $\hat{v}(S_t, \mathbf{w})$ AND $\hat{v}(S_{t+1}, \mathbf{w})$ (which appears in the target). Semi-gradient methods **treat the target as a constant** — they only differentiate $\hat{v}(S_t, \mathbf{w})$.
> 
> This makes the update simpler and often works well in practice, but it means we're not doing true gradient descent on any well-defined loss function.

## Semi-Gradient TD(0)

> [!formula] Semi-Gradient TD(0) Update
> $$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \underbrace{\left[ R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t) \right]}_{\delta_t \text{ (TD error)}} \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w}_t)$$
> 
> Note: the gradient $\nabla$ is only of $\hat{v}(S_t, \mathbf{w})$, NOT of the target $R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w})$.

```pseudo
Algorithm: Semi-Gradient TD(0) for estimating v̂ ≈ v_π
──────────────────────────────────────────────────────
Input: policy π, step-size α, differentiable v̂(s,w)
Initialize: w arbitrarily (e.g., w = 0)

Loop for each episode:
  Initialize S
  Loop for each step of episode:
    Choose A ~ π(·|S)
    Take action A, observe R, S'
    If S' is terminal:
      w ← w + α[R - v̂(S,w)] ∇v̂(S,w)
      Go to next episode
    w ← w + α[R + γv̂(S',w) - v̂(S,w)] ∇v̂(S,w)
    S ← S'
```

## For Linear Function Approximation

With $\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)$, the gradient simplifies:
$$\nabla_\mathbf{w} \hat{v}(s, \mathbf{w}) = \mathbf{x}(s)$$

So the update becomes:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \delta_t \, \mathbf{x}(S_t)$$

## Convergence Properties

- **Linear semi-gradient TD(0)** converges to the **[[TD Fixed Point]]**: $\mathbf{w}_{TD}$ where $\overline{VE}(\mathbf{w}_{TD}) \leq \frac{1}{1-\gamma} \min_\mathbf{w} \overline{VE}(\mathbf{w})$
- **Not** guaranteed to converge to the global minimum of $\overline{VE}$
- With **non-linear** approximators (neural nets): no convergence guarantees in general
- **Off-policy**: can diverge (see [[Deadly Triad]])

> [!warning] Semi-Gradient ≠ Convergence to Optimal
> Linear semi-gradient TD doesn't find the $\mathbf{w}$ that minimizes $\overline{VE}$. It finds the TD fixed point, which is bounded by $\frac{1}{1-\gamma}$ times the best possible error. For $\gamma$ close to 1, this bound can be loose.

## Semi-Gradient Control

### Semi-Gradient Sarsa
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t) \right] \nabla_\mathbf{w} \hat{q}(S_t, A_t, \mathbf{w}_t)$$

See [[Episodic Semi-Gradient Control]] for the full algorithm.

## Connections

- Extends: [[Temporal Difference Learning]] to function approximation
- Types: [[Linear Function Approximation]], [[Neural Network Function Approximation]]
- Alternative: [[LSTD]] (closed-form solution for linear case)
- Danger: [[Deadly Triad]] (off-policy + bootstrapping + FA)
- True gradient alternatives: [[Gradient-TD Methods]] (TDC, GTD2)

## Appears In

- [[RL-L05 - Tabular to Approximation]]
- [[RL-L06 - On-Policy TD with Approximation]]
- [[RL-L07 - Off-Policy RL with Approximation]]
- [[RL-Book Ch9 - On-Policy Prediction with Approximation]]
