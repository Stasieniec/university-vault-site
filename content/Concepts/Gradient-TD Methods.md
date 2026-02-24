---
type: concept
aliases: [GTD, GTD2, TDC, gradient TD]
course: [RL]
tags: [approximation, exam-topic]
status: complete
---

# Gradient-TD Methods

> [!definition] Gradient-TD Methods
> **Gradient-TD methods** (GTD, GTD2, TDC) are true stochastic gradient descent methods that converge even with **off-policy** data and **[[Linear Function Approximation]]**. They avoid the [[Deadly Triad]] by performing gradient descent on a proper objective function (the projected Bellman error or the mean squared TD error).

## Why Needed

[[Semi-Gradient Methods]] can **diverge** with off-policy + function approximation. Gradient-TD methods fix this by computing the **full gradient** (including the gradient through the target).

## Key Algorithms

### GTD2

> [!formula] GTD2 Update
> $$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left(\mathbf{x}_t - \gamma \mathbf{x}_{t+1}\right)(\mathbf{x}_t^\top \mathbf{v}_t)$$
> $$\mathbf{v}_{t+1} = \mathbf{v}_t + \beta \left(\delta_t - \mathbf{x}_t^\top \mathbf{v}_t\right)\mathbf{x}_t$$
> 
> Uses an auxiliary weight vector $\mathbf{v}$ to estimate the expected TD error direction.

### TDC (TD with gradient Correction)

> [!formula] TDC Update
> $$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[\delta_t \mathbf{x}_t - \gamma (\mathbf{x}_t^\top \mathbf{v}_t) \mathbf{x}_{t+1}\right]$$
> $$\mathbf{v}_{t+1} = \mathbf{v}_t + \beta \left(\delta_t - \mathbf{x}_t^\top \mathbf{v}_t\right)\mathbf{x}_t$$
> 
> First term = semi-gradient TD. Second term = **correction** that makes it a true gradient method.

## Trade-offs

- ✅ Converges off-policy with linear FA (solves the deadly triad)
- ❌ Two sets of weights to maintain ($\mathbf{w}$ and $\mathbf{v}$)
- ❌ Two step sizes to tune ($\alpha$ and $\beta$)
- ❌ Slower convergence than semi-gradient TD (when semi-gradient converges)

## Connections

- Fixes: [[Deadly Triad]] (for linear case)
- Alternative to: [[Semi-Gradient Methods]] (off-policy)
- Related: [[LSTD]] (also solves the same fixed point, but in closed form)

## Appears In

- [[RL-L07 - Off-Policy RL with Approximation]]
- [[RL-Book Ch11 - Off-Policy Methods with Approximation]] (§11.7)
