---
type: concept
aliases: [LSTD, Least-Squares TD, least-squares temporal difference]
course: [RL]
tags: [approximation, key-formula, exam-topic]
status: complete
---

# Least-Squares TD (LSTD)

> [!definition] LSTD
> **LSTD** computes the [[TD Fixed Point]] directly in closed form rather than iteratively. For [[Linear Function Approximation]], the TD fixed point satisfies $\mathbf{A}\mathbf{w} = \mathbf{b}$, and LSTD solves for $\mathbf{w}$ exactly.

> [!formula] LSTD Solution
> $$\mathbf{w}_{TD} = \mathbf{A}^{-1}\mathbf{b}$$
> 
> where:
> $$\mathbf{A} = \sum_{t=0}^{T-1} \mathbf{x}(S_t)\left[\mathbf{x}(S_t) - \gamma \mathbf{x}(S_{t+1})\right]^\top$$
> $$\mathbf{b} = \sum_{t=0}^{T-1} R_{t+1} \, \mathbf{x}(S_t)$$

**Updated incrementally** using Sherman-Morrison:
$$\mathbf{A}_t^{-1} = \mathbf{A}_{t-1}^{-1} - \frac{\mathbf{A}_{t-1}^{-1} \mathbf{x}_t (\mathbf{x}_t - \gamma \mathbf{x}_{t+1})^\top \mathbf{A}_{t-1}^{-1}}{1 + (\mathbf{x}_t - \gamma \mathbf{x}_{t+1})^\top \mathbf{A}_{t-1}^{-1} \mathbf{x}_t}$$

## Trade-offs

| Property | LSTD | Semi-gradient TD |
|----------|------|-----------------|
| Step-size $\alpha$? | No (direct solution) | Yes (sensitive to tuning) |
| Convergence speed | Faster (data-efficient) | Slower (iterative) |
| Computation per step | $O(d^2)$ | $O(d)$ |
| Memory | $O(d^2)$ (stores $\mathbf{A}^{-1}$) | $O(d)$ |

> [!tip] When to Use LSTD
> LSTD is ideal when: data is expensive (few samples), feature dimension $d$ is moderate, and you want to avoid tuning $\alpha$. Not practical for very high-dimensional feature spaces.

## Connections

- Solves: [[TD Fixed Point]]
- For: [[Linear Function Approximation]] only
- Alternative to: [[Semi-Gradient Methods|Semi-gradient TD]]
- Extended by: LSTD(λ), LSPI (for control)

## Appears In

- [[RL-L06 - On-Policy TD with Approximation]]
- [[RL-Book Ch9 - On-Policy Prediction with Approximation]] (§9.8)
