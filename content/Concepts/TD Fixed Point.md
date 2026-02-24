---
type: concept
aliases: [TD fixed point]
course: [RL]
tags: [approximation, key-formula, exam-topic]
status: complete
---

# TD Fixed Point

> [!definition] TD Fixed Point
> The **TD fixed point** $\mathbf{w}_{TD}$ is the weight vector where the expected semi-gradient TD(0) update is zero. For [[Linear Function Approximation]], this is the unique solution to $\mathbf{A}\mathbf{w} = \mathbf{b}$.

> [!formula] TD Fixed Point Equation
> $$\mathbf{A}\mathbf{w}_{TD} = \mathbf{b}$$
> where:
> $$\mathbf{A} = \mathbb{E}\left[\mathbf{x}(S_t)\left(\mathbf{x}(S_t) - \gamma \mathbf{x}(S_{t+1})\right)^\top\right]$$
> $$\mathbf{b} = \mathbb{E}\left[R_{t+1} \, \mathbf{x}(S_t)\right]$$

## Error Bound

> [!formula] TD Fixed Point Error Bound
> $$\overline{VE}(\mathbf{w}_{TD}) \leq \frac{1}{1-\gamma} \min_\mathbf{w} \overline{VE}(\mathbf{w})$$
> 
> The error at the TD fixed point is at most $\frac{1}{1-\gamma}$ times the best possible error with the given features.

> [!warning] Not the Minimum
> The TD fixed point is generally NOT the $\mathbf{w}$ that minimizes $\overline{VE}$. MC gradient descent finds the true minimum; TD finds a different (potentially worse) point. The bound above quantifies how much worse it can be.

## Connections

- Solved by: [[LSTD]] (directly), [[Semi-Gradient Methods|Semi-gradient TD]] (iteratively)
- Only guaranteed for: [[Linear Function Approximation]]

## Appears In

- [[RL-L06 - On-Policy TD with Approximation]], [[RL-Book Ch9 - On-Policy Prediction with Approximation]]
