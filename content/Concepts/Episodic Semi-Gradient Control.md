---
type: concept
aliases: [episodic semi-gradient control, semi-gradient Sarsa]
course: [RL]
tags: [approximation, exam-topic]
status: complete
---

# Episodic Semi-Gradient Control

> [!definition] Episodic Semi-Gradient Control
> Extension of [[Semi-Gradient Methods]] to the **control** setting using action-value approximation $\hat{q}(s, a, \mathbf{w})$.

> [!formula] Semi-Gradient Sarsa Update
> $$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t) \right] \nabla_\mathbf{w} \hat{q}(S_t, A_t, \mathbf{w}_t)$$

```pseudo
Algorithm: Episodic Semi-Gradient Sarsa
───────────────────────────────────────
Initialize w arbitrarily
Loop for each episode:
  S ← initial state; A ← ε-greedy(q̂(S,·,w))
  Loop for each step:
    Take A, observe R, S'
    If S' is terminal:
      w ← w + α[R - q̂(S,A,w)] ∇q̂(S,A,w)
      Go to next episode
    A' ← ε-greedy(q̂(S',·,w))
    w ← w + α[R + γq̂(S',A',w) - q̂(S,A,w)] ∇q̂(S,A,w)
    S ← S'; A ← A'
```

With linear FA: $\hat{q}(s,a,\mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s,a)$, so $\nabla = \mathbf{x}(s,a)$.

## Appears In

- [[RL-L07 - Off-Policy RL with Approximation]] (Ch 10.1)
- [[RL-Book Ch10 - On-Policy Control with Approximation]]
