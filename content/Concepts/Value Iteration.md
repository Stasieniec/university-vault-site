---
type: concept
aliases: [value iteration]
course: [RL]
tags: [tabular-methods, exam-topic]
status: complete
---

# Value Iteration

> [!definition] Value Iteration
> **Value iteration** combines one sweep of policy evaluation with policy improvement into a single update using the [[Bellman Optimality Equation]]:
> 
> $$V_{k+1}(s) = \max_a \sum_{s',r} p(s',r|s,a)\left[r + \gamma V_k(s')\right]$$

```pseudo
Algorithm: Value Iteration
──────────────────────────
Initialize V(s) arbitrarily (V(terminal) = 0)

Loop until convergence (Δ < θ):
  Δ ← 0
  For each s ∈ S:
    v ← V(s)
    V(s) ← max_a Σ_{s',r} p(s',r|s,a) [r + γV(s')]
    Δ ← max(Δ, |v - V(s)|)

Output: π(s) = argmax_a Σ_{s',r} p(s',r|s,a) [r + γV(s')]
```

> [!intuition] Truncated Policy Iteration
> Value iteration = [[Policy Iteration]] where the evaluation step is truncated to a **single sweep**. It converges because each sweep applies the Bellman optimality operator, which is a γ-contraction.

- Converges to $v_*$ for any initialization
- Often faster than policy iteration in total compute (fewer sweeps overall)
- Policy can be extracted at the end from $V$ using one greedy step

## Appears In

- [[RL-L02 - Dynamic Programming]], [[RL-Book Ch4 - Dynamic Programming]], [[RL-CA01 - Dynamic Programming]]
