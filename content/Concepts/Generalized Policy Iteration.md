---
type: concept
aliases: [GPI, generalized policy iteration]
course: [RL]
tags: [foundations, exam-topic]
status: complete
---

# Generalized Policy Iteration (GPI)

> [!definition] Generalized Policy Iteration
> **GPI** is the general framework of letting policy evaluation and policy improvement interact, regardless of the granularity of each step. Almost all RL methods can be viewed as instances of GPI.

```
    Evaluation:  v ≈ v_π
         ↕
    Improvement: π ≈ greedy(v)
         ↓
    Both converge → v* and π*
```

> [!intuition] The Push and Pull
> Evaluation and improvement work against each other: improvement makes the policy greedy w.r.t. the current value function (which makes the value function wrong for the new policy), and evaluation updates the value function to match the current policy (which makes the policy suboptimal). But they jointly drive toward optimality — like two competing processes that converge to equilibrium.

## Instances of GPI

| Algorithm | Evaluation step | Improvement step |
|-----------|----------------|-----------------|
| [[Policy Iteration]] | Full convergence of $V$ | Greedy update of $\pi$ |
| [[Value Iteration]] | One sweep of $V$ | Implicit greedy (built into update) |
| [[Monte Carlo Methods]] | Estimate Q from episodes | ε-greedy w.r.t. Q |
| [[SARSA]] | One TD step on Q | ε-greedy w.r.t. Q |
| [[Q-Learning]] | One TD step on Q | Greedy w.r.t. Q |

## Appears In

- [[RL-L02 - Dynamic Programming]], [[RL-L03 - Monte Carlo Methods]], [[RL-L04 - Temporal Difference Learning]]
- [[RL-Book Ch4 - Dynamic Programming]]
