---
type: concept
aliases: [every-visit MC, every-visit Monte Carlo]
course: [RL]
tags: [tabular-methods]
status: complete
---

# Every-Visit MC

> [!definition] Every-Visit MC
> A [[Monte Carlo Methods|Monte Carlo]] prediction method that averages returns from **every** visit to a state $s$ within each episode, not just the first. If state $s$ is visited $k$ times in an episode, all $k$ returns are used.

## Intuition

Where [[First-Visit MC]] only uses the return from the *first* time a state is encountered per episode, Every-Visit MC uses *all* visits. This means a single episode can contribute multiple samples for the same state — yielding more data per episode but with correlated samples.

## Algorithm

```pseudo
Algorithm: Every-Visit MC Prediction (estimating V ≈ v_π)
─────────────────────────────────────────────────────────
Input: policy π to evaluate
Initialize:
  V(s) ∈ ℝ arbitrarily, for all s ∈ S
  Returns(s) ← empty list, for all s ∈ S

Loop forever (for each episode):
  Generate episode following π: S₀,A₀,R₁, ..., S_{T-1},A_{T-1},R_T
  G ← 0
  Loop t = T-1, T-2, ..., 0:
    G ← γG + R_{t+1}
    Append G to Returns(S_t)          // ← no first-visit check!
    V(S_t) ← average(Returns(S_t))
```

The only difference from [[First-Visit MC]] is the removal of the "unless $S_t$ appears in $S_0, \ldots, S_{t-1}$" check.

## Mathematical Properties

- **Biased** for finite samples (returns within the same episode are correlated)
- **Consistent**: converges to $v_\pi(s)$ as the number of episodes → ∞
- **Lower variance per update** compared to first-visit when states are revisited often
- Converges **quadratically** — slightly faster in practice for many problems

## Comparison with First-Visit MC

| Property | First-Visit MC | Every-Visit MC |
|----------|---------------|----------------|
| Samples per episode per state | 1 | Up to $k$ (number of visits) |
| Bias | Unbiased | Biased (finite samples) |
| Consistency | ✓ | ✓ |
| Data efficiency | Less | More (uses all visits) |
| Implementation | Needs visit tracking | Simpler |

## Connections

- Core variant of [[Monte Carlo Methods]]
- Compared with [[First-Visit MC]] — the other main MC prediction variant
- Both lead into [[Monte Carlo Control]] when combined with [[Generalized Policy Iteration]]
- Extended to action values for [[Monte Carlo Control]] with [[Exploring Starts]] or [[Epsilon-Greedy Policy|ε-greedy]] policies

## Appears In

- [[RL-L03 - Monte Carlo Methods]]
- [[RL-Book Ch5 - Monte Carlo Methods]] (§5.1)
- [[RL-ES02 - Exercise Set Week 2]]
