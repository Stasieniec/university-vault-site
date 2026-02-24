---
type: concept
aliases: [Bellman error, BE, mean squared Bellman error]
course: [RL]
tags: [approximation, exam-topic]
status: complete
---

# Bellman Error

> [!definition] Bellman Error
> The **Bellman error** at a state $s$ measures how far the current value estimate is from satisfying the [[Bellman Equation]]:
> $$\bar{\delta}_\mathbf{w}(s) = \left(\sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma \hat{v}(s', \mathbf{w})]\right) - \hat{v}(s, \mathbf{w})$$

> [!formula] Mean Squared Bellman Error ($\overline{BE}$)
> $$\overline{BE}(\mathbf{w}) = \sum_s \mu(s) \left[\bar{\delta}_\mathbf{w}(s)\right]^2$$

> [!warning] Bellman Error Is Not Learnable
> A key result from Ch 11.6: the $\overline{BE}$ cannot be learned from data alone — different MDPs can produce identical data but have different $\overline{BE}$ values. This is why gradient methods that minimize $\overline{BE}$ directly are problematic.

Alternative objectives: Projected Bellman Error (PBE), Mean Squared TD Error — these are learnable and used by [[Gradient-TD Methods]].

## Appears In

- [[RL-L07 - Off-Policy RL with Approximation]]
- [[RL-Book Ch11 - Off-Policy Methods with Approximation]] (§11.5-11.6)
