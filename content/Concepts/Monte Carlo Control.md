---
type: concept
aliases: [MC control, Monte Carlo control]
course: [RL]
tags: [tabular-methods]
status: complete
---

# Monte Carlo Control

Finding optimal policies using [[Monte Carlo Methods]]. Learns $q_\pi(s,a)$ (not $v_\pi(s)$) to enable model-free policy improvement.

Three main variants:
1. **MC with [[Exploring Starts]]**: Guarantees coverage but unrealistic
2. **On-policy MC** ([[Epsilon-Greedy Policy]]): Practical, converges to best ε-soft policy
3. **Off-policy MC** ([[Importance Sampling]]): Learns optimal policy from exploratory data

All follow the [[Generalized Policy Iteration]] framework.

See [[RL-L03 - Monte Carlo Methods]] for full algorithms.
