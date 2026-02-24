---
type: concept
aliases: [policy evaluation, prediction]
course: [RL]
tags: [tabular-methods]
status: complete
---

# Policy Evaluation

> [!definition] Policy Evaluation (Prediction)
> Computing the state-value function $v_\pi(s)$ for a given policy $\pi$. Also called the **prediction problem**.

Iterative update:
$$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V_k(s')]$$

Converges to $v_\pi$ as $k \to \infty$. Used as a subroutine in [[Policy Iteration]].

## Appears In

- [[RL-L02 - Dynamic Programming]], [[RL-Book Ch4 - Dynamic Programming]]
