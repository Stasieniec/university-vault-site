---
type: concept
aliases: [policy improvement, policy improvement theorem]
course: [RL]
tags: [tabular-methods, exam-topic]
status: complete
---

# Policy Improvement

> [!definition] Policy Improvement Theorem
> If $q_\pi(s, \pi'(s)) \geq v_\pi(s)$ for all $s$, then $\pi'$ is at least as good as $\pi$: $v_{\pi'}(s) \geq v_\pi(s)$ for all $s$.

The greedy policy $\pi'(s) = \arg\max_a q_\pi(s,a)$ satisfies this condition (by definition of max). Therefore, making the policy greedy w.r.t. its own value function always improves it (or leaves it unchanged if already optimal).

This theorem justifies the improvement step in [[Policy Iteration]] and [[Generalized Policy Iteration]].

## Appears In

- [[RL-L02 - Dynamic Programming]], [[RL-Book Ch4 - Dynamic Programming]]
