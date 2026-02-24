---
type: concept
aliases: [policy iteration]
course: [RL]
tags: [tabular-methods, exam-topic]
status: complete
---

# Policy Iteration

> [!definition] Policy Iteration
> **Policy iteration** finds the optimal policy by alternating two steps until convergence:
> 1. **Policy Evaluation**: Compute $v_\pi(s)$ for the current policy (iteratively until convergence)
> 2. **Policy Improvement**: Make the policy greedy w.r.t. the value function: $\pi'(s) = \arg\max_a q_\pi(s,a)$

```pseudo
Algorithm: Policy Iteration
───────────────────────────
Initialize V(s) and π(s) arbitrarily for all s

1. Policy Evaluation:
   Loop until convergence:
     For each s ∈ S:
       V(s) ← Σ_{s',r} p(s',r|s,π(s)) [r + γV(s')]

2. Policy Improvement:
   policy_stable ← true
   For each s ∈ S:
     old_action ← π(s)
     π(s) ← argmax_a Σ_{s',r} p(s',r|s,a) [r + γV(s')]
     If old_action ≠ π(s): policy_stable ← false
   
   If policy_stable: stop (found π*)
   Else: go to step 1
```

> [!tip] Policy Iteration vs [[Value Iteration]]
> Policy iteration does **full evaluation** (many sweeps until $V$ converges) then one improvement step. Value iteration does **one sweep** of evaluation combined with improvement. Policy iteration typically needs fewer total iterations but each iteration is more expensive.

## Appears In

- [[RL-L02 - Dynamic Programming]], [[RL-Book Ch4 - Dynamic Programming]], [[RL-CA01 - Dynamic Programming]]
