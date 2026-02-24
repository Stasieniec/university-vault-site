---
type: coding-assignment
course: RL
week: 1
language: python
concepts:
  - "[[Dynamic Programming]]"
  - "[[Policy Iteration]]"
  - "[[Value Iteration]]"
  - "[[Bellman Equation]]"
status: complete
---

# RL-CA01: Coding Assignment 1 — Dynamic Programming

## Overview

Implementation of [[Policy Iteration]] and [[Value Iteration]] algorithms on a gridworld environment.

**Files:**
- `DP_lab.ipynb` — Main notebook
- `gridworld.py` — Gridworld environment
- `dp_autograde.py` — Autograding tests

## What You Implement

1. **Policy Evaluation**: Iteratively compute $V^\pi(s)$ for a given policy
2. **Policy Improvement**: Compute greedy policy w.r.t. current $V$
3. **Policy Iteration**: Alternate evaluation + improvement until convergence
4. **Value Iteration**: Single update combining both steps: $V(s) \leftarrow \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')]$

## Key Implementation Details

### Policy Evaluation
```python
# Core loop: iterate until convergence
while delta > theta:
    for s in states:
        v = V[s]
        V[s] = sum(pi[s,a] * sum(p * (r + gamma * V[s_next]) 
                    for p, s_next, r, done in env.P[s][a])
                    for a in actions)
        delta = max(delta, abs(v - V[s]))
```

### Value Iteration
```python
# Core: combine evaluation + improvement in one step
V[s] = max(sum(p * (r + gamma * V[s_next]) 
           for p, s_next, r, done in env.P[s][a])
           for a in actions)
```

## Key Takeaways

- Policy iteration converges in fewer outer iterations but each is expensive (full evaluation)
- Value iteration is simpler (one sweep per iteration) and often faster in total wall-clock time
- Both require the full model $p(s',r|s,a)$ — this is a limitation addressed by [[Monte Carlo Methods]] and [[Temporal Difference Learning]]

## Related Homework

See [[RL-HW01 - Homework 1]] for theoretical questions about these algorithms.
