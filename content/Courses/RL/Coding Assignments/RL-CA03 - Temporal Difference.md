---
type: coding-assignment
course: RL
week: 3
language: python
concepts:
  - "[[Temporal Difference Learning]]"
  - "[[SARSA]]"
  - "[[Q-Learning]]"
  - "[[On-Policy vs Off-Policy]]"
  - "[[Epsilon-Greedy Policy]]"
status: complete
---

# RL-CA03: Coding Assignment 3 — Temporal Difference Learning

## Overview

Implementation of SARSA and Q-Learning on Windy Gridworld and Cliff Walking environments.

**Files:**
- `TD_lab.ipynb` — Main notebook  
- `windy_gridworld.py` — Windy Gridworld environment

## What You Implement

1. **[[SARSA]]**: On-policy TD control
2. **[[Q-Learning]]**: Off-policy TD control
3. Comparison of both on different environments

## Key Implementation Details

### SARSA (On-Policy)
```python
# Choose A from S using ε-greedy
A = epsilon_greedy(Q, S, epsilon)
for each step:
    # Take action, observe next state and reward
    S_next, R, done = env.step(A)
    # Choose A' from S' using ε-greedy (ON-POLICY: same policy)
    A_next = epsilon_greedy(Q, S_next, epsilon)
    # Update: uses actual next action A'
    Q[S,A] += alpha * (R + gamma * Q[S_next, A_next] - Q[S,A])
    S, A = S_next, A_next
```

### Q-Learning (Off-Policy)
```python
for each step:
    A = epsilon_greedy(Q, S, epsilon)
    S_next, R, done = env.step(A)
    # Update: uses MAX over next actions (OFF-POLICY: greedy target)
    Q[S,A] += alpha * (R + gamma * max(Q[S_next, :]) - Q[S,A])
    S = S_next
```

## Key Observations

### Windy Gridworld Results
- Both algorithms learn to navigate the grid with wind effects
- Compare average returns during training

### SARSA vs Q-Learning Behavior
> [!example] Cliff Walking Phenomenon
> - **Q-learning** learns the optimal (shortest) path **along the cliff edge** — risky with ε-greedy exploration
> - **SARSA** learns a **safer path** further from the cliff — accounts for exploration randomness
> - Q-learning has higher optimal-policy value but lower average training return

## Key Takeaways

- SARSA: safe, accounts for exploration in learned values
- Q-learning: finds optimal policy, but ε-greedy execution can be risky
- They converge to the same thing when $\varepsilon = 0$

## Related Homework

See [[RL-HW03 - Homework 3]] for theoretical questions about these algorithms, plus [[Function Approximation]] problems.
