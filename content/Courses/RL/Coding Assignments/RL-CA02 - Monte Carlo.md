---
type: coding-assignment
course: RL
week: 2
language: python
concepts:
  - "[[Monte Carlo Methods]]"
  - "[[First-Visit MC]]"
  - "[[Importance Sampling]]"
  - "[[On-Policy vs Off-Policy]]"
  - "[[Epsilon-Greedy Policy]]"
status: complete
---

# RL-CA02: Coding Assignment 2 — Monte Carlo Methods

## Overview

Implementation of Monte Carlo prediction and control methods on a Blackjack environment.

**Files:**
- `MC_lab.ipynb` — Main notebook
- `blackjack.py` — Blackjack environment
- `mc_autograde.py` — Autograding tests

## What You Implement

1. **On-policy MC prediction**: First-visit MC to estimate $V^\pi(s)$ and $Q^\pi(s,a)$
2. **On-policy MC control**: With ε-greedy policy improvement
3. **Off-policy MC prediction**: Using ordinary importance sampling

## Key Implementation Details

### First-Visit MC Prediction
```python
# For each episode:
# 1. Generate episode following pi
# 2. Walk backwards through episode
# 3. Compute returns G, update V(s) with running average
G = 0
for t in reversed(range(len(episode))):
    s, a, r = episode[t]
    G = gamma * G + r
    if s not in [x[0] for x in episode[:t]]:  # first-visit check
        N[s] += 1
        V[s] += (G - V[s]) / N[s]  # incremental average
```

### Off-policy with Ordinary IS
Key: importance sampling ratio for episode from $t$:
$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

Incremental update (from [[RL-HW02 - Homework 2|HW2 Q1a]]):
$$V_n = V_{n-1} + \frac{1}{n}(\rho \cdot G - V_{n-1})$$

## Key Takeaways

- MC is model-free — doesn't need transition probabilities
- First-visit MC: simpler, unbiased
- Ordinary IS: unbiased but high variance (visible in the plots)
- Weighted IS: biased but much lower variance (smoother convergence)

## Related Homework

See [[RL-HW02 - Homework 2]] for theoretical questions.
