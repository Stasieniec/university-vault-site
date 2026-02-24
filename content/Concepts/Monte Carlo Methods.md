---
type: concept
aliases: [MC methods, Monte Carlo, MC]
course: [RL]
tags: [tabular-methods, exam-topic]
status: complete
---

# Monte Carlo Methods

## Definition

> [!definition] Monte Carlo Methods
> **Monte Carlo (MC) methods** learn value functions and optimal policies from **complete episodes of experience**. They estimate values by averaging actual [[Return|returns]] — no model of the environment required, and no bootstrapping.

## Key Idea

> [!intuition] The Core Insight
> To estimate $v_\pi(s)$: play many episodes, and every time you visit state $s$, record the return $G_t$ you got from there. The average of those returns converges to the true value. That's it. No equations to solve, no model needed — just sample and average.

## MC Prediction

### [[First-Visit MC]]
Averages returns only from the **first** visit to each state within an episode.
- Unbiased estimator of $v_\pi(s)$
- Converges by the Law of Large Numbers

### [[Every-Visit MC]]
Averages returns from **every** visit to each state within an episode.
- Also converges, but individual samples within an episode are correlated
- Slightly biased but asymptotically unbiased

## MC Control

MC control learns $q_\pi(s,a)$ (not $v_\pi(s)$) because action values allow model-free policy improvement.

### With [[Exploring Starts]]
- Every state-action pair has non-zero probability of being the episode start
- Guarantees all pairs are visited → can learn optimal policy
- Unrealistic in practice

### On-Policy (ε-Greedy)
- Uses [[Epsilon-Greedy Policy]] to ensure exploration
- Converges to the best ε-soft policy (not truly optimal, but close for small ε)

### Off-Policy with [[Importance Sampling]]
- Generate data with behavior policy $b$, learn about target policy $\pi$
- Correct distribution mismatch using importance sampling ratios
- Ordinary IS: unbiased but high variance
- Weighted IS: biased (for finite samples) but much lower variance

## Comparison with Other Methods

| Property | MC | [[Temporal Difference Learning\|TD]] | [[Dynamic Programming\|DP]] |
|----------|:---:|:---:|:---:|
| Model-free | ✅ | ✅ | ❌ |
| Bootstraps | ❌ | ✅ | ✅ |
| Complete episodes needed | ✅ | ❌ | N/A |
| Unbiased value estimates | ✅ | ❌ | N/A |
| Variance | High | Low | N/A |
| Works for non-Markov | ✅ | ❌ | ❌ |

## Key Properties

- **No bootstrapping** → unbiased but higher variance than TD
- **Episode-based** → can't be used for continuing tasks (no termination = no return)
- **State estimates independent** → updating $V(s)$ doesn't affect $V(s')$
- **Handles non-Markov** environments (doesn't rely on Markov property)

## Connections

- Estimates: [[Value Function]], [[Return]]
- Exploration mechanisms: [[Exploring Starts]], [[Epsilon-Greedy Policy]], [[Importance Sampling]]
- Compared with: [[Dynamic Programming]], [[Temporal Difference Learning]]
- Framework: [[Generalized Policy Iteration]]

## Appears In

- [[RL-L03 - Monte Carlo Methods]]
- [[RL-Book Ch5 - Monte Carlo Methods]]
- [[RL-ES02 - Exercise Set Week 2]]
- [[RL-CA02 - Monte Carlo]]
