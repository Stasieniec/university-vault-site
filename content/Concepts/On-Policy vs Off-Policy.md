---
type: concept
aliases: [on-policy, off-policy, on-policy vs off-policy]
course: [RL]
tags: [foundations, exam-topic]
status: complete
---

# On-Policy vs Off-Policy

> [!definition] On-Policy vs Off-Policy
> - **On-policy**: The agent learns about the **same policy** it uses to make decisions. The behavior policy $b$ equals the target policy $\pi$.
> - **Off-policy**: The agent learns about a **different policy** (target $\pi$) from the one generating data (behavior $b$). Requires $b \neq \pi$.

## Comparison

| Property | On-Policy | Off-Policy |
|----------|-----------|-----------|
| Example | [[SARSA]], On-policy MC | [[Q-Learning]], Off-policy MC |
| Behavior = Target? | Yes ($b = \pi$) | No ($b \neq \pi$) |
| Needs IS correction? | No | Sometimes ([[Importance Sampling]]) |
| Can reuse old data? | No (data becomes stale) | Yes (with corrections) |
| Convergence | Generally more stable | Can diverge with FA ([[Deadly Triad]]) |
| Explores how? | Must be built into the policy (ε-greedy) | Behavior policy can be anything exploratory |

> [!intuition] The Key Benefit of Off-Policy
> Off-policy methods can learn the optimal policy while following an exploratory policy. They can also learn from demonstrations, old data, or other agents' experience. This flexibility comes at the cost of potential instability.

> [!tip] Q-Learning's Trick
> [[Q-Learning]] is off-policy but **doesn't need importance sampling** for control. The $\max_a$ in its update directly targets the greedy (optimal) policy. This is a special property of Q-learning — most off-policy methods need IS corrections.

## Connections

- On-policy algorithms: [[SARSA]], [[Monte Carlo Methods|On-policy MC]], [[Semi-Gradient Methods|Semi-gradient TD]]
- Off-policy algorithms: [[Q-Learning]], [[Deep Q-Network (DQN)]], Off-policy MC
- Correction technique: [[Importance Sampling]]
- Danger: [[Deadly Triad]]

## Appears In

- [[RL-L03 - Monte Carlo Methods]], [[RL-L04 - Temporal Difference Learning]], [[RL-L07 - Off-Policy RL with Approximation]]
