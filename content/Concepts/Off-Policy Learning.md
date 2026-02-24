---
type: concept
aliases: [Off-policy, Behavior policy vs Target policy]
course: [RL]
tags: [foundations, exam-topic]
status: complete
---

# Off-Policy Learning

## Definition

> [!definition] Off-Policy Learning
> **Off-policy learning** is a reinforcement learning paradigm where the **target policy** $\pi$ (the policy being learned) is different from the **behavior policy** $b$ (the policy used to generate data/interact with the environment).

## Key Components

- **Target Policy ($\pi$)**: The policy we want to evaluate or optimize (often the greedy policy).
- **Behavior Policy ($b$)**: The policy used to explore and collect experience (often $\epsilon$-greedy).

## Intuition

> [!intuition] Learning by Watching
> Off-policy learning is like learning to drive by watching a movie of someone else driving. You can evaluate how good their choices were (target policy) even though you aren't the one making them (behavior policy). This allows you to "re-watch" old experiences and learn from them even after your driving style has changed.

## Comparison: On vs Off

| Feature | [[On-Policy Learning]] | **Off-Policy Learning** |
|---------|-----------------------|-------------------------|
| Data Source | Current policy | Any policy (old self, human, random) |
| Variance | Typically Lower | Typically Higher (requires [[Importance Sampling]]) |
| Efficiency | Less sample efficient | More efficient (supports [[Experience Replay]]) |
| Stability | Generally more stable | Can be unstable with FA ([[Deadly Triad]]) |

## Mechanisms

To learn off-policy, one must account for the difference in distributions:
1. **Importance Sampling**: Weighting returns by the ratio $\frac{\pi(a|s)}{b(a|s)}$ to correct for the frequency of actions.
2. **Max Operator**: Algorithms like [[Q-Learning]] avoid importance sampling by directly updating towards the maximum possible value, effectively learning the greedy policy regardless of the behavior.

## Connections

- Primary Example: [[Q-Learning]]
- Risk: [[Deadly Triad]] (Function Approximation + Bootstrapping + Off-Policy)
- Enables: [[Experience Replay]]
- Method: [[Importance Sampling]]

## Appears In

- [[RL-L04 - Temporal Difference Learning]]
- [[RL-L05 - Monte Carlo Methods]]
- [[RL-L07 - Eligibility Traces]]
- [[RL-Book Ch5.5 - Off-policy Prediction via Importance Sampling]]
