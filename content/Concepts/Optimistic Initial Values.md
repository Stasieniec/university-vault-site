---
type: concept
aliases: [Optimistic Initialization]
course: [RL]
tags: [foundations, exam-topic]
status: complete
---

# Optimistic Initial Values

> [!definition] Optimistic Initial Values
> **Optimistic Initial Values** is a simple exploration technique in Reinforcement Learning where the initial estimates of the action-values ($Q$-values) are set to be significantly higher than the expected returns.

> [!intuition] Forced Exploration
> By starting with very high estimates (e.g., $Q_0(a) = +5$ when rewards are in $[0,1]$), the agent is consistently "disappointed" by the actual rewards it receives. This disappointment causes the $Q$-value for the tried action to decrease, making other (unexplored) actions appear more attractive. Consequently, the agent is forced to try every action multiple times before settling on the truly best one.

## Key Properties

- **Exploration for Free**: Unlike $\epsilon$-greedy where exploration is random and persistent, optimistic initialization leads to targeted exploration at the beginning of learning.
- **Symmetry Breaking**: Encourages trying all actions to resolve the initial "optimism."
- **Limitations**: It only helps with *initial* exploration; once the values have converged to their true means, it provides no further exploration. It is not helpful for non-stationary problems where the environment changes over time.

## Examples

> [!example] 10-Armed Bandit
> If the true rewards are around 0, setting $Q_1(a) = 5$ for all $a$ will make the agent explore all 10 arms before the values settle down. Even if the first arm chosen gives a reward of 1, the new average will be lower than 5, making the other 9 arms look better.

## Connections

- Comparison: [[$\epsilon$-greedy Exploration]]
- Specific strategy for: [[Exploration-Exploitation Trade-off]]

## Appears In

- [[RL-L01 - Introduction and Multi-armed Bandits]]
