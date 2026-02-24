---
type: concept
aliases: [Tabular Methods, Discrete RL]
course: [RL]
tags: [foundations]
status: complete
---

# Tabular RL

## Definition

> [!definition] Tabular RL
> **Tabular RL** refers to reinforcement learning methods used in problems with small, discrete state and action spaces. In these settings, value functions ($V$ or $Q$) can be represented as a literal table (or array) where every state or state-action pair has its own dedicated entry.

## Characteristics

- **Exact Representation**: Every state is unique; updating one doesn't affect another.
- **No Generalization**: Learning about one state tells you nothing about a similar state you haven't visited yet.
- **Foundation**: These methods provide the theoretical basis (convergence proofs) before moving to [[Deep Reinforcement Learning|Function Approximation]].

## Main Categories of Methods

1. **Dynamic Programming (DP)**: Requires a model. Includes Value Iteration and Policy Iteration.
2. **Monte Carlo (MC)**: Model-free. Learns from full returns at the end of episodes.
3. **Temporal Difference (TD)**: Model-free. Updates estimates based on other estimates (bootstrapping). Includes [[Q-Learning]] and [[SARSA]].

## Convergence

In the tabular case, most standard RL algorithms are guaranteed to converge to the optimal value function $v_*$ or $q_*$ given sufficient exploration and decaying learning rates.

> [!intuition] The Table is the Map
> Imagine the environment as a small grid. Every cell (state) in the grid has a value written in it. When you move, you just erase the old value and write a better one. There's no "guessing" — just record-keeping.

## Connections

- Contrasts with: [[Deep Reinforcement Learning]] (Function Approximation)
- Includes: [[Q-Learning]], [[SARSA]], [[Monte Carlo Methods]]
- Limitation: Curse of Dimensionality (doesn't scale to large or continuous spaces)

## Appears In

- [[RL-L01 - Introduction to RL]]
- [[RL-L02 - Planning and Learning with Tabular Methods]]
- [[RL-L03 - Dynamic Programming]]
- [[RL-L04 - Temporal Difference Learning]]
- [[RL-L05 - Monte Carlo Methods]]
- [[RL-Book Ch1]] through [[RL-Book Ch8]]
