---
type: concept
aliases: [optimal policy, π*]
course: [RL]
tags: [foundations, exam-topic]
status: complete
---

# Optimal Policy

## Definition

> [!definition] Optimal Policy ($\pi_*$)
> A policy $\pi$ is defined to be better than or equal to a policy $\pi'$ if its expected return is greater than or equal to that of $\pi'$ for all states. An **optimal policy** $\pi_*$ is any policy that is better than or equal to all other policies.

## Key Properties

- **Existence**: At least one optimal policy always exists for any Markov Decision Process (MDP).
- **Shared Value Function**: All optimal policies share the same **optimal state-value function** $v_*$ and **optimal action-value function** $q_*$.
- **Greedy Selection**: Once $q_*$ is known, an optimal policy can be found by being greedy with respect to it:
  $$\pi_*(s) = \arg\max_a q_*(s, a)$$
- **Uniqueness**: While the value functions $v_*$ and $q_*$ are unique, the optimal policy $\pi_*$ may not be (e.g., if multiple actions share the same maximum value).

## Mathematical Relation

> [!formula] Relationship to Optimal Value Function
> $$v_*(s) = \max_\pi v_\pi(s) \quad \forall s \in \mathcal{S}$$
> $$q_*(s, a) = \max_\pi q_\pi(s, a) \quad \forall s \in \mathcal{S}, a \in \mathcal{A}$$
> 
> The Bellman Optimality Equation for $v_*$ is:
> $$v_*(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_*(s')]$$

## Intuition

> [!intuition] The Ceiling of Performance
> The optimal policy represents the best possible way to behave in an environment. Reinforcement Learning algorithms (like [[Q-Learning]] or [[REINFORCE]]) are essentially searching for this policy or its corresponding value function.

## Connections

- Attained when solving: [[Bellman Equation]] (optimality version)
- Goal of: [[Q-Learning]] (approximates $q_*$)
- Foundation for: MDP theory

## Appears In

- [[RL-L01 - Intro & MDPs]]
- [[RL-L02 - Bellman Equations]]
- [[RL-Book Ch3 - Finite Markov Decision Processes]]
