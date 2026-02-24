---
type: concept
aliases: [Approximation in RL, Value Approximation]
course: [RL]
tags: [approximation]
status: complete
---

# Optimality and Approximation

> [!definition] The Limit of Exact Solutions
> In large or continuous state spaces, finding the exact optimal policy $\pi^*$ or value function $v^*$ is computationally infeasible. **Approximation** refers to using parameterized functions (like neural networks or linear models) to represent the value functions, accepting a small amount of error to gain the ability to generalize across states.

> [!intuition] Coverage vs. Accuracy
> When we have millions of states (like in Go or Chess), we cannot store a table with one entry per state. Instead, we use a compact representation (a "function approximator"). This means:
> 1. **Generalization**: Learning about one state helps us estimate the value of similar, unvisited states.
> 2. **Prioritization**: We focus our limited "accuracy budget" on the states the agent actually visits.

## Core Concepts

- **Online Learning**: By learning as the agent interacts with the world, the approximation naturally focuses on the most relevant parts of the state space (the [[On-Policy Distribution]]).
- **Functional Forms**: We typically approximate the value function as $v_\pi(s) \approx \hat{v}(s, w)$, where $w$ are the weights of the model.
- **The "Deadly Triad"**: A warning in RL that combining **Function Approximation**, **Bootstrapping**, and **Off-policy learning** can lead to instability and divergence.

## Connections

- Solved using: [[Gradient Descent]], [[Stochastic Gradient Descent]]
- Metric for success: [[Mean Squared Value Error (MSVE)]]
- Fundamental shift from: [[Tabular RL]] (where exact solutions are possible).

## Appears In

- [[RL-L06 - Value Function Approximation]]
