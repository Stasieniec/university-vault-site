---
type: concept
aliases: [state space, S]
course: [RL]
tags: [foundations]
status: complete
---

# State Space

## Definition

> [!definition] State Space ($\mathcal{S}$)
> The **State Space** is the set of all possible states $s$ that an agent can inhabit in a Markov Decision Process (MDP). It defines the scope of what the agent can sense or know about the environment at any given time $t$.

## Types of State Spaces

- **Discrete State Space**: A countable (often finite) set of states.
  - *Example*: A 4x4 Gridworld has 16 discrete states.
- **Continuous State Space**: An infinite, uncountable set of states, often represented as vectors in $\mathbb{R}^n$.
  - *Example*: The joint angles and velocities of a robotic arm.

## Key Properties

- **Size and Complexity**: The size of the state space determines the memory and computational requirements of RL algorithms.
  - Large discrete spaces or continuous spaces usually require [[Function Approximation]] (e.g., neural networks) because a tabular approach (storing values for every $s$) is impossible.
- **Observability**:
  - **Fully Observable**: The agent's state $S_t$ contains all information needed to make an optimal decision (Markov property).
  - **Partially Observable (POMDP)**: The agent only sees an observation $O_t$, which may not uniquely identify the true state of the environment.

## Intuition

> [!intuition] The Environment's Configuration
> think of the state space as the set of all "snapshots" the world can be in. In a game of Chess, the state space is the set of all possible legal board configurations. In a self-driving car, it includes the car's position, speed, and the relative positions of all surrounding obstacles.

## Connections

- Part of: Markov Decision Process (MDP) definition ($S, A, P, R, \gamma$)
- Mapped to actions by: Policy $\pi(a|s)$
- Measured by: Value Function $V(s)$

## Appears In

- [[RL-L01 - Intro & MDPs]]
- [[RL-Book Ch3 - Finite Markov Decision Processes]]
