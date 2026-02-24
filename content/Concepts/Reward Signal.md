---
type: concept
aliases: [reward, reward function, return]
course: [RL]
tags: [foundations]
status: complete
---

# Reward Signal

## Definition

> [!definition] Reward Signal ($R_t$)
> The **Reward Signal** is a scalar value $R_t \in \mathbb{R}$ that the environment sends to the agent at each time step. It defines the goal of the Reinforcement Learning problem: the agent's objective is to maximize the total cumulative reward (**return**) it receives over the long run.

## The Reward Hypothesis

> [!tip] The Reward Hypothesis
> "That all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward)." — Sutton & Barto

## Mathematical Formulation

The agent maximizes the **return** $G_t$:

> [!formula] Total Discounted Return
> $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$$
> 
> where:
> - $\gamma \in [0, 1]$ is the **discount factor**, determining the present value of future rewards.

## Key Properties

- **Immediate vs. Delayed**: A reward is immediate feedback, but the agent must often sacrifice immediate reward to achieve higher long-term return.
- **Scalar**: It must be a single number (though multi-objective RL exists, it usually boils down to a weighted scalar).
- **Environment Bound**: The reward is defined by the environment, not the agent. The agent cannot "change the rules" to get more reward.

## Intuition

> [!intuition] What vs. How
> The reward signal should tell the agent **what** you want it to achieve, not **how** to achieve it.
> - *Bad Reward*: Giving a chess AI points for taking pieces (it might take pieces but lose the game).
> - *Good Reward*: +1 for winning, -1 for losing, 0 otherwise.

## Connections

- Input for: [[Temporal Difference Learning]] and [[Q-Learning]]
- Defines: [[Optimal Policy]]
- Linked to: [[Model of the Environment]] ($p(s', r | s, a)$)

## Appears In

- [[RL-L01 - Intro & MDPs]]
- [[RL-Book Ch3 - Finite Markov Decision Processes]]
