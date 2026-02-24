---
type: concept
aliases: [Deep RL, DRL]
course: [RL]
tags: [deep-rl]
status: complete
---

# Deep Reinforcement Learning

## Definition

> [!definition] Deep Reinforcement Learning (Deep RL)
> **Deep Reinforcement Learning** is the field of RL that uses **Deep Neural Networks** as function approximators for value functions ($V$ or $Q$) or policies ($\pi$). This allows RL to scale to high-dimensional state spaces (like raw pixels) where [[Tabular RL]] is impossible due to the curse of dimensionality.

## Challenges

Using deep learning in RL is notoriously unstable due to three main factors:

1. **Non-stationarity**: The target values for the neural network change as the policy improves.
2. **Correlated Samples**: Successive states in an episode are highly correlated, violating the IID assumption of SGD.
3. **The Deadly Triad**: The combination of **Function Approximation**, **Bootstrapping** (TD learning), and **Off-Policy learning** often leads to divergence.

## Key Techniques & Solutions

| Challenge | Solution | Algorithm Example |
|-----------|----------|-------------------|
| Correlated Samples | [[Experience Replay]] | [[DQN]] |
| Moving Targets | **Target Networks** | [[DQN]] |
| High Variance | **Baselines / Critics** | A3C, PPO |
| Overestimation | **Double Learning** | Double DQN |

## Primary Methods

- **Value-Based**: [[Deep Q-Network (DQN)]] and its variants.
- **Policy-Based**: REINFORCE with deep network policies.
- **Actor-Critic**: A3C, A2C, [[PPO]], [[SAC (Soft Actor-Critic)]].

## Connections

- Foundation: [[Reinforcement Learning]], [[Neural Networks]]
- Departure from: [[Tabular RL]]
- Overcomes: Curse of Dimensionality
- Risk: [[Deadly Triad]]

## Appears In

- [[RL-L08 - Deep Reinforcement Learning]]
- [[RL-Book Ch9 - On-policy Prediction with Approximation]]
- [[RL-Book Ch11 - Off-policy Methods with Approximation]]
