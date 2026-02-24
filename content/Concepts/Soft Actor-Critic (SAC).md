---
type: concept
aliases: [SAC]
course: [RL]
tags: [deep-rl, exam-topic, actor-critic]
status: complete
---

# Soft Actor-Critic (SAC)

## Definition

> [!definition] Soft Actor-Critic (SAC)
> **Soft Actor-Critic** is an **off-policy** actor-critic algorithm that uses **entropy regularization**. Instead of maximizing just the expected return, the agent maximizes the expected return **plus** the entropy of the policy, encouraging exploration and robustness.

## Objective Function

> [!formula] Soft RL Objective
> $$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$
> 
> where:
> - $\mathcal{H}(\pi(\cdot|s_t)) = -\int \pi(a|s_t) \log \pi(a|s_t) da$ — Entropy of the policy
> - $\alpha$ — Temperature parameter that controls the trade-off between reward and entropy (exploration)

## Key Components

1. **Actor**: Parameterized policy $\pi_\theta$.
2. **Critics**: SAC typically uses **two** soft Q-functions ($Q_{w_1}, Q_{w_2}$) to mitigate overestimation bias (similar to Double DQN).
3. **Target Networks**: Uses moving average versions of the Q-functions for stability.
4. **Experience Replay**: Being off-policy, it stores transitions in a buffer to reuse data.

## Intuition

> [!intuition] Maximum Entropy RL
> By including entropy in the objective, the agent is forced to be as "random" as possible while still obtaining rewards. This prevents the policy from collapsing into a single deterministic action too early. It leads to:
> - **Better exploration**: Testing many promising paths.
> - **Robustness**: The policy can recover from perturbations because it has learned a wider distribution of behavior.

## Key Properties

- **Off-policy**: Very sample efficient compared to on-policy methods (like PPO).
- **Stable**: One of the most stable and reliable deep RL algorithms for continuous control.
- **Continuous Action Spaces**: Primarily designed for continuous tasks (e.g., robotics).

## Connections

- An instance of: [[Actor-Critic]]
- Improves upon: DDPG (which is often unstable)
- Related concepts: [[Reward Signal]] (modified with entropy), [[State Space]]

## Appears In

- future Week 6 lecture
