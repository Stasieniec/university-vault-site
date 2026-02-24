---
type: concept
aliases: [Monte Carlo Policy Gradient]
course: [RL]
tags: [policy-gradient, exam-topic]
status: complete
---

# REINFORCE

## Definition

> [!definition] REINFORCE
> **REINFORCE** (also known as Monte Carlo Policy Gradient) is a policy gradient algorithm that uses the observed return from full episodes to estimate the gradient of the performance objective. It was introduced by Ronald Williams in 1992.

## Algorithm

The algorithm follows the [[Policy Gradient Theorem]] by using the return $G_t$ as an unbiased (but high-variance) estimate of $q_\pi(S_t, A_t)$.

```pseudo
Algorithm: REINFORCE (Monte Carlo Policy Gradient)
────────────────────────────────────────────────────
Input: a differentiable policy parameterization π(a|s,θ)
Algorithm parameters: step size α > 0
Initialize policy parameters θ ∈ ℝ^d (e.g., to 0)

Loop forever (for each episode):
  Generate an episode S₀, A₀, R₁, ..., S_{T-1}, A_{T-1}, R_T following π(·|·,θ)
  Loop for each step of the episode t = 0, 1, ..., T-1:
    G ← Σ_{k=t+1}^{T} γ^{k-t-1} R_k
    θ ← θ + α γ^t G ∇ log π(A_t|S_t, θ)
```

## Update Rule

> [!formula] REINFORCE Update
> $$\theta_{t+1} = \theta_t + \alpha G_t \nabla \log \pi_\theta(A_t|S_t)$$
> 
> where:
> - $G_t$ — Total discounted return from time $t$ to end of episode
> - $\nabla \log \pi_\theta(A_t|S_t)$ — Direction in parameter space that increases the probability of taking action $A_t$ in state $S_t$

## Key Properties

- **Monte Carlo**: Requires full episodes to calculate $G_t$.
- **Unbiased**: The gradient estimate is unbiased.
- **High Variance**: Because $G_t$ depends on all future rewards and actions in the episode, updates can be very noisy.
- **Simplest PG**: Often the first policy gradient method taught because it doesn't require a Critic/Value Function.

## REINFORCE with Baseline

To reduce variance, a **baseline** $b(s)$ (often a learned state-value function $V(s)$) is subtracted from the return:
$$\theta_{t+1} = \theta_t + \alpha (G_t - b(S_t)) \nabla \log \pi_\theta(A_t|S_t)$$
This maintains unbiasedness while reducing the magnitude of the updates.

## Connections

- Direct application of: [[Policy Gradient Theorem]]
- Precursor to: [[Actor-Critic]] (which bootstraps instead of using full $G_t$)
- Limited by: High variance and off-policy instability

## Appears In

- future Week 5 lecture
