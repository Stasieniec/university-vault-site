---
type: concept
aliases: [PG methods, Policy Gradient]
course: [RL]
tags: [policy-gradient, exam-topic]
status: complete
---

# Policy Gradient Methods

## Definition

> [!definition] Policy Gradient Methods
> **Policy Gradient (PG) methods** are a family of reinforcement learning algorithms that directly optimize a parameterized policy $\pi(a|s, \theta)$ by following the gradient of some performance measure $J(\theta)$ with respect to the policy parameters $\theta$. Unlike value-based methods, they do not require a value function to select actions (though they may use one for variance reduction).

## The Policy Gradient Theorem

> [!formula] Policy Gradient Theorem
> For any differentiable policy $\pi(a|s, \theta)$, the gradient of the performance measure $J(\theta)$ (under certain conditions) is:
> $$\nabla J(\theta) \propto \sum_{s} \mu(s) \sum_{a} q_\pi(s, a) \nabla \pi(a|s, \theta)$$
> 
> In practice, this leads to the stochastic gradient ascent update:
> $$\nabla J(\theta) = \mathbb{E}_\pi [G_t \nabla \ln \pi(A_t|S_t, \theta)]$$
> 
> where:
> - $\theta$ — policy parameters
> - $G_t$ — the return (reward to go)
> - $\nabla \ln \pi(A_t|S_t, \theta)$ — the "score function" or eligibility vector

## Advantages vs Value-Based Methods

1. **Continuous Action Spaces**: Can learn exact action probabilities or parameters of a distribution (e.g., mean/std of a Gaussian), whereas $\max Q$ is hard in continuous space.
2. **Stochastic Policies**: Can learn the optimal stochastic policy (e.g., in Rock-Paper-Scissors or Aliased Gridworlds), while value-based methods are typically deterministic.
3. **Convergence**: Often have stronger convergence guarantees as changes in parameters lead to smooth changes in the policy.

## Key Algorithms

- **REINFORCE**: The basic Monte Carlo PG algorithm using the full return $G_t$.
- **Actor-Critic**: Uses a "Critic" (value function) to estimate $q_\pi(s, a)$ instead of the full return to reduce variance.
- **PPO (Proximal Policy Optimization)**: A modern standard that uses a clipped objective to prevent destructively large updates.

## Connections

- Contrast with: [[Q-Learning]], [[SARSA]] (Value-based methods)
- Component of: [[Actor-Critic Methods]]
- Uses: [[Neural Networks]] (usually as the policy function)
- Strategy: Stochastic Gradient Ascent

## Appears In

- Future Week 5 Lecture
- [[RL-Book Ch13 - Policy Gradient Methods]]
