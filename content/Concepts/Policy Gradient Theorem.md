---
type: concept
aliases: [PGT, Policy Gradient]
course: [RL]
tags: [policy-gradient, key-formula, exam-topic]
status: complete
---

# Policy Gradient Theorem

## Definition

> [!definition] Policy Gradient Theorem
> The **Policy Gradient Theorem** provides an analytic expression for the gradient of the performance objective $J(\theta)$ with respect to the policy parameters $\theta$. It allows the agent to update its policy directly without necessarily needing a value function.

## Mathematical Formulation

The theorem states that for any differentiable policy $\pi_\theta(a|s)$, the gradient of the performance (e.g., the average reward per step or the value of the start state) is:

> [!formula] Policy Gradient Theorem
> $$\nabla J(\theta) \propto \sum_s \mu(s) \sum_a q_\pi(s, a) \nabla \pi_\theta(a|s, \theta)$$
> 
> In expected value form (commonly used for stochastic gradient ascent):
> $$\nabla J(\theta) = \mathbb{E}_\pi [\nabla \log \pi_\theta(a|s, \theta) q_\pi(s, a)]$$
> 
> where:
> - $J(\theta)$ — Performance measure (e.g., $V^{\pi_\theta}(s_0)$)
> - $\mu(s)$ — On-policy state distribution under $\pi_\theta$
> - $q_\pi(s, a)$ — True action-value function under policy $\pi_\theta$
> - $\nabla \log \pi_\theta(a|s, \theta)$ — **Score function**

## Intuition

> [!intuition] The Likelihood Ratio Trick
> The theorem is powerful because the gradient $\nabla J(\theta)$ does **not** depend on the gradient of the state distribution $\mu(s)$, which is typically unknown and depends on the environment's dynamics. Instead, it only depends on the gradient of the policy and the value of action $a$ in state $s$. We increase the probability of actions that lead to high reward ($q_\pi(s,a)$ is positive/large) and decrease it otherwise.

## Key Properties

- **Foundation**: It is the theoretical basis for all policy gradient algorithms.
- **Objective**: Direct optimization of the policy $\pi$.
- **Action Selection**: Naturally handles continuous action spaces (unlike max-based methods like [[Q-Learning]]).

## Connections

- Derived for: [[REINFORCE]] (uses sample returns $G_t$ to estimate $q_\pi$)
- Extended in: [[Actor-Critic]] (uses a learned Critic to estimate $q_\pi$)
- Relies on: [[State Space]] and [[Reward Signal]] definitions

## Appears In

- future Week 5 lecture
