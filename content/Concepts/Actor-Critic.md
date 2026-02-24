---
type: concept
aliases: [Actor-Critic architecture, AC]
course: [RL]
tags: [policy-gradient, actor-critic, exam-topic]
status: complete
---

# Actor-Critic

## Definition

> [!definition] Actor-Critic Methods
> **Actor-Critic** methods are a class of Reinforcement Learning algorithms that combine both **policy-based** and **value-based** approaches. They consist of two components:
> 1. **Actor**: Responsible for selecting actions by maintaining a parameterized policy $\pi_\theta(a|s)$.
> 2. **Critic**: Responsible for evaluating the actions taken by the actor by estimating a value function (usually the state-value function $V_w(s)$ or action-value function $Q_w(s,a)$).

## Intuition

> [!intuition] Evaluation Reduces Variance
> In pure policy gradient methods like [[REINFORCE]], the update uses the full episode return $G_t$, which has high variance because it depends on many random actions and transitions. Actor-Critic methods replace the return with a value estimate from the **Critic**. This "bootstrapping" significantly reduces variance, leading to faster and more stable learning, though at the cost of introducing some bias from the value estimate.

## Mathematical Formulation

The Actor update typically follows the gradient of the performance objective, often using the **Advantage function** to indicate how much better an action was than average:

> [!formula] Actor-Critic Update (Advantage Actor-Critic)
> **Actor Update:** 
> $$\theta_{t+1} \leftarrow \theta_t + \alpha_\theta \nabla \log \pi_\theta(A_t|S_t) \hat{A}(S_t, A_t)$$
> 
> **Critic Update (TD Error):**
> $$\delta_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)$$
> $$w_{t+1} \leftarrow w_t + \alpha_w \delta_t \nabla V_w(S_t)$$
> 
> where:
> - $\hat{A}(S_t, A_t) \approx \delta_t$ is the **Advantage** (estimated by the TD error)
> - $\theta$ — Actor parameters
> - $w$ — Critic parameters

## Variants and Key Properties

- **A2C (Advantage Actor-Critic)**: Synchronous version where multiple workers update a global model.
- **A3C (Asynchronous Advantage Actor-Critic)**: Asynchronous version where workers update global parameters independently.
- **Relationship to Policy Gradient**: It is a specialized form of the [[Policy Gradient Theorem]] where the Critic provides the $Q$-value or Advantage estimate.

## Connections

- Combines: [[Policy Gradient Theorem]] and [[Temporal Difference Learning]]
- Improves upon: [[REINFORCE]] (by reduces variance through bootstrapping)
- Foundation for: [[Soft Actor-Critic (SAC)]], PPO, DDPG

## Appears In

- [[RL-L08 - Policy Gradient and Actor-Critic]] (mentioned)
- future weeks
