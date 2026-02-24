---
type: concept
aliases: [MSVE, VE, Value Error, Mean Squared Value Error (MSVE)]
course: [RL]
tags: [approximation, key-formula, exam-topic]
status: complete
---

# Mean Squared Value Error

## Definition

> [!definition] Mean Squared Value Error (MSVE)
> In RL with **function approximation**, we cannot represent the true value function $v_\pi$ exactly for all states. The **Mean Squared Value Error (MSVE)** is the standard objective function used to measure how well our approximate value function $\hat{v}(s, \mathbf{w})$ matches the true value function $v_\pi(s)$.

## Mathematical Formulation

> [!formula] MSVE
> $$\overline{VE}(\mathbf{w}) = \sum_{s \in \mathcal{S}} \mu(s) \left[ v_\pi(s) - \hat{v}(s, \mathbf{w}) \right]^2$$
> 
> where:
> - $\mathbf{w}$ — Learnable weights of the function approximator
> - $v_\pi(s)$ — True value of state $s$ under policy $\pi$
> - $\hat{v}(s, \mathbf{w})$ — Approximate value (e.g., from a neural network or linear combination)
> - $\mu(s)$ — **State distribution**, usually the on-policy distribution (stationary distribution under $\pi$). It weights the error by how often the agent actually visits state $s$.

## Why We Need This

> [!intuition] Trade-offs in Approximation
> With function approximation, we have fewer parameters than states ($d \ll |\mathcal{S}|$). This means improving the accuracy in one state usually makes it worse in another. The MSVE tells us which states are more important to get "right" based on the distribution $\mu(s)$. We accept more error in rarely visited states to achieve lower error in frequently visited ones.

## Key Properties

- **Objective**: Algorithms like Gradient Descent minimize this error by calculating $\nabla \overline{VE}(\mathbf{w})$.
- **The Ideal Goal**: In the tabular case, $\overline{VE} = 0$. In approximation, we seek a global (or local) minimum.
- **Challenge**: In RL, we don't actually know the true $v_\pi(s)$ (the "target"). Algorithms like TD replace it with a bootstrapped estimate, which changes the optimization landscape.

## Connections

- Used for: [[Function Approximation]]
- Optimized via: Stochastic Gradient Descent (SGD)
- Relies on: [[State Space]] distribution $\mu$

## Appears In

- [[RL-L06 - Value Function Approximation]]
- [[RL-Book Ch9 - On-policy Prediction with Approximation]]
