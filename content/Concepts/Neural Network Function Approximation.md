---
type: concept
aliases: [neural network FA, nonlinear function approximation, deep function approximation]
course: [RL]
tags: [approximation, deep-rl]
status: complete
---

# Neural Network Function Approximation

> [!definition] Neural Network Function Approximation
> Using neural networks as non-linear function approximators for value functions or policies: $\hat{v}(s, \boldsymbol{\theta}) = f_{\boldsymbol{\theta}}(s)$ where $f$ is a neural network with parameters $\boldsymbol{\theta}$.

## Advantages over Linear FA

- **Automatic feature learning**: No manual [[Feature Construction]] needed
- **Representational power**: Can approximate any continuous function (universal approximation theorem)
- **Handles raw inputs**: Can process pixels, text, etc. directly

## Challenges in RL

- **No convergence guarantees** for [[Semi-Gradient Methods]] with non-linear FA
- **[[Deadly Triad]]** becomes more dangerous — non-linear + bootstrapping + off-policy
- **Non-stationarity**: Target values change as policy improves
- **Catastrophic forgetting**: Updating for new states can degrade performance on old states

Stabilization techniques: [[Experience Replay]], [[Target Network]] (as in [[Deep Q-Network (DQN)]])

## Appears In

- [[RL-L06 - On-Policy TD with Approximation]] (§9.7), [[RL-L08 - Deep RL Value-Based]]
