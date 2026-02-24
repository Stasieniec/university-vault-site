---
type: concept
aliases: [Root Mean Square Propagation]
course: [RL, IR]
tags: [optimization]
status: complete
---

# RMSProp

## Definition

> [!definition] RMSProp
> **RMSProp** (Root Mean Square Propagation) is an adaptive learning rate optimization method designed to tackle the radically diminishing learning rates of AdaGrad. It limits the influence of historical gradients by using an **exponentially decaying average** of squared gradients.

## Intuition

> [!intuition] Normalizing the Gradient
> If a gradient is consistently large, we want to slow down in that direction to avoid overshooting. If a gradient is small, we want to speed up. RMSProp does this by dividing the current gradient by a "running average" of recent gradient magnitudes. This keeps the updates at a similar scale across all dimensions.

## Mathematical Formulation

For each parameter $w$:

1. **Accumulate Squared Gradient**:
   $$v_t = \rho v_{t-1} + (1 - \rho) g_t^2$$
2. **Update Weights**:
   $$w \leftarrow w - \frac{\alpha}{\sqrt{v_t + \epsilon}} g_t$$

where:
- $\alpha$ — learning rate
- $\rho$ — forgetting factor (decay rate, typically 0.9)
- $g_t$ — current gradient $\frac{\partial L}{\partial w}$
- $\epsilon$ — small constant for stability

## RMSProp vs AdaGrad

- **AdaGrad**: Accumulates *all* past squared gradients. This causes the learning rate to eventually shrink to zero, stopping learning prematurely.
- **RMSProp**: Only "remembers" recent gradients via the decay factor $\rho$. This allows the optimizer to continue learning indefinitely in non-stationary environments.

## Connections

- Sub-component of: [[Adam]]
- Improved version of: [[Adagrad]]
- Context: Optimization for [[Neural Networks]]

## Appears In

- Deep Learning Optimization
- [[RL-L08 - Deep Reinforcement Learning]] (often used in A3C)
