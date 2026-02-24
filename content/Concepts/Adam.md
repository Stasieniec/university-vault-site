---
type: concept
aliases: [Adam Optimizer, Adaptive Moment Estimation]
course: [RL, IR]
tags: [optimization]
status: complete
---

# Adam

## Definition

> [!definition] Adam (Adaptive Moment Estimation)
> **Adam** is an optimization algorithm for gradient-based optimization of stochastic objective functions. It combines the advantages of **Momentum** (keeping track of the moving average of gradients) and **RMSProp** (scaling gradients by a moving average of squared gradients).

## The Update Rule

Adam maintains two moving averages (moments):
1. **First Moment ($m_t$)**: Mean of gradients (Momentum)
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
2. **Second Moment ($v_t$)**: Uncentered variance of gradients (RMSProp)
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

After **bias correction** ($\hat{m}_t = \frac{m_t}{1-\beta_1^t}$ and $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$), the weights are updated:

> [!formula] Adam Update
> $$w \leftarrow w - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
> 
> where:
> - $\alpha$ — learning rate (step size)
> - $\beta_1, \beta_2$ — decay rates for moment estimates (typically 0.9 and 0.999)
> - $\epsilon$ — small constant to prevent division by zero (e.g., $10^{-8}$)

## Key Advantages

- **Individual Learning Rates**: Each parameter gets its own adaptive learning rate.
- **Robustness**: Handles noisy gradients and non-stationary objectives well.
- **Efficiency**: Computationally efficient and requires little memory.
- **Default Choice**: Currently the most popular optimizer in Deep Learning.

## Connections

- Combines: [[Momentum]] and [[RMSProp]]
- Alternative to: [[SGD]], [[Adagrad]]
- Used for: Training [[Neural Networks]]

## Appears In

- Deep Learning Foundations
- RL and IR optimization sections
