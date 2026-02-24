---
type: concept
aliases: [TD error, temporal difference error, δ]
course: [RL]
tags: [tabular-methods, key-formula]
status: complete
---

# TD Error

> [!formula] TD Error
> $$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
> 
> where:
> - $R_{t+1} + \gamma V(S_{t+1})$ — the **TD target** (better estimate of $V(S_t)$)
> - $V(S_t)$ — current estimate
> - $\delta_t$ — the "surprise": how much better (or worse) the next step was than expected

> [!intuition] What It Measures
> The TD error is the difference between a **new estimate** of the value (based on what just happened) and the **old estimate**. If $\delta_t > 0$, things went better than expected; if $\delta_t < 0$, worse. The update $V(S_t) \leftarrow V(S_t) + \alpha \delta_t$ nudges the estimate toward the new evidence.

The TD error is the fundamental signal for all [[Temporal Difference Learning]] methods. It also appears in:
- [[SARSA]]: $\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$
- [[Q-Learning]]: $\delta_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)$
- [[Semi-Gradient Methods]]: Same form, drives weight updates

## Appears In

- [[RL-L04 - Temporal Difference Learning]], [[RL-L05 - Tabular to Approximation]]
