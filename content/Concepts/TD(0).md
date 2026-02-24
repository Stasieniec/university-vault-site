---
type: concept
aliases: [TD(0), TD zero]
course: [RL]
tags: [tabular-methods, key-formula]
status: complete
---

# TD(0)

The simplest [[Temporal Difference Learning]] method. One-step TD prediction.

> [!formula] TD(0) Update
> $$V(S_t) \leftarrow V(S_t) + \alpha \left[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\right]$$

- Uses a single step of experience: $(S_t, R_{t+1}, S_{t+1})$
- The [[TD Error]]: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$
- Bootstraps from $V(S_{t+1})$ — does not wait for episode end

See [[Temporal Difference Learning]] for full details and comparison with MC/DP.
