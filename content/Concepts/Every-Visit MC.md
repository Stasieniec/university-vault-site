---
type: concept
aliases: [every-visit MC, every-visit Monte Carlo]
course: [RL]
tags: [tabular-methods]
status: complete
---

# Every-Visit MC

> [!definition] Every-Visit MC
> A [[Monte Carlo Methods|Monte Carlo]] prediction method that averages returns from **every** visit to a state within each episode. Returns within an episode are correlated, but the estimator is still consistent (converges asymptotically).

Compared to [[First-Visit MC]]: slightly biased for finite samples but also converges. In practice, often comparable performance.
