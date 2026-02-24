---
type: concept
aliases: [first-visit MC, first-visit Monte Carlo]
course: [RL]
tags: [tabular-methods]
status: complete
---

# First-Visit MC

> [!definition] First-Visit MC
> A [[Monte Carlo Methods|Monte Carlo]] prediction method that averages returns only from the **first time** a state is visited in each episode. Each return is an independent, identically distributed estimate of $v_\pi(s)$.

Converges to $v_\pi(s)$ by the Law of Large Numbers. Contrast with [[Every-Visit MC]], which uses all visits.

See [[RL-L03 - Monte Carlo Methods]] for the full algorithm.
