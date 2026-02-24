---
type: concept
aliases: [bootstrapping, bootstrap]
course: [RL]
tags: [foundations]
status: complete
---

# Bootstrapping

> [!definition] Bootstrapping
> In RL, **bootstrapping** means updating an estimate based partly on other estimates (rather than exclusively on actual observed values). The update target includes a current estimate of a value function.

## Examples

- **[[Dynamic Programming]]**: $V(s) \leftarrow \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')]$ — uses $V(s')$, which is itself an estimate
- **[[Temporal Difference Learning|TD(0)]]**: $V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$ — uses $V(S_{t+1})$
- **[[Monte Carlo Methods]]**: $V(S_t) \leftarrow V(S_t) + \alpha[G_t - V(S_t)]$ — uses actual return $G_t$, **NOT bootstrapping**

> [!intuition] Trade-Off
> Bootstrapping introduces **bias** (estimates are wrong initially) but reduces **variance** (don't need to wait for the full noisy return). MC has zero bias but high variance. TD has some bias but much lower variance.

## Role in the [[Deadly Triad]]

Bootstrapping is one of the three elements. Combined with [[Function Approximation]] and off-policy learning, it can cause divergence.

## Appears In

- [[RL-L04 - Temporal Difference Learning]]
- [[RL-L05 - Tabular to Approximation]]
