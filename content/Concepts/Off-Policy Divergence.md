---
type: concept
aliases: [off-policy divergence, Baird's counterexample]
course: [RL]
tags: [approximation, exam-topic]
status: complete
---

# Off-Policy Divergence

> [!definition] Off-Policy Divergence
> The phenomenon where [[Semi-Gradient Methods]] with [[Function Approximation]] and off-policy data cause the weights $\mathbf{w}$ to grow unbounded, despite the problem being simple and well-defined.

## Baird's Counterexample

The classic demonstration: a 7-state MDP where semi-gradient TD with linear FA diverges under off-policy updates. All three elements of the [[Deadly Triad]] are present:
1. Linear function approximation ✓
2. Bootstrapping (TD) ✓  
3. Off-policy (behavior ≠ target policy) ✓

Even with small step sizes and favorable conditions, the weights diverge. This motivated [[Gradient-TD Methods]] as a theoretically sound alternative.

## Appears In

- [[RL-L07 - Off-Policy RL with Approximation]]
- [[RL-Book Ch11 - Off-Policy Methods with Approximation]] (§11.2)
