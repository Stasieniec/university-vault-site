---
type: concept
aliases: [deadly triad, the deadly triad]
course: [RL]
tags: [approximation, exam-topic, key-formula]
status: complete
---

# The Deadly Triad

## Definition

> [!definition] The Deadly Triad
> The **Deadly Triad** refers to the combination of three elements that, when present together, can cause RL algorithms to become **unstable and diverge**:
> 1. **[[Function Approximation]]** — parameterized value function (not tabular)
> 2. **[[Bootstrapping]]** — updating estimates based on other estimates (e.g., TD, not MC)
> 3. **Off-policy learning** — learning about a policy different from the one generating data

> [!warning] Any Two Are Fine
> Any **two** of these three elements is safe:
> - FA + Bootstrapping + **On-policy** → Semi-gradient TD converges (linear case)
> - FA + **No bootstrapping** + Off-policy → MC with importance sampling converges
> - **Tabular** + Bootstrapping + Off-policy → Q-learning converges
> 
> It's the combination of **all three** that can cause divergence.

## Why It Diverges

> [!intuition] The Feedback Loop from Hell
> 1. Off-policy data comes from a different distribution than what the target policy visits
> 2. Function approximation generalizes — updating one state changes values of others
> 3. Bootstrapping uses these (potentially wrong) estimates as targets
> 4. Result: errors can **amplify** rather than decrease. The updates push weights in a direction that increases error on states the target policy actually visits.

## Baird's Counterexample

The classic demonstration of deadly triad divergence:
- 7-state MDP with specific transitions
- Semi-gradient TD with linear FA
- Off-policy: behavior policy (uniform random) ≠ target policy
- **Result**: weights diverge to infinity, despite the problem being simple

## Solutions and Mitigations

| Approach | How it helps |
|----------|-------------|
| **Use on-policy data** | Removes off-policy element |
| **Don't bootstrap** (MC) | Removes bootstrapping element |
| **Use tabular** | Removes FA element (but defeats the purpose for large problems) |
| **[[Gradient-TD Methods]]** (GTD, GTD2, TDC) | True gradient methods that converge even off-policy with linear FA |
| **[[Experience Replay]]** + [[Target Network]] ([[Deep Q-Network (DQN)\|DQN]]) | Stabilizes training (doesn't guarantee convergence but works in practice) |
| **Emphatic-TD** | Reweights updates to correct for off-policy distribution |

## Connections

- Components: [[Function Approximation]], [[Bootstrapping]], [[On-Policy vs Off-Policy]]
- Demonstrated by: Baird's counterexample, Tsitsiklis & Van Roy examples
- Solutions: [[Gradient-TD Methods]], [[LSTD]]
- Practical workaround: [[Deep Q-Network (DQN)]] (experience replay + target networks)

## Appears In

- [[RL-L07 - Off-Policy RL with Approximation]]
- [[RL-Book Ch11 - Off-Policy Methods with Approximation]]
- [[RL-ES03 - Exercise Set Week 3]]
