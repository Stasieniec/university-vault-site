---
type: concept
aliases: [discount factor, gamma, γ, discounting]
course: [RL]
tags: [foundations]
status: complete
---

# Discount Factor ($\gamma$)

## Definition

> [!definition] Discount Factor
> The **discount factor** $\gamma \in [0, 1]$ determines the present value of future rewards. A reward received $k$ time steps in the future is worth $\gamma^k$ times what it would be worth if received immediately.

> [!intuition] What $\gamma$ Controls
> - $\gamma = 0$: **Myopic** — only cares about immediate reward $R_{t+1}$
> - $\gamma = 1$: **Far-sighted** — future rewards are just as important as immediate ones (only valid for episodic tasks)
> - $\gamma \approx 0.9 \text{–} 0.99$: Typical values — balances short and long-term

## Why Discount?

1. **Mathematical convenience**: Ensures the [[Return]] is finite for continuing (infinite-horizon) tasks when rewards are bounded: $G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1} \leq \frac{R_{\max}}{1 - \gamma}$
2. **Uncertainty**: Future is less predictable — discounting is a form of "doubt" about far-future rewards
3. **Preference for sooner rewards**: Like interest rates in economics

> [!warning] $\gamma = 1$ in Continuing Tasks
> If $\gamma = 1$ and the task never terminates, the return can diverge to infinity. Only use $\gamma = 1$ for episodic tasks.

## Connections

- Scales: [[Return]]
- Appears in: [[Bellman Equation]], [[Value Function]]
- Affects convergence of: [[Dynamic Programming]], [[Temporal Difference Learning]]

## Appears In

- [[RL-L01 - Intro, MDPs & Bandits]]
- [[RL-Book Ch3 - Finite MDPs]]
