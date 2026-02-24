---
type: concept
aliases: [return, Gt, cumulative reward, discounted return, Returns]
course: [RL]
tags: [foundations, key-formula]
status: complete
---

# Return

## Definition

> [!definition] Return
> The **return** $G_t$ is the total accumulated reward from time step $t$ onward. It is the quantity that RL agents seek to maximize (in expectation).

> [!formula] Return (Discounted)
> $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$
> 
> where:
> - $R_{t+k+1}$ — reward received $k+1$ steps after time $t$
> - $\gamma \in [0, 1]$ — [[Discount Factor]]

## Variants

**Episodic (undiscounted or discounted):**
$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$

where $T$ is the terminal time step. With $\gamma = 1$, this is just the sum of all remaining rewards.

**Continuing (must have $\gamma < 1$):**
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

Converges as long as $\gamma < 1$ and rewards are bounded.

## Recursive Property

> [!formula] Recursive Return
> $$G_t = R_{t+1} + \gamma G_{t+1}$$
> 
> This is the key recursive relationship that enables [[Bootstrapping]] and the [[Bellman Equation]].

> [!intuition] Why This Matters
> You don't need to compute the entire sum from scratch. The return at time $t$ equals the immediate reward plus the discounted return from the next step. This decomposition is the foundation of [[Dynamic Programming]] and [[Temporal Difference Learning]].

## Role in RL Methods

- **[[Monte Carlo Methods]]**: Estimate $v_\pi(s)$ by averaging **actual returns** $G_t$ observed after visiting state $s$
- **[[Temporal Difference Learning]]**: Approximates $G_t$ with $R_{t+1} + \gamma V(S_{t+1})$ (one-step bootstrap)
- **[[Value Function]]**: Defined as the expected return: $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$

## Connections

- Used to define: [[Value Function]]
- Discounted by: [[Discount Factor]]
- Estimated by: [[Monte Carlo Methods]] (full), [[Temporal Difference Learning]] (bootstrapped)
- Recursive structure enables: [[Bellman Equation]]

## Appears In

- [[RL-L01 - Intro, MDPs & Bandits]], [[RL-L03 - Monte Carlo Methods]], [[RL-L04 - Temporal Difference Learning]]
