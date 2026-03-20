---
type: concept
aliases: [Maximum Entropy Reinforcement Learning, MaxEnt RL, entropy-regularized RL]
course: [RL]
tags: [deep-rl, policy-gradient]
status: complete
---

# Maximum Entropy RL

> [!definition] Maximum Entropy RL
> A framework that augments the standard [[Reinforcement Learning|RL]] objective with an **entropy bonus**, encouraging the agent to maximize expected return while acting as randomly as possible. The agent simultaneously maximizes reward and policy entropy.

## Objective

> [!formula] Maximum Entropy Objective
> $$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$
>
> where:
> - $\mathcal{H}(\pi(\cdot|s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$ — entropy of the policy at state $s$
> - $\alpha > 0$ — **temperature parameter** controlling the exploration–exploitation tradeoff
> - $\rho_\pi$ — state-action distribution induced by $\pi$

## Intuition

> [!intuition] Why Add Entropy?
> Standard RL finds a single optimal action per state. Maximum Entropy RL says: "Among all policies that achieve high reward, prefer the one that is **most random**." This has several benefits:
> - **Better exploration**: the agent is incentivized to try diverse actions
> - **Robustness**: the policy doesn't collapse to a single brittle action
> - **Multi-modality**: can capture multiple near-optimal strategies
> - **Composability**: entropy-regularized policies combine well across tasks

## Effect of Temperature $\alpha$

| $\alpha$ | Behavior |
|-----------|----------|
| $\alpha \to 0$ | Standard (greedy) RL — exploit only |
| $\alpha$ small | Slight exploration bonus |
| $\alpha$ large | Highly stochastic — explore aggressively |
| $\alpha \to \infty$ | Uniform random policy |

## Soft Bellman Equation

The entropy bonus modifies the Bellman equations:

> [!formula] Soft Value Functions
> $$V_{\text{soft}}(s) = \mathbb{E}_{a \sim \pi}\left[Q_{\text{soft}}(s, a) - \alpha \log \pi(a|s)\right]$$
>
> $$Q_{\text{soft}}(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim p}\left[V_{\text{soft}}(s')\right]$$

## Key Properties

- Provides a principled way to trade off exploration and exploitation via $\alpha$
- Leads to stochastic optimal policies (unlike standard RL which yields deterministic ones)
- Foundation for [[Soft Actor-Critic (SAC)]]
- Can be interpreted as KL-regularized RL (keeping policy close to a uniform prior)

## Connections

- Implemented by [[Soft Actor-Critic (SAC)]]
- Related to [[Exploration vs Exploitation]] — entropy provides intrinsic exploration
- Builds on [[Policy Gradient Methods]] and [[Actor-Critic]]
- Temperature $\alpha$ plays a similar role to exploration parameters in [[Epsilon-Greedy Policy]]

## Appears In

- [[RL-L11 - SAC, Decision Transformer & Diffuser]]
