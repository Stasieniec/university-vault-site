---
type: concept
aliases: [policy, policies, π]
course: [RL]
tags: [foundations, exam-topic]
status: complete
---

# Policy

## Definition

> [!definition] Policy
> A **policy** $\pi$ is a mapping from states to probabilities of selecting each possible action. If an agent follows policy $\pi$ at time $t$, then $\pi(a|s)$ is the probability that $A_t = a$ given $S_t = s$.

$$\pi(a|s) = P(A_t = a \mid S_t = s)$$

- **Deterministic policy**: $\pi(s) = a$ — maps each state to exactly one action
- **Stochastic policy**: $\pi(a|s) \in [0, 1]$ — a probability distribution over actions for each state, where $\sum_a \pi(a|s) = 1$

## Types of Policies

### Greedy Policy
$$\pi(s) = \arg\max_a q(s, a)$$
Always picks the action with the highest estimated value. Pure exploitation, no exploration.

### ε-Greedy Policy ([[Epsilon-Greedy Policy]])
$$\pi(a|s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}(s)|} & \text{if } a = \arg\max_{a'} q(s, a') \\ \frac{\varepsilon}{|\mathcal{A}(s)|} & \text{otherwise} \end{cases}$$
Mostly greedy, but with probability $\varepsilon$ picks a random action. Balances [[Exploration vs Exploitation]].

### Softmax / Boltzmann Policy
$$\pi(a|s) = \frac{e^{q(s,a)/\tau}}{\sum_{a'} e^{q(s,a')/\tau}}$$
Temperature $\tau$ controls exploration: high $\tau$ → uniform, low $\tau$ → greedy.

## Optimal Policy

> [!definition] Optimal Policy
> A policy $\pi_*$ is **optimal** if $v_{\pi_*}(s) \geq v_\pi(s)$ for all $s \in \mathcal{S}$ and all policies $\pi$. There always exists at least one optimal policy for any finite MDP.

All optimal policies share the same [[Value Function|optimal value functions]] $v_*$ and $q_*$. Given $q_*$:
$$\pi_*(s) = \arg\max_a q_*(s, a)$$

## Policy in Different RL Methods

| Method | How policy is used |
|--------|-------------------|
| [[Policy Iteration]] | Explicit policy, alternates evaluation and improvement |
| [[Value Iteration]] | Implicit policy (greedy w.r.t. current $V$) |
| [[Monte Carlo Methods]] | Generates episodes, improved via ε-greedy |
| [[SARSA]] | On-policy: follows and improves ε-greedy |
| [[Q-Learning]] | Off-policy: follows ε-greedy, learns about greedy |
| [[REINFORCE]] | Directly parameterized: $\pi_\theta(a\|s)$ |

## On-Policy vs Off-Policy

> [!tip] Key Distinction
> - **Behavior policy** $b$: the policy used to generate data (select actions)
> - **Target policy** $\pi$: the policy being evaluated or improved
> - **On-policy**: $b = \pi$ (same policy)
> - **Off-policy**: $b \neq \pi$ (different policies, requires [[Importance Sampling]] correction)

See [[On-Policy vs Off-Policy]] for details.

## Connections

- Acts within: [[Markov Decision Process]]
- Evaluated by: [[Value Function]], [[Bellman Equation]]
- Improved by: [[Policy Iteration]], [[Generalized Policy Iteration]]
- Parameterized: [[REINFORCE]], [[Policy Gradient Theorem]]

## Appears In

- [[RL-L01 - Intro, MDPs & Bandits]], [[RL-L02 - Dynamic Programming]], [[RL-L03 - Monte Carlo Methods]], [[RL-L04 - Temporal Difference Learning]]
