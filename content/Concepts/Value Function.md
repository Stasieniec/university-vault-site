---
type: concept
aliases: [value function, state-value function, action-value function, V(s), Q(s a)]
course: [RL]
tags: [foundations, key-formula, exam-topic]
status: complete
---

# Value Function

## Definition

> [!definition] Value Function
> A **value function** estimates "how good" it is for an agent to be in a given state (or to take a given action in a state). "How good" is defined in terms of expected future rewards — specifically, the expected [[Return]].

There are two types:

### State-Value Function $v_\pi(s)$

> [!formula] State-Value Function
> $$v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s\right]$$
> 
> The expected [[Return]] when starting in state $s$ and following [[Policy]] $\pi$ thereafter.

### Action-Value Function $q_\pi(s, a)$

> [!formula] Action-Value Function
> $$q_\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s, A_t = a\right]$$
> 
> The expected [[Return]] when starting in state $s$, taking action $a$, and following $\pi$ thereafter.

### Relationship

$$v_\pi(s) = \sum_{a \in \mathcal{A}(s)} \pi(a|s) \, q_\pi(s, a)$$

> [!intuition] $V$ vs $Q$
> - $v_\pi(s)$: "How good is this state?" (averaged over what my policy would do)
> - $q_\pi(s, a)$: "How good is taking this specific action in this state?"
> 
> **For control (finding the best policy), we usually need $Q$**, because choosing $\arg\max_a v_\pi(s')$ requires knowing the model $p(s'|s,a)$, while $\arg\max_a q_\pi(s,a)$ doesn't.

## Optimal Value Functions

> [!formula] Optimal State-Value Function
> $$v_*(s) = \max_\pi v_\pi(s) = \max_a q_*(s, a)$$
> 
> The best possible value of state $s$ under any policy.

> [!formula] Optimal Action-Value Function
> $$q_*(s, a) = \max_\pi q_\pi(s, a) = \mathbb{E}[R_{t+1} + \gamma \, v_*(S_{t+1}) \mid S_t = s, A_t = a]$$
> 
> The best possible value of taking action $a$ in state $s$.

If we know $q_*$, the optimal [[Policy]] is trivially:
$$\pi_*(s) = \arg\max_a q_*(s, a)$$

## Estimation Methods

| Method | How it estimates $V$ or $Q$ |
|--------|---------------------------|
| [[Dynamic Programming]] | Solves [[Bellman Equation]] exactly (requires model) |
| [[Monte Carlo Methods]] | Averages sampled returns $G_t$ |
| [[Temporal Difference Learning]] | Bootstraps: $V(s) \leftarrow V(s) + \alpha[R + \gamma V(s') - V(s)]$ |
| [[Function Approximation]] | Parameterized $\hat{v}(s, \mathbf{w})$ trained with SGD |

## Key Properties

- Value functions satisfy the [[Bellman Equation]] (recursive relationship)
- There exists a **partial ordering** over policies defined by value functions: $\pi \geq \pi'$ iff $v_\pi(s) \geq v_{\pi'}(s)$ for all $s$
- At least one policy is better than or equal to all others — the **optimal policy** $\pi_*$
- All optimal policies share the same $v_*$ and $q_*$

> [!warning] Tabular vs Approximate
> In tabular settings, $V(s)$ is stored as a table with one entry per state. With [[Function Approximation]], we use $\hat{v}(s, \mathbf{w})$ — a parameterized function. The fundamental concept is the same, but convergence guarantees differ.

## Connections

- Defined on: [[Markov Decision Process]]
- Recursive structure: [[Bellman Equation]]
- Estimated by: [[Monte Carlo Methods]], [[Temporal Difference Learning]], [[Dynamic Programming]]
- Approximated by: [[Function Approximation]], [[Linear Function Approximation]], [[Deep Q-Network (DQN)]]

## Appears In

- [[RL-L01 - Intro, MDPs & Bandits]] (definition, intuition)
- [[RL-L02 - Dynamic Programming]] (computation)
- [[RL-L05 - Tabular to Approximation]] (approximation)
- All exercise sets
