---
type: concept
aliases: [MDP, Markov decision process]
course: [RL]
tags: [foundations, exam-topic]
status: complete
---

# Markov Decision Process (MDP)

## Definition

> [!definition] Markov Decision Process
> A **Markov Decision Process** is a mathematical framework for modeling sequential decision-making problems where outcomes are partly random and partly under the control of a decision-maker (agent). An MDP is defined by the tuple $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$.

**Components:**
- $\mathcal{S}$ — **State space**: the set of all possible states
- $\mathcal{A}$ (or $\mathcal{A}(s)$) — **Action space**: the set of all possible actions (may depend on state)
- $p(s', r \mid s, a)$ — **Dynamics function**: probability of transitioning to state $s'$ with reward $r$, given state $s$ and action $a$
- $r(s, a)$ or $R_t$ — **Reward signal**: immediate numerical feedback
- $\gamma \in [0, 1]$ — **[[Discount Factor]]**: controls importance of future rewards

## The Markov Property

> [!formula] Markov Property
> $$P(S_{t+1} = s', R_{t+1} = r \mid S_t, A_t, S_{t-1}, A_{t-1}, \ldots, S_0, A_0) = P(S_{t+1} = s', R_{t+1} = r \mid S_t, A_t)$$
> 
> The future is conditionally independent of the past given the present state. The state captures all relevant information from the history.

> [!intuition] Why Markov Matters
> "The state is a sufficient statistic of history." If you know the current state, knowing *how* you got there doesn't give you any additional information about what will happen next. This is what makes MDPs computationally tractable — you don't need to store the entire history.

## Dynamics Function

The dynamics function $p$ completely characterizes the environment:

$$p(s', r \mid s, a) = \Pr\{S_{t+1} = s', R_{t+1} = r \mid S_t = s, A_t = a\}$$

From $p$, we can derive everything else:

> [!formula] Derived Quantities from Dynamics
> **State-transition probabilities:**
> $$p(s' \mid s, a) = \sum_{r \in \mathcal{R}} p(s', r \mid s, a)$$
> 
> **Expected reward for state-action pair:**
> $$r(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a] = \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r \mid s, a)$$
> 
> **Expected reward for state-action-next-state triple:**
> $$r(s, a, s') = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s'] = \frac{\sum_{r \in \mathcal{R}} r \cdot p(s', r \mid s, a)}{p(s' \mid s, a)}$$

## Agent–Environment Interface

```
        ┌─────────┐
   Aₜ   │         │  Sₜ₊₁, Rₜ₊₁
───────► │  Env    │─────────────►
         │         │              │
         └─────────┘              │
              ▲                   │
              │                   ▼
         ┌─────────┐
         │  Agent  │
         │ (Policy)│
         └─────────┘
```

At each time step $t$:
1. Agent observes state $S_t \in \mathcal{S}$
2. Agent selects action $A_t \in \mathcal{A}(S_t)$ according to its [[Policy]] $\pi$
3. Environment transitions to $S_{t+1}$ and emits reward $R_{t+1}$ according to $p(s', r \mid s, a)$
4. Repeat

## Episodic vs Continuing Tasks

- **Episodic tasks**: Interaction naturally breaks into episodes with a terminal state (e.g., games, maze navigation). The [[Return]] is a finite sum.
- **Continuing tasks**: Interaction goes on forever without natural termination (e.g., process control). Requires $\gamma < 1$ for the return to be finite.

## Key Properties

- MDPs provide the theoretical foundation for all of RL
- [[Dynamic Programming]] methods require knowing $p(s', r \mid s, a)$ explicitly (model-based)
- [[Monte Carlo Methods]] and [[Temporal Difference Learning]] learn without knowing $p$ (model-free)
- The optimal solution is found via the [[Bellman Optimality Equation]]

## Connections

- Generalizes: [[Multi-Armed Bandit]] (bandit = 1-state MDP)
- Foundation for: [[Value Function]], [[Bellman Equation]], [[Policy]], [[Dynamic Programming]]
- Extended by: POMDP (partial observability), Semi-MDPs, Factored MDPs

## Appears In

- [[RL-L01 - Intro, MDPs & Bandits]]
- [[RL-Book Ch3 - Finite MDPs]]
- [[RL-ES01 - Exercise Set Week 1]]
