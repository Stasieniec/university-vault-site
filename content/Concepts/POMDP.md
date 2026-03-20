---
type: concept
aliases: [POMDP, Partially Observable MDP, Partially Observable Markov Decision Process]
course: [RL]
tags: [foundations, exam-topic]
status: complete
---

# POMDP (Partially Observable Markov Decision Process)

> [!definition] POMDP
> A **Partially Observable Markov Decision Process** extends the [[Markov Decision Process|MDP]] framework to settings where the agent cannot directly observe the true state $x_t$ of the environment. Instead, the agent receives **observations** $o_t$ that provide partial or noisy information about the underlying state.

## Formal Definition

A POMDP is defined by the tuple $(X, A, O, T, R, Z, \gamma)$:

| Component | Description |
|-----------|-------------|
| $X$ | Set of (hidden/latent) states |
| $A$ | Set of actions |
| $O$ | Set of observations |
| $T(x'|x,a)$ | Transition function $p(x_{t+1}|x_t, a_t)$ |
| $R(x,a)$ | Reward function |
| $Z(o|x',a)$ | Observation function $p(o_{t+1}|x_{t+1}, a_t)$ |
| $\gamma$ | Discount factor |

## The Challenge

> [!intuition] Why Partial Observability Is Hard
> In an MDP, the current state $s_t$ tells you everything you need to make an optimal decision. In a POMDP, the observation $o_t$ doesn't — two different underlying states might produce the same observation. The agent must **reason about what state it might be in** based on its history of observations and actions.

- **History**: $h_t = (o_0, a_0, o_1, a_1, \ldots, o_t)$ contains all available information
- The history grows without bound — we need a compact sufficient statistic

## Approaches to Handle Partial Observability

### 1. [[Belief State]]
Maintain a probability distribution over hidden states: $b_t(x) = \Pr(X_t = x | H_t = h_t)$. The belief state MDP is fully observable.

### 2. [[Predictive State Representation]]
Define internal state as predictions about future observations rather than beliefs about hidden states.

### 3. Approximate Methods
Use recent observations as state (frame stacking) or recurrent networks ([[Deep Recurrent Q-Learning]]).

## Markov Criterion for Internal State

> [!formula] Markov Criterion
> An internal state representation $f(h)$ is Markov if:
> $$f(h) = f(h') \implies \Pr\{O_{t+1} = o | H_t = h, A_t = a\} = \Pr\{O_{t+1} = o | H_t = h', A_t = a\}$$
>
> That is, if two histories map to the same internal state, they must predict the same future observations.

## Key Properties

- POMDPs are strictly harder than MDPs (optimal POMDP policies may be stochastic even when MDP optimal policies are deterministic)
- The belief state MDP converts a POMDP into a (continuous-state) MDP
- In practice, many RL systems ignore partial observability and treat observations as states

## Connections

- Generalizes [[Markov Decision Process]] — MDP is a POMDP where observations = states
- Solved via [[Belief State]] (exact) or approximate methods
- [[Deep Recurrent Q-Learning]] — deep RL approach to POMDPs
- Related to [[Importance Sampling]] in the sense that both deal with incomplete information

## Appears In

- [[RL-L13 - Partial Observability]]
- [[RL-Book Ch17 - Frontiers]] (§17.3)
