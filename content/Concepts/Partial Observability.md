---
type: concept
aliases: [partial observability, partially observable]
course: [RL]
tags: [foundations, exam-topic]
status: complete
---

# Partial Observability

> [!definition] Partial Observability
> A setting where the agent cannot directly observe the true state $x_t$ of the environment. Instead, it receives **observations** $o_t$ that provide incomplete or noisy information about the underlying state. The standard [[Markov Decision Process|MDP]] assumption of full state access is violated.

## Intuition

> [!intuition] Seeing Through a Keyhole
> Full observability is like playing chess — you can see the entire board. Partial observability is like playing poker — you can only see your own cards. Your decisions must account for uncertainty about what you can't see.

## Handling Partial Observability

Three approaches (from least to most approximate):

| Approach | Idea | Requirements | Limitations |
|----------|------|--------------|-------------|
| **[[Belief State]]** | Probability distribution over hidden states | Full model (transitions, observations) | Discrete states only, model needed |
| **[[Predictive State Representation]]** | Predictions about future observations | Core tests | Tabular setting |
| **Approximate** | Use recent observations as state | Nothing extra | Not optimal, no guarantees |

### Approximate Methods in Practice

1. **Single observation**: $S_t = O_t$ — simplest, often "good enough"
2. **Frame stacking**: $S_t = (O_{t-k}, A_{t-k}, \ldots, O_t)$ — used in Atari DQN (4 frames)
3. **Recurrent networks**: [[Deep Recurrent Q-Learning]] — LSTM maintains internal memory

> [!tip] Practical Reality
> In practice, many successful RL systems simply treat observations as states ($S = O$). With function approximation, there's typically no guarantee that the features define a Markov state anyway. As long as the system is "close enough" to Markov, this can work well enough.

## Connections

- Formalized by [[POMDP]]
- Generalizes [[Markov Decision Process]] (MDP is a special case with $O = X$)
- [[Belief State]] and [[Predictive State Representation]] are exact approaches
- [[Deep Recurrent Q-Learning]] is an approximate deep learning approach

## Appears In

- [[RL-L13 - Partial Observability]]
- [[RL-Book Ch17 - Frontiers]] (§17.3)
