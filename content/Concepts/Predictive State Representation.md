---
type: concept
aliases: [PSR, predictive state representation, predictive state]
course: [RL]
tags: [foundations]
status: complete
---

# Predictive State Representation

> [!definition] Predictive State Representation (PSR)
> An alternative to [[Belief State]]s for handling [[Partial Observability]] in [[POMDP]]s. Instead of maintaining a probability distribution over hidden states, a PSR defines the internal state as a vector of **predictions about future observations** (test probabilities).

## Core Idea

> [!intuition] Predict the Future, Not the Hidden Present
> Belief states answer: "What hidden state am I likely in?" PSRs answer: "What would I observe if I did certain things?" Both are valid Markov representations, but PSRs don't require knowledge of the hidden state space or transition/observation models.

## Definition

Define a "test" $\tau = a_1 o_1 a_2 o_2 \ldots a_k o_k$ as a sequence of actions and observations. The test probability is:

> [!formula] Test Probability
> $$p(\tau | h) = \Pr\{O_{t+1} = o_1, O_{t+2} = o_2, \ldots, O_{t+k} = o_k \mid H_t = h, A_t = a_1, A_{t+1} = a_2, \ldots\}$$

For a set of **core tests** $\tau_1, \tau_2, \ldots, \tau_d$, the PSR is:

$$f(h) = [p(\tau_1 | h), \; p(\tau_2 | h), \; \ldots, \; p(\tau_d | h)]$$

It can be proven that for special sets of core tests, this vector is a **Markov state** — it satisfies the Markov criterion by definition, since it fully characterizes the distribution of future observations.

## Tiger Problem Example

In the [[POMDP#The Tiger Problem|Tiger problem]], all information can be captured by just two tests:
- $p(HL \mid h, L)$ — probability of hearing left if we listen
- $p(HR \mid h, L)$ — probability of hearing right if we listen

These probabilities can be learned from data (e.g., with an LSTM classifier).

## Advantages and Disadvantages

| Advantages | Disadvantages |
|------------|---------------|
| Test probabilities learnable from data | Limited to tabular setting (extensions exist) |
| As compact or more so than belief states | Finding core tests can be difficult |
| Can still be updated recursively | Less intuitive than belief states |
| No model of hidden states needed | |

## Connections

- Alternative to [[Belief State]] for [[POMDP]]s
- Both satisfy the Markov criterion for [[Partial Observability]]
- More practical when the hidden state model is unknown
- Can be learned with methods like LSTM classifiers

## Appears In

- [[RL-L13 - Partial Observability]]
- [[RL-Book Ch17 - Frontiers]] (§17.3)
- Littman, Sutton & Singh, "Predictive Representations of State" (2001)
