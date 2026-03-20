---
type: concept
aliases: [belief state, belief MDP, belief distribution]
course: [RL]
tags: [foundations, exam-topic]
status: complete
---

# Belief State

> [!definition] Belief State
> A **belief state** $b_t$ is a probability distribution over the hidden states of a [[POMDP]], representing the agent's uncertainty about which state it is in given its history of observations and actions:
> $$b_t(x) = \Pr(X_t = x \mid H_t = h_t)$$

## Bayesian Update

> [!formula] Belief State Update
> After taking action $a_t$ and observing $o_{t+1}$, the belief is updated via Bayes' rule:
> $$b_{t+1}(x') \propto p(o_{t+1}|x') \sum_{x \in X} p(x'|x, a_t) \, b_t(x)$$
>
> where:
> - $p(o_{t+1}|x')$ — observation likelihood (how likely is this observation if the true state is $x'$)
> - $p(x'|x, a_t)$ — transition probability
> - $b_t(x)$ — prior belief about state $x$
> - The result is normalized to sum to 1

## Intuition

> [!intuition] Tracking Where You Might Be
> Imagine you're in a dark room. You can't see where you are (hidden state), but you can feel around (observations). Your belief state is your mental map of where you think you are — a probability over all possible locations. Each time you move and get a new sensory input, you update this mental map using Bayes' rule.

## The Tiger Problem (Classic Example)

A tiger is behind one of two doors. The agent can:
- **Open Left (OL)**: reward +100 if treasure, -100 if tiger
- **Open Right (OR)**: reward +100 if treasure, -100 if tiger
- **Listen (L)**: reward -1, get noisy observation (85% correct)

**Belief evolution** (from the lecture, starting at $b(HL) = 0.5$):

| Step | Action | Observation | $b(HL)$ | Best action value |
|------|--------|-------------|---------|-------------------|
| Start | — | — | 0.50 | Listen: $Q^* = 5.6$ |
| 1 | Listen | Hear Left | 0.85 | Listen: $Q^* = 6.6$ |
| 2 | Listen | Hear Left | 0.97 | Listen: $Q^* = 8.0$ |
| 3 | Listen | Hear Left | ~0.995 | Open Right becomes best |

After enough consistent observations, the agent becomes confident enough to open the door.

## Key Properties

- The belief state is a **sufficient statistic** for the history — it captures all relevant information
- The belief state MDP is **fully observable** (we know what belief we're in)
- Planning (e.g., [[Dynamic Programming]]) in belief space yields the **optimal** POMDP policy
- Belief states live in a continuous space (a probability simplex) even if the underlying state space is discrete

## Advantages and Disadvantages

| Advantages | Disadvantages |
|------------|---------------|
| Concrete meaning: probability over latent states | Requires knowledge of underlying models $T$, $Z$ |
| Relatively compact: dim($b$) = |$X$| | Underlying model difficult to learn |
| Can be updated recursively | Only practical for discrete state spaces |
| Converts POMDP to (continuous) MDP | Continuous belief space makes planning hard |

## Connections

- Central concept in [[POMDP]] theory
- Updated via [[Bayes' Theorem]]
- Alternative: [[Predictive State Representation]] (doesn't require model knowledge)
- Planning in belief space uses [[Dynamic Programming]] or [[Value Iteration]]

## Appears In

- [[RL-L13 - Partial Observability]]
- [[RL-Book Ch17 - Frontiers]] (§17.3)
