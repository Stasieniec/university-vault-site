---
type: concept
aliases: [Q-learning, q-learning]
course: [RL]
tags: [tabular-methods, key-formula, exam-topic]
status: complete
---

# Q-Learning

## Definition

> [!definition] Q-Learning
> **Q-learning** is an **off-policy** [[Temporal Difference Learning|TD]] control algorithm. It directly approximates the optimal action-value function $q_*$, regardless of the policy being followed. The key insight: the update target uses $\max_a Q(S_{t+1}, a)$ — the value of the **best** action in the next state — not the action actually taken.

## Update Rule

> [!formula] Q-Learning Update
> $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$
> 
> where:
> - $\alpha$ — step size (learning rate)
> - $R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$ — TD target (using best next action)
> - $\gamma \max_a Q(S_{t+1}, a)$ — bootstrapped estimate of future value under **optimal** policy

## Algorithm

```pseudo
Algorithm: Q-Learning (Off-Policy TD Control)
──────────────────────────────────────────────
Initialize Q(s,a) arbitrarily for all s,a
  (Q(terminal, ·) = 0)

Loop for each episode:
  Initialize S
  Loop for each step of episode:
    Choose A from S using policy derived from Q
      (e.g., ε-greedy w.r.t. Q)
    Take action A, observe R, S'
    Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
    S ← S'
  until S is terminal
```

## Why Off-Policy?

> [!intuition] The Max Makes It Off-Policy
> The behavior policy (used to select actions) is typically ε-greedy for exploration. But the update target uses $\max_a$ — the greedy policy's value. So we're learning about the **greedy (optimal) policy** while **following an exploratory policy**. 
> 
> Unlike [[SARSA]], Q-learning doesn't need [[Importance Sampling]] corrections because the max operation directly estimates $q_*$.

## Q-Learning vs SARSA

| Property | Q-Learning | [[SARSA]] |
|----------|-----------|---------|
| Type | Off-policy | On-policy |
| Target | $R + \gamma \max_a Q(S', a)$ | $R + \gamma Q(S', A')$ |
| Learns about | Optimal (greedy) policy | Current (ε-greedy) policy |
| Cliff Walking behavior | Finds optimal (risky) path | Finds safer path |
| Convergence | To $q_*$ (with conditions) | To $q_\pi$ for current ε-greedy $\pi$ |

## Convergence

Q-learning converges to $q_*$ under standard conditions:
1. All state-action pairs visited infinitely often
2. Step sizes satisfy: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$

> [!warning] With Function Approximation
> Tabular Q-learning converges. With [[Function Approximation]], Q-learning can **diverge** (the [[Deadly Triad]]). This motivated [[Deep Q-Network (DQN)]]'s stabilization techniques ([[Experience Replay]] + [[Target Network]]).

## Connections

- Instance of: [[Temporal Difference Learning]] (off-policy control)
- Compared with: [[SARSA]] (on-policy), [[Expected SARSA]]
- Extended by: Double Q-learning, [[Deep Q-Network (DQN)]]
- Danger with FA: [[Deadly Triad]]

## Appears In

- [[RL-L04 - Temporal Difference Learning]]
- [[RL-Book Ch6 - Temporal-Difference Learning]]
- [[RL-CA03 - Temporal Difference]]
- [[RL-ES02 - Exercise Set Week 2]]
