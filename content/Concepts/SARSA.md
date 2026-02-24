---
type: concept
aliases: [Sarsa, SARSA, sarsa]
course: [RL]
tags: [tabular-methods, key-formula, exam-topic]
status: complete
---

# SARSA

## Definition

> [!definition] SARSA
> **SARSA** is an **on-policy** [[Temporal Difference Learning|TD]] control algorithm. It learns action-value function $q_\pi$ for the policy currently being followed. The name comes from the quintuple used in each update: $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$.

## Update Rule

> [!formula] SARSA Update
> $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$
> 
> where:
> - $A_{t+1}$ is the **actual next action** chosen by the current policy (e.g., ε-greedy)
> - Contrast with [[Q-Learning]] which uses $\max_a Q(S_{t+1}, a)$ instead

## Algorithm

```pseudo
Algorithm: SARSA (On-Policy TD Control)
───────────────────────────────────────
Initialize Q(s,a) arbitrarily for all s,a
  (Q(terminal, ·) = 0)

Loop for each episode:
  Initialize S
  Choose A from S using policy derived from Q (e.g., ε-greedy)
  Loop for each step of episode:
    Take action A, observe R, S'
    Choose A' from S' using policy derived from Q (e.g., ε-greedy)
    Q(S,A) ← Q(S,A) + α[R + γ Q(S',A') - Q(S,A)]
    S ← S';  A ← A'
  until S is terminal
```

> [!intuition] On-Policy = What You Do Is What You Learn About
> SARSA updates Q toward the value of the action it **actually takes** next ($A_{t+1}$, chosen by the ε-greedy policy). This means it learns the value of the ε-greedy policy, including the cost of occasional random actions. This can lead to "safer" behavior in risky environments.

## Cliff Walking Example

> [!example] Cliff Walking: SARSA vs Q-Learning
> In the Cliff Walking gridworld (a cliff along the bottom edge with -100 reward for falling off):
> - **Q-Learning** learns the **optimal** path: right along the cliff edge (shortest, but risky with ε-greedy exploration → occasional falls)
> - **SARSA** learns a **safer** path: goes around the top (longer but avoids cliff, because it accounts for the ε-probability of falling off during exploration)
> 
> Q-learning's online performance is worse (more cliff falls during training) even though its learned policy is optimal. SARSA's learned policy is suboptimal but safer under ε-greedy execution.

## Connections

- Instance of: [[Temporal Difference Learning]] (on-policy control)
- Compared with: [[Q-Learning]] (off-policy), [[Expected SARSA]]
- Extended by: Semi-gradient SARSA (with [[Function Approximation]])
- Framework: [[Generalized Policy Iteration]]

## Appears In

- [[RL-L04 - Temporal Difference Learning]]
- [[RL-Book Ch6 - Temporal-Difference Learning]]
- [[RL-CA03 - Temporal Difference]]
- [[RL-ES02 - Exercise Set Week 2]]
