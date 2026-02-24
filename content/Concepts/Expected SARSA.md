---
type: concept
aliases: [Expected SARSA, expected sarsa]
course: [RL]
tags: [tabular-methods, key-formula]
status: complete
---

# Expected SARSA

> [!formula] Expected SARSA Update
> $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

> [!intuition] Between SARSA and Q-Learning
> Instead of sampling a single next action $A_{t+1}$ ([[SARSA]]) or taking the max ([[Q-Learning]]), Expected SARSA takes the **expectation** over all possible next actions under the current policy. This reduces variance compared to SARSA while being more general than Q-learning.

- With a greedy target policy: Expected SARSA = Q-learning
- With an ε-greedy target policy: Expected SARSA accounts for the exploration probability
- Generally lower variance than SARSA, slightly more computation per step

## Appears In

- [[RL-L04 - Temporal Difference Learning]], [[RL-Book Ch6 - Temporal-Difference Learning]]
