---
type: concept
aliases: [ε-greedy, epsilon-greedy, e-greedy, Epsilon-Greedy]
course: [RL]
tags: [foundations, exam-topic]
status: complete
---

# Epsilon-Greedy Policy

> [!definition] ε-Greedy Policy
> An **ε-greedy policy** selects the greedy action (highest estimated value) with probability $1 - \varepsilon$, and a uniformly random action with probability $\varepsilon$.

> [!formula] ε-Greedy Action Probabilities
> $$\pi(a|s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}(s)|} & \text{if } a = \arg\max_{a'} Q(s, a') \\ \frac{\varepsilon}{|\mathcal{A}(s)|} & \text{otherwise} \end{cases}$$

- Simplest method for [[Exploration vs Exploitation]] balance
- ε-greedy is a special case of **ε-soft** policies (where $\pi(a|s) \geq \varepsilon / |\mathcal{A}|$ for all $a$)
- Common to decay $\varepsilon$ over time: high early (explore) → low later (exploit)

## Connections

- Used by: [[SARSA]], [[Q-Learning]], [[Monte Carlo Control]], [[Deep Q-Network (DQN)]]
- Alternative: [[Upper Confidence Bound]], Boltzmann/softmax

## Appears In

- [[RL-L01 - Intro, MDPs & Bandits]], [[RL-L03 - Monte Carlo Methods]], [[RL-L04 - Temporal Difference Learning]]
