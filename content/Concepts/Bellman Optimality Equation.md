---
type: concept
aliases: [Bellman optimality equation, optimal Bellman equation]
course: [RL]
tags: [foundations, key-formula]
status: complete
---

# Bellman Optimality Equation

See [[Bellman Equation#Bellman Optimality Equations]] for the full treatment.

> [!formula] For $v_*$
> $$v_*(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma \, v_*(s')]$$

> [!formula] For $q_*$
> $$q_*(s,a) = \sum_{s',r} p(s',r|s,a)\left[r + \gamma \max_{a'} q_*(s',a')\right]$$

Solved by: [[Value Iteration]], [[Policy Iteration]], [[Q-Learning]] (sample-based)

## Appears In

- [[RL-L01 - Intro, MDPs & Bandits]], [[RL-L02 - Dynamic Programming]]
