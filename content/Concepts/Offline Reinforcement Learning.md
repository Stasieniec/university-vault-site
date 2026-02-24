---
type: concept
aliases: [offline RL, batch RL]
course: [RL]
tags: [deep-rl]
status: complete
---

# Offline Reinforcement Learning

> [!definition] Offline RL
> **Offline RL** (also called batch RL) learns a policy from a **fixed dataset** of previously collected transitions, without any further interaction with the environment.

Key challenge: **distribution shift** — the learned policy may select actions not well-represented in the dataset, leading to unreliable value estimates.

Solutions: [[Conservative Q-Learning (CQL)]], BCQ, BEAR, IQL — all constrain the learned policy to stay close to the dataset distribution.

## Appears In

- [[RL-L08 - Deep RL Value-Based]]
