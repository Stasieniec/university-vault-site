---
type: concept
aliases: [target network, target Q-network]
course: [RL]
tags: [deep-rl, exam-topic]
status: complete
---

# Target Network

> [!definition] Target Network
> A **target network** is a separate, periodically-updated copy of the Q-network used to compute TD targets during training. Its weights $\boldsymbol{\theta}^-$ are held fixed for $C$ steps, then copied from the online network $\boldsymbol{\theta}$.

**Why needed:**
- Without it, the TD target $r + \gamma \max_{a'} Q(s', a'; \boldsymbol{\theta})$ changes with every gradient step → **moving target problem**
- Freezing the target provides a stable regression objective
- Alternative: **soft update** $\boldsymbol{\theta}^- \leftarrow \tau \boldsymbol{\theta} + (1-\tau)\boldsymbol{\theta}^-$ (Polyak averaging, used in DDPG/SAC)

## Appears In

- [[RL-L08 - Deep RL Value-Based]], [[Deep Q-Network (DQN)]]
