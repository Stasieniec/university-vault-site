---
type: concept
aliases: [experience replay, replay buffer, replay memory]
course: [RL]
tags: [deep-rl, exam-topic]
status: complete
---

# Experience Replay

> [!definition] Experience Replay
> **Experience replay** stores agent transitions $(S_t, A_t, R_{t+1}, S_{t+1})$ in a buffer $\mathcal{D}$ and trains by sampling **random mini-batches** from this buffer instead of using consecutive online samples.

**Benefits:**
1. **Breaks correlations**: Consecutive samples are highly correlated → bad for SGD. Random sampling decorrelates the training data.
2. **Data efficiency**: Each transition can be reused in many updates.
3. **Distribution smoothing**: Averages over many past behavior policies.

**Limitations:**
- Memory-intensive (large buffer needed)
- Off-policy by nature (old data from previous policies)
- Not suitable for on-policy methods without modifications

## Appears In

- [[RL-L08 - Deep RL Value-Based]], [[Deep Q-Network (DQN)]]
