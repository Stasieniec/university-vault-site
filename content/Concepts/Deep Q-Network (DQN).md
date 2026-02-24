---
type: concept
aliases: [DQN, Deep Q-Network, deep Q-network]
course: [RL]
tags: [deep-rl, exam-topic, key-formula]
status: complete
---

# Deep Q-Network (DQN)

## Definition

> [!definition] Deep Q-Network
> **DQN** (Mnih et al., 2015) is a deep RL algorithm that combines [[Q-Learning]] with deep [[Neural Network Function Approximation|neural networks]] to handle high-dimensional state spaces (e.g., raw pixel input). It introduced two key stabilization techniques — [[Experience Replay]] and [[Target Network]] — to address the instability of combining [[Function Approximation]] with bootstrapping.

## Architecture

- **Input**: State representation (e.g., 4 stacked frames of Atari game pixels: $84 \times 84 \times 4$)
- **Network**: Convolutional neural network
- **Output**: $Q(s, a; \boldsymbol{\theta})$ for **all actions** simultaneously (one output per action)
- **Action selection**: $A_t = \arg\max_a Q(S_t, a; \boldsymbol{\theta})$ (with ε-greedy for exploration)

## Loss Function

> [!formula] DQN Loss
> $$L(\boldsymbol{\theta}) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \boldsymbol{\theta}^-) - Q(s, a; \boldsymbol{\theta}) \right)^2 \right]$$
> 
> where:
> - $\boldsymbol{\theta}$ — current network weights (being trained)
> - $\boldsymbol{\theta}^-$ — **target network** weights (frozen copy, updated periodically)
> - $\mathcal{D}$ — [[Experience Replay|replay buffer]] of past transitions
> - $(s, a, r, s')$ — sampled transition

## Key Innovation 1: [[Experience Replay]]

> [!definition] Experience Replay
> Store transitions $(S_t, A_t, R_{t+1}, S_{t+1})$ in a replay buffer $\mathcal{D}$. Train by sampling **random mini-batches** from $\mathcal{D}$ rather than using consecutive transitions.

**Why it helps:**
1. **Breaks temporal correlations**: Consecutive samples are highly correlated → bad for SGD. Random sampling decorrelates.
2. **Data efficiency**: Each transition can be reused many times.
3. **Smooths over data distribution**: Averages over many past policies' behavior.

## Key Innovation 2: [[Target Network]]

> [!definition] Target Network
> A separate copy of the Q-network with weights $\boldsymbol{\theta}^-$, updated to match $\boldsymbol{\theta}$ only every $C$ steps (or via soft update $\boldsymbol{\theta}^- \leftarrow \tau\boldsymbol{\theta} + (1-\tau)\boldsymbol{\theta}^-$).

**Why it helps:**
- Without it, the target $r + \gamma \max_{a'} Q(s', a'; \boldsymbol{\theta})$ changes with every weight update → **moving target problem** → instability
- Freezing the target network for $C$ steps provides a stable regression target

## Algorithm

```pseudo
Algorithm: Deep Q-Network (DQN)
───────────────────────────────
Initialize replay buffer D (capacity N)
Initialize Q-network with random weights θ
Initialize target network with weights θ⁻ = θ

For episode = 1 to M:
  Initialize state S₁ (e.g., preprocess game frame)
  For t = 1 to T:
    With probability ε: select random action Aₜ
    Otherwise: Aₜ = argmax_a Q(Sₜ, a; θ)
    
    Execute Aₜ, observe Rₜ₊₁, Sₜ₊₁
    Store (Sₜ, Aₜ, Rₜ₊₁, Sₜ₊₁) in D
    
    Sample random minibatch of (sⱼ, aⱼ, rⱼ, s'ⱼ) from D
    Set yⱼ = rⱼ + γ max_{a'} Q(s'ⱼ, a'; θ⁻)  [or yⱼ = rⱼ if terminal]
    
    Perform gradient descent on (yⱼ - Q(sⱼ, aⱼ; θ))² w.r.t. θ
    
    Every C steps: θ⁻ ← θ
```

## DQN Improvements

| Variant | Key Idea |
|---------|----------|
| **Double DQN** | Use online network to select action, target network to evaluate: $y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \boldsymbol{\theta}); \boldsymbol{\theta}^-)$. Reduces overestimation bias. |
| **Dueling DQN** | Separate network streams for $V(s)$ and advantage $A(s,a)$: $Q(s,a) = V(s) + A(s,a) - \text{mean}(A)$ |
| **Prioritized Replay** | Sample important transitions (high TD error) more frequently |

## Relation to the [[Deadly Triad]]

DQN has all three deadly triad elements (FA + bootstrapping + off-policy). It works in practice due to:
- Experience replay → stabilizes data distribution
- Target network → stabilizes bootstrap targets
- No theoretical convergence guarantee, but empirically very successful

## [[Conservative Q-Learning (CQL)]]

Extension for **offline RL** (learning from fixed datasets without environment interaction). Adds a regularizer that pushes down Q-values for unseen actions to avoid overestimation on out-of-distribution actions.

## Connections

- Extends: [[Q-Learning]] with deep neural networks
- Stabilized by: [[Experience Replay]], [[Target Network]]
- Addresses: [[Deadly Triad]] (practically, not theoretically)
- Offline variant: [[Conservative Q-Learning (CQL)]]
- Alternatives: [[SARSA]] variants, policy gradient methods

## Appears In

- [[RL-L08 - Deep RL Value-Based]]
- [[RL-Book Ch16 - Applications and Case Studies]] (§16.5)
