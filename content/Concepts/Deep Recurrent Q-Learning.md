---
type: concept
aliases: [DRQN, Deep Recurrent Q-Learning, Deep Recurrent Q-Network]
course: [RL]
tags: [deep-rl]
status: complete
---

# Deep Recurrent Q-Learning (DRQN)

> [!definition] Deep Recurrent Q-Learning
> An extension of [[Deep Q-Network (DQN)]] for [[Partial Observability|partially observable]] environments. DRQN replaces the first fully connected layer of DQN with an **LSTM** (Long Short-Term Memory) layer, enabling the network to maintain an internal memory across timesteps.

## Architecture

```
Standard DQN:    [Observation] → [Conv layers] → [FC layers] → Q(s,a)

DRQN:            [Observation] → [Conv layers] → [LSTM] → [FC layers] → Q(o,h,a)
```

The LSTM processes a sequence of observations over time, maintaining a hidden state that aggregates information from past observations. This hidden state serves as an approximate internal state for the [[POMDP]].

## Training Strategies

Two approaches for unrolling the LSTM during training:

1. **Bootstrapped random updates**: Sample random starting points in episodes, unroll LSTM for a fixed number of steps. The LSTM hidden state starts from zero.
2. **Sequential updates**: Process episodes sequentially, carrying the LSTM hidden state forward. More accurate but less diverse sampling.

## Key Properties

- Handles partial observability by learning to aggregate information over time
- The LSTM hidden state acts as a learned internal state (approximating a [[Belief State]])
- Works with [[Experience Replay]], though care is needed with LSTM state initialization
- Simple modification to DQN — just swap one layer

## Connections

- Extends [[Deep Q-Network (DQN)]] to partially observable settings
- Addresses [[Partial Observability]] / [[POMDP]]
- Alternative to frame stacking (which is a simpler approximation)
- Uses LSTM (a type of recurrent neural network)

## Appears In

- [[RL-L13 - Partial Observability]]
- Hausknecht & Stone, "Deep Recurrent Q-Learning for Partially Observable MDPs" (2015)
