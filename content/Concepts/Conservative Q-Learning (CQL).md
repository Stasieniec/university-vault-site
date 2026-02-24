---
type: concept
aliases: [CQL, Conservative Q-Learning]
course: [RL]
tags: [deep-rl, exam-topic]
status: complete
---

# Conservative Q-Learning (CQL)

> [!definition] Conservative Q-Learning
> **CQL** (Kumar et al., 2020) is an [[Offline Reinforcement Learning|offline RL]] algorithm that learns a **conservative** (pessimistic) Q-function. It penalizes Q-values for actions not seen in the dataset, preventing overestimation on out-of-distribution actions.

## The Offline RL Problem

> [!warning] Why Standard Q-Learning Fails Offline
> In offline RL, we learn from a fixed dataset without further environment interaction. Standard [[Q-Learning]] / [[Deep Q-Network (DQN)|DQN]] can overestimate Q-values for actions not in the dataset (because $\max_a Q(s',a)$ might select an action we've never seen, whose Q-value is unreliable). This causes **extrapolation error** that compounds through bootstrapping.

## CQL Key Idea

Add a regularizer that **pushes down** Q-values for all actions, then **pushes up** Q-values for actions in the dataset:

> [!formula] CQL Objective (simplified)
> $$\min_Q \; \alpha \left( \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp Q(s,a) \right] - \mathbb{E}_{(s,a) \sim \mathcal{D}} [Q(s,a)] \right) + \text{standard TD loss}$$
> 
> - First term: pushes down Q-values for all actions (via logsumexp, which emphasizes high Q actions)
> - Second term: pushes up Q-values for dataset actions
> - Net effect: Q-values for **unseen actions** are conservatively low

## Connections

- Addresses: [[Offline Reinforcement Learning]] distribution shift
- Extends: [[Deep Q-Network (DQN)]]
- Alternative: BCQ, BEAR, IQL (other offline RL methods)

## Appears In

- [[RL-L08 - Deep RL Value-Based]]
