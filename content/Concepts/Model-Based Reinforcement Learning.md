---
type: concept
aliases: [Model-Based RL, MBRL, model-based reinforcement learning]
course: [RL]
tags: [planning, exam-topic]
status: complete
---

# Model-Based Reinforcement Learning

> [!definition] Model-Based Reinforcement Learning
> An approach to [[Reinforcement Learning]] that learns a **model of the environment** — the transition dynamics $p(s'|s,a)$ and reward function $r(s,a)$ — and uses this model for **planning** to improve the policy or value function. Contrasts with model-free methods that learn value functions or policies directly from experience.

## Intuition

> [!intuition] Learning a Mental Model
> Model-free RL learns "what to do" from trial and error. Model-based RL first learns "how the world works" (a model), then uses that model to figure out what to do. This is like learning the rules of chess before deciding on a strategy, rather than just memorizing which moves worked in past games.

## Model Learning

The model is learned from collected data $D = \{(s_i, a_i, r_i, s_i')\}_{i=1}^N$ using supervised learning:

| Setting | Approach |
|---------|----------|
| **Tabular** | Store transition counts, compute $\hat{p}(s'|s,a)$ and $\hat{r}(s,a)$ as averages |
| **Function approximation** | Train neural network, Gaussian process, etc. to predict $s'$ and $r$ from $(s,a)$ |

## Using the Model: Two Approaches

### 1. Background Planning

Generate simulated experience from the model and use it to update value functions/policies **between** real environment steps. Example: [[Dyna]].

### 2. Decision-Time Planning

Plan from the **current state** at the moment a decision is needed. Examples: [[Rollout Algorithm]], [[Monte Carlo Tree Search (MCTS)]].

## Advantages

- **Sample efficient**: can extract more learning from each real interaction by replaying/simulating
- **Transfer**: a good model can be reused across different reward functions or tasks
- **Interpretability**: the model captures environment dynamics explicitly

## Disadvantages

> [!warning] Model Errors Compound
> Model errors accumulate over multi-step rollouts. A small error per step can lead to highly inaccurate predictions over long horizons. This is the fundamental challenge of MBRL.

- Model learning is itself a hard problem (especially in high-dimensional spaces)
- Computational overhead of maintaining and querying the model
- Model bias can lead to suboptimal policies that exploit model inaccuracies

## Big Picture

```
                    Known model              Learned model
Rollout/MCTS    Rollout algorithm,      Rollout algorithm,
                MCTS, AlphaGo          MCTS, AlphaGo

Planning with   Planning               Model-based RL
MF RL tools                            (e.g., Dyna)
```

## Connections

- [[Dyna]] — integrates model learning with background planning
- [[Monte Carlo Tree Search (MCTS)]] — decision-time planning using tree search
- [[AlphaGo Zero]] — MCTS guided by neural networks
- [[Rollout Algorithm]] — decision-time planning using simulated rollouts
- Contrasts with model-free approaches: [[Q-Learning]], [[Policy Gradient Methods]]

## Appears In

- [[RL-L12 - Model-Based RL]]
- [[RL-Book Ch8 - Planning and Learning]]
