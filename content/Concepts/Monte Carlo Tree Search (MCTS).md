---
type: concept
aliases: [MCTS, Monte Carlo Tree Search]
course: [RL]
tags: [planning, deep-rl, exam-topic]
status: complete
---

# Monte Carlo Tree Search (MCTS)

> [!definition] Monte Carlo Tree Search
> A planning algorithm that builds a search tree incrementally by combining tree search with [[Monte Carlo Methods|Monte Carlo]] sampling (rollouts). At each step it **selects** a promising node, **expands** it, **simulates** a rollout from it, and **backpropagates** the result up the tree.

## Intuition

Instead of exhaustively exploring a game/decision tree (impossible for large state spaces), MCTS focuses search on the most promising branches by using random simulations to estimate state values. Over many iterations, it converges to better estimates for states visited more often.

## The Four Phases

```
1. SELECTION     — From root, use a tree policy (e.g., UCB1) to traverse 
                   existing nodes until reaching a leaf or unexpanded node
2. EXPANSION     — Add one (or more) child nodes to the tree
3. SIMULATION    — From the new node, run a rollout (random or heuristic 
                   policy) until a terminal state → get reward
4. BACKPROPAGATION — Propagate the reward back up the tree, updating 
                     visit counts N(s) and value estimates Q(s)
```

## UCB1 Selection (UCT)

The most common tree policy uses [[Upper Confidence Bound|UCB]] adapted for trees (UCT — Upper Confidence bounds applied to Trees):

$$a^* = \arg\max_a \left[ Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}} \right]$$

where:
- $Q(s,a)$ — average return from taking action $a$ in state $s$
- $N(s)$ — visit count for state $s$
- $N(s,a)$ — visit count for action $a$ in state $s$
- $c$ — exploration constant (balances [[Exploration vs Exploitation]])

## Key Properties

- **Anytime**: returns better estimates the more iterations it gets
- **Asymptotically optimal**: converges to minimax with enough iterations
- **No domain knowledge required** (but benefits from good rollout policies)
- **Scales to huge state spaces**: doesn't need to explore the full tree

## Connection to RL

MCTS is a form of **planning at decision time** — it plans from the current state using a model of the environment (the game rules). This connects to:
- [[Dyna]] — integrating planning with learning
- [[Model of the Environment]] — MCTS requires a generative model
- Rollout algorithms (§8.10 in Sutton & Barto) — MCTS generalizes rollouts with the tree structure

## Applications

- **AlphaGo** / **AlphaGo Zero**: MCTS + deep neural networks for Go
- Board games (Chess, Shogi) — AlphaZero
- General game playing, real-time strategy games

## Connections

- Builds on [[Multi-Armed Bandit]] (UCB) and [[Monte Carlo Methods]]
- Related to [[Dynamic Programming]] (both compute values, different strategies)
- Key component of [[Deep Reinforcement Learning]] systems for games

## Appears In

- [[RL-Book Ch8 - Planning and Learning]] (§8.11)
- [[RL-Book Ch16 - Applications and Case Studies]] (§16.6 — AlphaGo)
