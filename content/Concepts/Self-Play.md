---
type: concept
aliases: [self-play, Self-play, self play]
course: [AGAI, RL]
tags: [multi-agent, marl, deep-rl, exam-topic]
status: complete
---

# Self-Play

## Definition

> [!definition] Self-play
> Training an agent's policy by letting it play against itself: **the same policy chooses the actions for each agent**. It requires agents with **symmetric roles** and **egocentric observations**, so that one policy can be used from the perspective of both agents.

## Intuition

> [!intuition] Your sparring partner improves as fast as you do
> An agent that trains against a fixed opponent stops improving once it beats that opponent. In self-play the opponent is the agent itself, so every improvement also makes the opponent stronger, and the training data always comes from play at the agent's own level. No human games or hand-made opponents are needed.

## Mathematical Formulation

In a two-player zero-sum game with policy $\pi_\theta$, both players' moves are sampled from $\pi_\theta$, each seeing the position from its own side (egocentric observation). With a single reward $z$ at the end of the game ($+1$ win, $-1$ loss for the player concerned), a REINFORCE update on one player's moves is

$$\theta \leftarrow \theta + \eta\, z\sum_t\nabla_\theta\log\pi(a_t \mid s_t;\theta)$$

which is how AlphaGo improved its policy network after the supervised stage. AlphaGo Zero and AlphaZero instead combine self-play with [[Monte Carlo Tree Search (MCTS)|MCTS]]: the network guides the search, and the search's move probabilities and the game outcome become the network's training targets.

## Key Properties / Variants

- **Needs symmetric roles.** One policy plays both sides, so the two agents must have the same kind of role, actions and observations (board games such as chess, shogi and Go). AlphaZero is self-play with a homogeneous population of one policy.
- **Can cycle** in games with non-transitive strategies (rock beats scissors, scissors beats paper, paper beats rock): training only against the latest self can forget how to beat older strategies (MARL book).
- **Population-based training** and **PSRO** generalise it: train against a population of policies, which may have different roles, and against mixtures of past policies.

## Connections

- Systems built on it: [[AlphaGo Zero]] (AlphaGo Zero, AlphaZero)
- Search it is combined with: [[Monte Carlo Tree Search (MCTS)]]
- Generalisations: [[Population-Based Training]], [[Policy Space Response Oracles (PSRO)]]
- One policy for all agents, as in [[Parameter Sharing]]
- Policy update: [[REINFORCE]], [[Policy Gradient Methods]]

## Appears In

- [[AGAI-L05 - Deep MARL, Part II]] (definition and requirements)
- [[AGAI-L06a - From Deep MARL to Reasoning]] (AlphaGo, AlphaGo Zero, AlphaZero; its limits)
