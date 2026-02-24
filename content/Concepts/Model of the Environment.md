---
type: concept
aliases: [transition model, world model, dynamics]
course: [RL]
tags: [foundations]
status: complete
---

# Model of the Environment

## Definition

> [!definition] Model
> A **Model** of the environment is something that mimics the behavior of the environment, or more generally, allows one to make inferences about how the environment will behave. It is defined by the transition and reward probabilities:
> $$p(s', r | s, a) = \Pr\{S_t = s', R_t = r | S_{t-1} = s, A_{t-1} = a\}$$

## Model-Based vs. Model-Free

- **Model-Free RL**: The agent learns purely through trial and error (experience) without ever explicitly trying to learn the transition dynamics $p$. 
  - *Examples*: [[Q-Learning]], [[SARSA]], [[REINFORCE]].
- **Model-Based RL**: The agent either uses a given model or learns one from experience. It uses the model for **planning** — considering future states before they are actually experienced.
  - *Examples*: Dyna-Q, MCTS (Monte Carlo Tree Search), AlphaZero.

## Types of Models

1. **Distribution Models**: Produce a full probability distribution over next states and rewards.
2. **Sample Models**: Produce a single sample next state and reward (often easier to obtain/build).

## Key Properties

- **Planning**: Having a model enables architectures like [[Dyna]] to improve the policy using "simulated" experience.
- **Inaccuracy (Model Bias)**: If a learned model is imperfect, planning with it can lead to suboptimal or even disastrous real-world performance.
- **Search**: Models enable lookahead search (like in Chess or Go).

## Intuition

> [!intuition] The Agent's Imagination
> A model is the agent's "internal simulator." If I am in state $s$ and I take action $a$, what will happen? A model-free agent has to actually take the action to find out. A model-based agent can "think" about the result without executing the action.

## Connections

- Used in: [[Dynamic Programming]] (requires a perfect model)
- Integrated into: [[Dyna]] (combines model-free and model-based)
- Defines: Markov Decision Process (MDP) dynamics

## Appears In

- [[RL-L01 - Intro & MDPs]]
- [[RL-L02 - Bellman Equations]]
- [[RL-Book Ch8 - Planning and Learning with Tabular Methods]]
