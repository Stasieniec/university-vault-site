---
type: concept
aliases: [rollout algorithm, rollout policy, rollout]
course: [RL]
tags: [planning]
status: complete
---

# Rollout Algorithm

> [!definition] Rollout Algorithm
> A **decision-time planning** method that estimates the value of each action from the current state by simulating complete trajectories (rollouts) using a **rollout policy** (also called base policy or default policy). The action with the highest estimated value is then executed.

## How It Works

1. From current state $s_t$, for each candidate action $a$:
   - Simulate a trajectory starting with action $a$, then following the rollout policy $b$ until episode end
   - Record the return $G$
2. Repeat multiple times to get average returns
3. Execute the action with the highest average return

> [!intuition] Look Before You Leap
> Instead of committing to an action based on a learned value function, a rollout algorithm "imagines" what would happen if it took each action and then followed some default strategy. It picks the action that leads to the best imagined outcome. This is planning at decision time — only the current decision is planned, not future ones.

## Key Properties

- Requires a **model** (simulator) of the environment to run rollouts
- The rollout policy $b$ can be simple (even random) — MCTS is an extension that builds a smarter tree policy
- The chosen action is a **policy improvement** over the rollout policy $b$
- Only one action is actually executed in the real environment per planning cycle
- Quality depends on rollout budget (number of simulations) and quality of rollout policy

## Connection to Policy Improvement

> [!formula] Rollout as Policy Improvement
> If $Q^b(s,a)$ is the action-value function under rollout policy $b$, then the rollout algorithm selects:
> $$a^* = \arg\max_a Q^b(s,a)$$
> This is equivalent to one step of policy improvement over $b$.

## Connections

- Generalized by [[Monte Carlo Tree Search (MCTS)]] — adds a tree structure and UCB selection
- Part of [[Model-Based Reinforcement Learning]] — requires a model/simulator
- Related to [[Monte Carlo Methods]] — uses sampled returns to estimate values
- Simpler than full [[Dynamic Programming]] — only plans from the current state

## Appears In

- [[RL-L12 - Model-Based RL]]
- [[RL-Book Ch8 - Planning and Learning]] (§8.10)
