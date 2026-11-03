---
type: concept
aliases: [independent learning, independent learners, independent learner, Independent Q-Learning, independent Q-learning, IQL, Independent Q-learners]
course: [AGAI]
tags: [multi-agent, marl, exam-topic]
status: complete
---

# Independent Learning

## Definition

> [!definition] Independent learning
> In multi-agent reinforcement learning, **independent learning** means that each agent learns its own policy with a single-agent RL algorithm, using only its own actions and rewards, and **treats the other agents as part of the environment**. The standard example is **independent Q-learning (IQL)**: each agent $i$ runs ordinary [[Q-Learning]] on a table $Q^i(s, a^i)$ over its own actions.

It is the decentralized branch of the "naive" approach to MARL, which reuses single-agent methods. The centralized branch instead trains one learner over the joint action of all agents, as if a single agent controlled them all.

## Intuition

> [!intuition] The others are just part of the world
> An independent learner does not know or care that the other things moving around are learners. It sees states, takes its actions, gets rewards, and learns as if it were alone in an MDP. The trouble is that this "environment" keeps changing, because the other agents are learning too: what worked yesterday may not work today. The upside is that each agent only ever deals with its own small action set, however many agents there are.

## Mathematical Formulation

Each agent applies the single-agent update to its own table:

$$Q^i(s,a^i) \leftarrow Q^i(s,a^i) + \alpha\Big[r^i + \gamma\max_{b}Q^i(s',b) - Q^i(s,a^i)\Big]$$

where:
- $a^i$: agent $i$'s own action; the other agents' actions do not appear
- $r^i$: agent $i$'s reward (in a fully cooperative game, the shared team reward)
- $\alpha$, $\gamma$: learning rate and discount factor

The environment agent $i$ effectively learns about averages over the others' policies:

$$P^i(s' \mid s, a^i) = \sum_{a^{-i}}\Big[\prod_{j \neq i}\pi^j(a^j \mid s)\Big]\,P(s' \mid s, a^i, a^{-i})$$

This changes whenever the others' policies $\pi^{-i}$ change, so the MDP each agent faces is not fixed: $\text{MDP}^i_t \neq \text{MDP}^i_{t+k}$.

In a one-state game with two agents and shared reward $R(a,b)$, agent A's Q-value converges towards the reward averaged over B's current behaviour:

$$Q_A(a) \approx \sum_b \pi_B(b)\,R(a,b)$$

## Key Properties / Variants

- **No guarantees.** The environment is non-stationary from each agent's view, so the convergence results of single-agent Q-learning do not apply; best responses can cycle.
- **Miscoordination.** Averaging over the partner hides good joint actions. In the climbing game (shared payoffs 11, −30, 0 / −30, 7, 6 / 0, 0, 5) independent learners settle on the safe action c and reach (c, c) = 5, while the optimum (a, a) = 11 needs both agents to risk −30. Learners tend towards "safe" equilibria; low, decaying exploration helps somewhat but the outcome varies between runs.
- **Scalability.** A centralized learner must track and explore $|A|^N$ joint actions; independent learners together handle $|A| \times N$. With 2 agents and 6 actions: 36 against 2 × 6. In level-based foraging IQL reached the optimal return sooner than centralized Q-learning.
- **Not only Q-learning.** The same idea gives IPPO (independent proximal policy optimisation) and independent versions of DDPG and SAC. Independent learning is a widely used baseline across MARL: messy, but it scales to large, complex environments where central learning is intractable.

## Connections

- Single-agent basis: [[Q-Learning]], [[Reinforcement Learning]]
- Setting: [[Stochastic Game]], [[Multi-Agent System]]
- Problems it runs into: non-stationarity, [[Multi-Agent Credit Assignment]], equilibrium selection ([[Nash Equilibrium]], [[Stag Hunt]])
- Exploration: [[Epsilon-Greedy Policy]], [[Exploration vs Exploitation]]
- Next step: [[Deep Reinforcement Learning]]

## Appears In

- [[AGAI-L03 - Intro to Multi-Agent Reinforcement Learning]]
