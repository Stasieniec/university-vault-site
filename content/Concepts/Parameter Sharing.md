---
type: concept
aliases: [parameter sharing, Experience Sharing, experience sharing, homogeneous agents, strongly homogeneous, weakly homogeneous]
course: [AGAI]
tags: [multi-agent, marl, deep-rl, exam-topic]
status: complete
---

# Parameter Sharing

## Definition

> [!definition] Parameter sharing
> In deep MARL, all agents **use the same set of parameter values** in their neural networks: one policy or value network $\theta$ for the whole team instead of one $\theta_i$ per agent. It suits **strongly homogeneous** agents.

> [!definition] Experience sharing
> Each agent keeps **its own network**, but all agents train on a **centralised buffer of experiences** collected by every agent. It suits **weakly homogeneous** agents.

> [!definition] Homogeneous agents
> Agents are homogeneous when their policies can be interchanged while preserving the overall result, regardless of which agent executes which policy.
> - **Weakly homogeneous:** for any joint policy, permuting the policies between agents leads to exactly the same outcomes (returns).
> - **Strongly homogeneous:** the optimal joint policy consists of identical policies for all agents (in the MARL book, on top of weak homogeneity).

## Intuition

> [!intuition] One brain or one memory
> If every agent should behave the same way, there is no reason to train $n$ copies of the same network: train one, and feed it everyone's experience. If the agents are interchangeable but should play different roles, they still need separate networks, but each can learn from what the others went through, because the others' situations are ones it could have been in.

## Mathematical Formulation

With parameter sharing, agent $i$ acts with $\pi(a_i \mid h_i;\theta)$ or $Q(h_i, a_i;\theta)$, the same $\theta$ for all $i$; the loss sums over all agents' transitions:

$$\mathcal{L}(\theta) = \sum_{i}\mathcal{L}_i(\theta)$$

so every gradient step uses the experience of all agents. A common addition is to append the agent index $i$ to $h_i$, which lets one network implement different behaviours.

## Key Properties / Variants

**Parameter sharing**
- Number of parameters **stays constant** as agents are added.
- Shared parameters learn from the more diverse experience of all agents, improving **sample efficiency**.
- **Constrains all agents to the same policy.** Agents still act differently when their observations differ, but in symmetric situations they act alike.
- Needs no exchange of observations at execution: each agent runs its own copy on its own observations.
- Fits when optimal policies are identical; works poorly when they differ, unless agent indices are observed.

**Experience sharing**
- More expensive: still $n$ networks to train.
- Applies to less restrictive domains, because learned policies can differ.
- On-policy algorithms need a correction (importance sampling), because other agents' trajectories come from other policies; shared experience actor-critic (SEAC) is the MARL book's example.

> [!example] One landmark or two
> Same actions and observations, and swapping policies does not change the return. **A:** both agents must reach the same landmark, so the optimal policies are identical: strongly homogeneous, parameter sharing fits without agent indices. **B:** they must reach two different landmarks: weakly homogeneous only, since identical policies could not reliably split the agents. Experience sharing suits B.

## Connections

- Algorithms it is combined with: [[Independent DQN]], [[Independent Learning]], [[Value Decomposition Networks (VDN)]], [[QMIX]], [[PPO]]
- Training mode: [[Centralised Training with Decentralised Execution]]
- Replay: [[Experience Replay]]
- Self-play as one policy for symmetric opponents: [[AlphaGo Zero]]

## Appears In

- [[AGAI-L05 - Deep MARL, Part II]]
