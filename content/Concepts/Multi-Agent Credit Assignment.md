---
type: concept
aliases: [multi-agent credit assignment, agent credit assignment, credit assignment in MARL, lazy agent problem]
course: [AGAI]
tags: [multi-agent, marl, exam-topic]
status: complete
---

# Multi-Agent Credit Assignment

## Definition

> [!definition] Multi-agent credit assignment
> The problem of working out **whose actions led to a reward** when several agents act in the same environment. It adds a new axis to single-agent **temporal** credit assignment, which asks which of the agent's earlier actions led to a reward.

## Intuition

> [!intuition] The lazy agent
> Three agents collect apples and all receive +1 whenever any apple is collected. Two of them do the work while the third wanders in a corner. The third still gets +1 each time, so whatever it was doing is reinforced. From its own reward alone it cannot tell that it contributed nothing.

Finding out who contributed requires counterfactuals (what would have happened had this agent acted differently?), which takes many samples and a lot of exploration. It is a causal inference problem with a very low signal-to-noise ratio.

## Mathematical Formulation

In a fully cooperative stochastic game every agent receives the same reward, $r^1 = \dots = r^N = r(s, a^1, \dots, a^N)$. An independent Q-learner then updates

$$Q^i(s,a^i) \leftarrow Q^i(s,a^i) + \alpha\Big[r + \gamma\max_{b}Q^i(s',b) - Q^i(s,a^i)\Big]$$

with the same $r$ for every agent, whatever its own action contributed. The reward carries no information about agent $i$ specifically; only the statistics over many episodes, where agent $i$'s action varies while the others' do too, can reveal its effect.

(Outside the course material: one standard remedy is a **difference reward**, which compares the actual reward with the reward had agent $i$ taken some default action $c^i$ while the others kept theirs, $r(s, a) - r(s, (c^i, a^{-i}))$, turning the counterfactual into a signal per agent.)

## Key Properties / Variants

- **Worst in fully cooperative games**: a shared team reward gives no individual signal at all.
- **Also in general-sum games**: rewards are individual but still depend on the others. In a fishing game with a lake that can be depleted, an agent that fishes aggressively while the others hold back learns that aggression is good, and one that fishes sustainably while the others hold back too, catching little, learns that sustainability is bad. (The AGAI course has the others holding back in both cases; the second is cleaner if the others fish aggressively and cause the small catch.) Each credits its own action with an outcome that depended on everyone (see [[Social Dilemma]]).
- **Grows with the number of agents**: more agents means more candidates for the credit and noisier individual signals.
- **Hurts sample efficiency**: causation has to be inferred from many, many samples.

## Connections

- Single-agent counterpart: temporal credit assignment in [[Reinforcement Learning]] and [[Temporal Difference Learning]]
- Arises for: [[Independent Learning]] in a [[Stochastic Game]]
- Related challenge: non-stationarity (the other agents' changing policies), equilibrium selection
- Setting: [[Multi-Agent System]]

## Appears In

- [[AGAI-L03 - Intro to Multi-Agent Reinforcement Learning]]
