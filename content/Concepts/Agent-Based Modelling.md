---
type: concept
aliases: [ABM, agent-based modeling, agent-based model, agent-based models]
course: [AGAI]
tags: [agents, simulation]
status: complete
---

# Agent-Based Modelling

## Definition

> [!definition] Agent-based modelling (ABM)
> Studying a system by specifying the behaviour of its individual agents and simulating their interactions, so that the system's behaviour emerges from them. Two main approaches specify the agents' behaviour: **cellular automata**, generally as if-then rules, and **equations**, possibly differential (ODE or PDE).

## Intuition

> [!intuition] Simple rules, complex wholes
> In cellular automaton Rule 22, a cell becomes 1 if there is exactly one 1 in its three-cell neighbourhood (left, centre, right). That one rule, applied everywhere, produces intricate global patterns. Conway's Game of Life and Schelling's segregation model work the same way: nobody programs the global outcome, it emerges from local rules. (Outside the course material: in Schelling's model, agents of two kinds move when too few neighbours are like them, and mild individual preferences produce strongly segregated neighbourhoods.)

## Key Properties / Variants

- **Cellular automata and rules.** Behaviour as if-then rules over a cell's neighbourhood. **Agent-oriented programming**, for example of [[BDI Model|BDI agents]], is a more expressive approach in the same arena: rules over beliefs, desires and intentions.
- **Equations.** Ordinary or partial differential equations over aggregate quantities, traditionally more associated with economics (for example price dynamics).
- **Network science.** When studying spreading dynamics (diseases, rumours), the topology enabling interactions becomes particularly relevant.
- **In game theory**, two traditional approaches:
  - **Classic game theory:** perfectly rational agents, forward-looking cognition (anticipation and decision before action), agents generally different and in very small numbers. Example: the [[Prisoner's Dilemma]].
  - **Evolutionary game theory:** bounded, possibly non-rational agents, replicative cognition (imitation) that is possibly adaptive, agents generally similar and in great numbers. Studied with [[Replicator Dynamics]].

## Connections

- Agents modelled: [[Intelligent Agent]], [[BDI Model]]
- Many agents together: [[Multi-Agent System]]
- Game-theoretic versions: [[Prisoner's Dilemma]], [[Replicator Dynamics]]

## Appears In

- [[AGAI-L06b - What Would Make a Chatbot an Agent]]
