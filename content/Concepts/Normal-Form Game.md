---
type: concept
aliases: [normal-form game, normal-form games, normal form game, strategic-form game, matrix game, one-shot game]
course: [AGAI]
tags: [game-theory, multi-agent, exam-topic]
status: complete
---

# Normal-Form Game

## Definition

> [!definition] Normal-form game
> A **normal-form game** defines an interaction between two or more agents with no time and no state, only actions and rewards. It is a 3-tuple $(I, \{A_i\}_{i \in I}, \{R_i\}_{i \in I})$:
> - $I$: a finite set of agents
> - $A_i$: a finite set of actions for each agent $i \in I$
> - $R_i : A \to \mathbb{R}$: the reward function of agent $i$, where $A = A_1 \times \dots \times A_n$ is the set of joint actions

## Intuition

> [!intuition] One simultaneous choice
> All agents choose an action at the same time, without seeing each other's choice, and each receives a reward that depends on the whole combination. With two agents the game is a table: rows for agent 1's actions, columns for agent 2's, and in each cell the pair (row agent's reward, column agent's reward). Nothing carries over: there is no state that the actions change and no later round.

> [!example] Prisoner's Dilemma
> | | C | D |
> |---|---|---|
> | **C** | 3, 3 | 0, 5 |
> | **D** | 5, 0 | 1, 1 |
>
> Two agents, two actions each (cooperate, defect), four joint actions. Agent 1's reward for joint action (D, C) is 5, agent 2's is 0.

## Mathematical Formulation

Agents follow **policies**: agent $i$ samples action $a_i \in A_i$ with probability $\pi_i(a_i)$. A policy that always plays one action is **pure**; one that randomises is **mixed**. With independent choices, the expected return of agent $i$ under the joint policy $\pi = (\pi_1, \dots, \pi_n)$ is

$$U_i(\pi) = \sum_{a \in A} \Big( \prod_{j \in I} \pi_j(a_j) \Big) R_i(a)$$

where:
- $a = (a_1, \dots, a_n)$: a joint action
- $\prod_j \pi_j(a_j)$: the probability that this joint action is played
- $R_i(a)$: agent $i$'s reward for it

Solution concepts such as the [[Best Response]], the [[Nash Equilibrium]] and [[Pareto Optimality]] are all defined in terms of $U_i$.

## Key Properties / Variants

- **Zero-sum:** rewards in every cell sum to zero, so one agent's gain is the other's loss (Matching Pennies, Rock-Paper-Scissors).
- **Common-reward:** all agents receive the same reward in every cell, so their interests coincide.
- **General-sum:** any other relation; interests partly align and partly conflict ([[Prisoner's Dilemma]], [[Stag Hunt]], Hawk-Dove).
- **Repeated normal-form game:** the same game played over time steps, with policies conditioned on the history of joint actions $h^t = (a^0, \dots, a^{t-1})$. See [[Repeated Game]].
- **Bottom of the hierarchy:** one-shot normal-form game, repeated normal-form game, [[Stochastic Game|stochastic game]] (adds states), partially observable stochastic game (adds private observations). A repeated normal-form game is a stochastic game with a single state.
- **Existence of equilibrium:** every finite normal-form game has at least one Nash equilibrium, possibly mixed (Nash, 1950).

## Connections

- Solution concepts: [[Best Response]], [[Nash Equilibrium]], [[Pareto Optimality]]
- Examples: [[Prisoner's Dilemma]], [[Stag Hunt]], [[Social Dilemma]]
- Generalisations: [[Repeated Game]], [[Stochastic Game]]
- Population view: [[Replicator Dynamics]]
- Setting: [[Multi-Agent System]]

## Appears In

- [[AGAI-L02 - Game Theory and Population Dynamics]]
