---
type: concept
aliases: [rationality, rational agents]
course: [AGAI]
tags: [agents]
status: complete
---

# Rational Agent

## Definition

> [!definition] Rational agent (Russell and Norvig)
> "For each possible percept sequence, a rational agent should select an action that is **expected to maximize its performance measure**, given the evidence provided by the percept sequence and whatever built-in knowledge the agent has."

What is rational at a given moment depends on four things:
1. the **performance measure** that defines success,
2. the agent's **prior knowledge** of the environment,
3. the **actions** the agent can perform,
4. the agent's **percept sequence** to date.

## Intuition

> [!intuition] Doing the right thing, judged by consequences
> AI judges an [[Intelligent Agent]] by the consequences of what it does (consequentialism). The agent's actions push the environment through a sequence of states, and a **performance measure** scores that sequence. The measure lives in the designer's head (or the user's): a machine has no preferences of its own.

Rationality is relative. The vacuum agent "suck if dirty, else move to the other square" is rational if the measure gives one point per clean square per time step, the geography is known, and the only actions are Left, Right and Suck. Add a penalty of one point per move and the same agent becomes irrational: once everything is clean it keeps shuttling back and forth, and a better agent would stay put.

The same vacuum world shows why writing the performance measure is hard. Measure "amount of dirt cleaned up" and a rational agent can maximise it by cleaning, dumping the dirt back on the floor, and cleaning again. Russell and Norvig's rule: design performance measures according to **what you actually want achieved in the environment**, rather than according to how you think the agent should behave. When the measure is wrong, a perfectly rational agent does the wrong thing very well: see [[Reward Hacking]].

## Mathematical Formulation

With uncertainty about outcomes, rational choice is expected-value maximisation:

$$a^{*} = \arg\max_{a \in \mathcal{A}} \; \mathbb{E}\big[\, U \mid a,\ e_{1:t},\ K \,\big]$$

where:
- $a^{*}$: the rational action at time $t$
- $\mathcal{A}$: the actions available to the agent
- $U$: the performance measure (or an internal utility function that agrees with it)
- $e_{1:t}$: the percept sequence up to time $t$
- $K$: the agent's built-in prior knowledge

The expectation is the point: rationality maximises **expected** performance, given what the agent could know. It is not judged by the outcome that actually happened.

## Key Properties / Variants

- **Rationality is different from omniscience.** An omniscient agent knows the actual outcome of its actions, which is impossible. Russell and Norvig's example: you cross an empty street and are flattened by a cargo door falling from an airliner. Crossing was still rational. Rationality maximises expected performance; perfection maximises actual performance.
- **Information gathering is part of rationality.** Crossing a busy road without looking is irrational, because looking (an action that changes future percepts) raises expected performance. Exploration of an unknown environment is the same idea: see [[Exploration vs Exploitation]].
- **Learning.** A rational agent should learn as much as possible from what it perceives. Agents that cannot adapt are fragile: the dung beetle keeps "plugging" its nest after the dung ball is taken away.
- **Autonomy.** An agent that leans on its designer's prior knowledge instead of its own percepts lacks autonomy. A rational agent should learn to compensate for partial or incorrect prior knowledge, though some built-in knowledge to start with is reasonable.
- **Perfect rationality is usually out of reach** in practice because of computational cost.
- **Rational is not the same as desirable in multi-agent settings.** Two rational players in the [[Prisoner's Dilemma]] both defect, although both prefer mutual cooperation. Individual rationality is captured by best responses and the [[Nash Equilibrium]]; desirability by [[Pareto Optimality]].
- **Utility-based agents** make rationality operational by internalising the performance measure as a utility function: see [[Agent Program]].

## Connections

- Defined for: [[Intelligent Agent]] in a [[Task Environment]] (P of PEAS is the performance measure)
- Implemented by: utility-based and learning designs in [[Agent Program]]
- Failure when the measure is wrong: [[Reward Hacking]]
- Multi-agent counterpart: [[Nash Equilibrium]], [[Prisoner's Dilemma]], [[Stag Hunt]]
- RL analogue: maximising expected [[Return]] under a [[Reward Signal]], which an [[Optimal Policy]] does

## Appears In

- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
- [[AGAI-Book Ch2 - Intelligent Agents]]
- [[AGAI-Lab01 - Simple Agents, Complex Dynamics]]
