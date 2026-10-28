---
type: concept
aliases: [MAS, multi-agent systems, multiagent system, multiagent systems]
course: [AGAI]
tags: [agents, multi-agent]
status: complete
---

# Multi-Agent System

## Definition

> [!definition] Multi-agent system (MAS)
> A **multi-agent system** is a system in which **more than one agent inhabits the environment**. Each agent observes and acts, and the agents may communicate, cooperate, come into conflict, or deceive one another.

Russell and Norvig give the test for when another entity has to be treated as an agent: entity B is an agent from agent A's point of view if B's behaviour is best described as **maximising a performance measure whose value depends on A's behaviour**. Another car in traffic qualifies; leaves blowing in the wind do not.

## Intuition

> [!intuition] My best action depends on yours
> With one agent, the environment is something to model and exploit. With several, part of the environment is other agents who are also choosing, adapting and possibly modelling you. What is best for one agent now depends on what the others do, so new questions appear: will they cooperate or cheat, coordinate or miscoordinate, tell the truth or lie? Examples: RoboCup robot football teams, several robot vacuums in one house, social media bots, autonomous vehicles, stock-trading bots, and LLM agents negotiating with each other.

## Key Properties / Variants

- **Competitive vs cooperative.** In chess, B maximising its measure minimises A's: competitive. In traffic, avoiding collisions is good for everyone: cooperative, though only one car fits in a parking space, so it is partially competitive too. Most interesting settings are mixed.
- **Different design problems.** Communication often emerges as rational behaviour in multiagent environments, and in some competitive ones **randomised behaviour is rational** because predictability can be exploited. In single-agent settings randomising is usually not rational (beyond helping a simple reflex agent escape a loop).
- **Dilemmas of cooperation.** Self-interest can destroy cooperation even when everyone would gain from it: the [[Prisoner's Dilemma]].
- **Dilemmas of coordination.** Even when cooperating is an equilibrium, agents may fail to reach it for lack of trust: the [[Stag Hunt]].
- **Tools for analysis.** Game theory and social choice, which come from mathematics, economics and philosophy, supply the solution concepts: [[Nash Equilibrium]] (what self-interested agents can settle on) and [[Pareto Optimality]] (which outcomes are not wasteful).
- **Populations.** With many simple adaptive agents the question becomes which strategies spread over time: [[Replicator Dynamics]]. Repeated interaction and memory change what is sustainable: [[Repeated Game]].
- **Learning is harder.** Each learner's environment includes the other learners, which keep changing their behaviour, so the environment is **non-stationary** from each agent's point of view. This is a central difficulty of multi-agent reinforcement learning.
- **Payoffs and communication.** Whether agents deceive each other (or humans) depends on how rewards and payoffs are defined, and defining them is not trivial (see [[Reward Hacking]]).

> [!example] LLM-based multi-agent systems
> Several [[LLM Agent|LLM agents]] can be wired together: orchestration patterns that assign roles and route messages, collaborative task-solving, communication protocols between agents (such as A2A), agent societies, and simulations of human behaviour such as the "generative agents" of Park et al. (2023). The risks studied for advanced multi-agent AI fall into three kinds: **miscoordination**, **conflict** and **collusion** (Hammond et al., 2025).

## Connections

- Built from: [[Intelligent Agent]]; the "agents" dimension of a [[Task Environment]]
- Strategic analysis: [[Nash Equilibrium]], [[Pareto Optimality]], [[Prisoner's Dilemma]], [[Stag Hunt]], [[Repeated Game]]
- Population analysis: [[Replicator Dynamics]]
- Modern systems: [[LLM Agent]], [[Agentic Search]]
- Learning: [[Reinforcement Learning]] with several agents (non-stationary environment)

## Appears In

- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
- [[AGAI-Book Ch2 - Intelligent Agents]]
- [[AGAI-Lab01 - Simple Agents, Complex Dynamics]]
