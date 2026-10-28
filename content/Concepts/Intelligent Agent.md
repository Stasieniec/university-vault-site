---
type: concept
aliases: [agent, agents, AI agent]
course: [AGAI]
tags: [agents]
status: complete
---

# Intelligent Agent

## Definition

> [!definition] Agent (Russell and Norvig)
> "An agent is anything that can be viewed as perceiving its environment through **sensors** and acting upon that environment through **actuators**."

> [!definition] Agent (Wooldridge)
> "An agent is a computer system that is situated in some environment, and that is capable of **autonomous action** in this environment to meet its design objectives."

The two definitions stress different things. Russell and Norvig's is about the perceive-act coupling with an environment; Wooldridge's adds autonomy and objectives, which is closer to how multi-agent systems research uses the word.

## Intuition

> [!intuition] A loop between a box and a world
> An agent sits in a loop: it **observes** the environment, decides, and **acts**, and the action changes what it observes next. A human has eyes and ears as sensors and hands and voice as actuators. A robot vacuum has a dirt sensor and wheels. A software agent reads files, network packets and keyboard input, and acts by writing files, sending packets or displaying text. An [[LLM Agent]] runs the same loop through natural language.

"Agent" is a way of analysing a system, and Russell and Norvig say so directly: the notion "is meant to be a tool for analyzing systems, not an absolute characterization that divides the world into agents and non-agents." A calculator can be described as an agent that outputs "4" after the percept sequence "2 + 2 =", but that description adds nothing to our understanding of the calculator. The agent view is worth using when the system's behaviour is best explained by what it is trying to achieve.

## Mathematical Formulation

The vocabulary that makes "agent" precise:

- A **percept** is what the sensors deliver at one instant.
- The **percept sequence** is the complete history of everything the agent has ever perceived.
- The **agent function** maps every possible percept sequence to an action:

$$f : \mathcal{P}^{*} \rightarrow \mathcal{A}$$

where:
- $\mathcal{P}$: the set of possible percepts
- $\mathcal{P}^{*}$: the set of all finite percept sequences
- $\mathcal{A}$: the set of actions the actuators can perform

The agent function is an abstract mathematical description of behaviour. The **agent program** is the concrete code that implements it, running on a physical **architecture** (a PC, a robotic car):

$$\text{agent} = \text{architecture} + \text{program}$$

The program receives only the current percept; if behaviour must depend on the history, the program has to remember it. See [[Agent Program]] for the designs this leads to.

> [!example] The vacuum world
> Two squares, A and B, each clean or dirty. Percepts are pairs like `[A, Dirty]`; actions are Left, Right and Suck. One agent function: if the current square is dirty, Suck; otherwise move to the other square. This is the running example for agents, rationality and [[Reward Hacking]].

## Key Properties / Variants

- **Behaviour is fixed by the agent function**: specifying the action for every possible percept sequence says more or less everything about an agent from the outside.
- **Choice depends only on what has been perceived** plus built-in knowledge, never on things the agent has not perceived.
- **Autonomy** (Wooldridge's emphasis, also in Russell and Norvig): an agent that relies on its designer's prior knowledge instead of its own percepts and learning lacks autonomy.
- **Quality of behaviour** is judged separately, by a performance measure: see [[Rational Agent]].
- **The world it lives in** is described by its [[Task Environment]] (PEAS and the environment properties).
- **More than one agent** in the same environment gives a [[Multi-Agent System]].
- **Levels of complexity**: the same observe-act description covers a reflex rule, a model-based planner, a learner, and an LLM with memory and tools.

## Connections

- Judged by: [[Rational Agent]] (performance measure, rationality)
- Lives in: [[Task Environment]]
- Implemented by: [[Agent Program]] (reflex, model-based, goal-based, utility-based, learning)
- Many of them: [[Multi-Agent System]]
- Modern form: [[LLM Agent]], [[ReAct]]
- In RL terms: the agent of a [[Markov Decision Process]], whose behaviour is a [[Policy]]; see [[Reinforcement Learning]]

## Appears In

- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
- [[AGAI-Book Ch2 - Intelligent Agents]]
- [[AGAI-Lab01 - Simple Agents, Complex Dynamics]]
