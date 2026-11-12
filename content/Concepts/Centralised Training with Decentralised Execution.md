---
type: concept
aliases: [CTDE, Centralized Training with Decentralized Execution, centralised training with decentralised execution, centralized training with decentralized execution, Centralised training and execution, Centralized training and execution, CTCE, Decentralised training and execution, Decentralized training and execution, DTDE, training and execution modes]
course: [AGAI]
tags: [multi-agent, marl, deep-rl, exam-topic]
status: complete
---

# Centralised Training with Decentralised Execution

## Definition

> [!definition] Centralised training with decentralised execution (CTDE)
> A training and execution mode for multi-agent deep RL in which agents are **trained using centrally shared information** (for example the other agents' observations or actions, a global state, or a centralised critic), but at execution time **each agent acts independently using only its own local observations**.

It is one of three modes, set by whether each phase may use information about the other agents:

- **Centralised:** an agent uses centrally shared information about the others (joint history, observations, global state, joint policy).
- **Decentralised:** an agent conditions only on its own individual history $h_i^t$.

| Mode | Training uses | Execution uses |
|---|---|---|
| CTCE: centralised training and execution | shared information | shared information |
| DTDE: decentralised training and execution | own history | own history |
| CTDE: centralised training with decentralised execution | shared information | own history |

## Intuition

> [!intuition] Practise with the full picture, play with what you can see
> Training usually happens in a simulator, where everything is observable and can be logged. Execution happens in the world, where agents are separate machines that cannot share everything in real time. CTDE uses the simulator's full information to learn, then removes it: whatever each agent learned has to work from its own sensors alone. Autonomous cars that cannot exchange all sensor data while driving, but can be trained in a simulator that provides all data, are the standard case.

## Mathematical Formulation

The three modes differ in what the learned functions may condition on. With $h_i^t$ agent $i$'s observation history and $\mathbf{h}^t = (h_1^t, \dots, h_n^t)$ the joint history:

- **CTCE:** one joint policy $\pi(\mathbf{a} \mid \mathbf{h}^t)$ over the joint action $\mathbf{a} = (a_1, \dots, a_n)$, trained on a joint reward. The joint action space has $\prod_i |A_i|$ elements, exponential in the number of agents.
- **DTDE:** each agent has $\pi_i(a_i \mid h_i^t)$ (or $Q(h_i^t, a_i;\theta_i)$), trained on its own reward and history only, as in [[Independent Learning|independent learning]].
- **CTDE:** each actor is decentralised, $\pi_i(a_i \mid h_i^t;\phi_i)$, but its training signal may come from a function of centralised information, such as a **centralised critic** $V(\mathbf{h}^t;\theta_i)$ or $Q(\mathbf{h}^t, \mathbf{a};\theta_i)$. The critic is used only in training, so execution needs nothing beyond $h_i^t$.

## Key Properties / Variants

- **CTCE** (centralised learning in AGAI Lecture 3): needs a joint reward; does not scale with the number of agents (joint action space grows exponentially); impossible when agents are distributed entities by design.
- **DTDE** (independent learning, also agent modelling): uses no information about the other agents; suffers non-stationarity, coordinates poorly, learns unstably, can settle in suboptimal equilibria, and cannot tell changes caused by other agents from changes in the environment. Examples: [[Independent DQN|IDQN]], independent REINFORCE, IA2C, IPPO.
- **CTDE** advantages: no central controller at execution; suits distributed settings with limited runtime communication; training can account for the interactions between agents, improving coordination.
- **CTDE** disadvantages: centralised training can still be a bottleneck; coordination is hard when local information is not sufficient at execution.
- The standard CTDE pattern is an [[Actor-Critic|actor-critic]] with a centralised critic and decentralised actors (multi-agent A2C, MAPPO); value-based CTDE methods also exist.
- "Decentralised training with centralised execution" is not one of the modes: execution is the phase where information sharing is hardest.

## Connections

- Setting: [[Stochastic Game]], [[Multi-Agent System]]
- The decentralised extreme: [[Independent Learning]], [[Independent DQN]]
- Built on: [[Actor-Critic]], [[Advantage Actor-Critic (A2C)]], [[PPO]], [[Deep Reinforcement Learning]]
- Problems it targets: non-stationarity, [[Multi-Agent Credit Assignment]]
- Execution from local information: [[Partial Observability]]

## Appears In

- [[AGAI-L04 - Deep RL and Intro to Deep MARL]] (the three modes; CTCE and DTDE in detail)
- [[AGAI-L05 - Deep MARL, Part II]] (CTDE methods, its advantages and disadvantages)
