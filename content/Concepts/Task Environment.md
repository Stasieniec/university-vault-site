---
type: concept
aliases: [PEAS, task environments, environment properties]
course: [AGAI]
tags: [agents]
status: complete
---

# Task Environment

## Definition

> [!definition] Task environment
> The **task environment** is the "problem" to which a [[Rational Agent]] is the "solution". It is specified by its **PEAS** description: **P**erformance measure, **E**nvironment, **A**ctuators, **S**ensors. Specifying the task environment as fully as possible is the first step in designing any agent.

## Intuition

> [!intuition] Know the problem before building the solver
> The right design for an [[Agent Program]] depends on the world it has to work in. A crossword solver can be a simple program; an automated taxi cannot. PEAS forces you to write down what counts as success and what the agent can sense and do, and the environment properties tell you how hard the problem is.

> [!example] PEAS for an automated taxi
> - **Performance measure**: safe, fast, legal, comfortable trip; maximise profits; minimise impact on other road users. Some goals conflict, so trade-offs are needed.
> - **Environment**: roads, other traffic, police, pedestrians, customers, weather.
> - **Actuators**: steering, accelerator, brake, signal, horn, display, speech.
> - **Sensors**: cameras, radar, speedometer, GPS, engine sensors, accelerometer, microphones, touchscreen.

## Key Properties / Variants

Task environments vary along a small number of dimensions, which largely decide the appropriate agent design.

- **Fully vs partially observable.** Fully observable if the sensors give access to the complete state (or at least everything relevant to the choice of action). Then the agent needs no internal state. Partial observability comes from noisy sensors or missing information: a vacuum with only a local dirt sensor cannot see the other square. No sensors at all: **unobservable**. See [[Partial Observability]].
- **Single-agent vs multiagent.** Does another entity B have to be treated as an agent? Yes if B's behaviour is best described as maximising a performance measure that depends on agent A's behaviour. Chess is **competitive**; avoiding collisions in traffic makes driving **partially cooperative** (and parking makes it partly competitive). See [[Multi-Agent System]].
- **Deterministic vs nondeterministic.** Deterministic if the next state is completely determined by the current state and the agents' actions. **Stochastic** is used when the uncertainty is quantified with probabilities. A partially observable environment can look nondeterministic.
- **Episodic vs sequential.** Episodic: experience splits into independent episodes (classifying parts on an assembly line). Sequential: the current decision affects all future ones (chess, driving).
- **Static vs dynamic.** Dynamic if the environment can change while the agent deliberates (taxi driving). **Semidynamic** if the environment stays put but the score changes with time (chess with a clock).
- **Discrete vs continuous.** Applies to states, time, percepts and actions. Chess is discrete; driving is continuous.
- **Known vs unknown.** About the agent's (or designer's) knowledge of the "laws of physics" of the environment, so strictly not a property of the environment itself. This is a separate axis from observability: solitaire is known but partially observable; a new video game can be fully observable but unknown. Even the performance measure can be unknown (a passenger's preferred driving style).

The hardest case is partially observable, multiagent, nondeterministic, sequential, dynamic, continuous and unknown. Taxi driving is all of these except that the environment is mostly known.

| Task environment | Observable | Agents | Deterministic | Episodic | Static | Discrete |
|---|---|---|---|---|---|---|
| Crossword | Fully | Single | Deterministic | Sequential | Static | Discrete |
| Chess with a clock | Fully | Multi | Deterministic | Sequential | Semi | Discrete |
| Poker | Partially | Multi | Stochastic | Sequential | Static | Discrete |
| Taxi driving | Partially | Multi | Stochastic | Sequential | Dynamic | Continuous |
| Image analysis | Fully | Single | Deterministic | Episodic | Semi | Continuous |

The classifications are not always clear-cut. Medical diagnosis is episodic if the task is "pick a diagnosis from symptoms" and sequential if it includes ordering tests and following treatment.

> [!example] The vacuum world, classified
> In the basic version, two squares, the geography is known, and actions have certain effects: deterministic, discrete, static. With only a local dirt sensor it is partially observable. With dirt that reappears at random, as in AGAI Lab 1, it becomes stochastic. Adding a second robot would make it multiagent.

## Connections

- Defines the problem for: [[Intelligent Agent]], judged as a [[Rational Agent]] by the P in PEAS
- Shapes the design of: [[Agent Program]] (partial observability calls for internal state, unknown environments call for learning)
- Multiagent case: [[Multi-Agent System]]
- Formal versions: a fully observable stochastic sequential environment is a [[Markov Decision Process]]; partial observability gives a [[POMDP]]
- Known vs unknown relates to having a [[Model of the Environment]]

## Appears In

- [[AGAI-Book Ch2 - Intelligent Agents]]
- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
- [[AGAI-Lab01 - Simple Agents, Complex Dynamics]]
