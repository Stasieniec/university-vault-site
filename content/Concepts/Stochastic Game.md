---
type: concept
aliases: [stochastic game, stochastic games, Markov game, Markov games, partially observable stochastic game, partially observable stochastic games, POSG, POSGs]
course: [AGAI]
tags: [game-theory, multi-agent, marl, exam-topic]
status: complete
---

# Stochastic Game

## Definition

> [!definition] Stochastic game
> A **stochastic game** introduces states into a multi-agent interaction. It is a 6-tuple $(I, S, \{A_i\}_{i \in I}, \{R_i\}_{i \in I}, T, \mu)$:
> - $I$: a finite set of agents
> - $S$: a finite set of states
> - $A_i$: the finite set of actions of agent $i$
> - $R_i$: the reward function of agent $i$
> - $T$: the transition function
> - $\mu$: the initial state distribution

> [!definition] Partially observable stochastic game (POSG)
> A stochastic game in which agents receive observations that give incomplete information about the state and about the other agents' actions. It is an 8-tuple $(I, S, \{A_i\}, \{R_i\}, T, \mu, \{O_i\}, \{\mathcal{O}_i\})$, adding for each agent:
> - $O_i$: a finite set of observations
> - $\mathcal{O}_i : A \times S \times O_i \to [0, 1]$: the observation function

## Intuition

> [!intuition] A game per state, and actions that move between them
> A [[Normal-Form Game|normal-form game]] is one table of rewards played once. A stochastic game is a set of such tables, one per state, linked by transitions: the joint action earns rewards now and also decides, with some probability, which state (and so which table) comes next. Agents therefore have to weigh immediate reward against where the game goes. A POSG removes the assumption that everyone sees the state: each agent gets its own, partial view, as a driver who cannot see behind a truck or a card player who cannot see the other hands.

> [!example] A three-state game
> Two agents with actions 1 and 2, states $s_1, s_2, s_3$, each with its own payoff matrix. In $s_1$, joint action (2, 2) pays (1, 1) and leads back to $s_1$ with probability 0.5 or to $s_2$ with probability 0.5. In $s_2$, (1, 2) pays (0, 4) and leads to $s_1$ with probability 0.8 or $s_3$ with 0.2. In $s_3$, (2, 1) pays (4, 1) and returns to $s_1$ with probability 1. So: 2 agents, 2 actions each, 3 states.

## Mathematical Formulation

The usual signatures (standard notation, outside the AGAI course material):

$$R_i : S \times A \times S \to \mathbb{R}, \qquad T : S \times A \times S \to [0, 1], \qquad \mu : S \to [0, 1]$$

where:
- $A = A_1 \times \dots \times A_n$: the joint action space
- $R_i(s, a, s')$: agent $i$'s reward when joint action $a$ in state $s$ leads to $s'$
- $T(s, a, s')$: the probability of moving to $s'$ from $s$ under joint action $a$
- $\mu(s)$: the probability that the game starts in $s$

In a POSG, $\mathcal{O}_i(a, s, o_i)$ is the probability that agent $i$ observes $o_i$ when joint action $a$ led to state $s$. Policies condition on the agent's own observation history:

$$\pi_i(a_i^t \mid h_i^t), \qquad h_i^t = (o_i^0, \dots, o_i^{t-1})$$

where $a_i^t$ is the action at step $t$. In a repeated normal-form game, by contrast, the history is the sequence of joint actions $(a^0, \dots, a^{t-1})$.

## Key Properties / Variants

- **Hierarchy of games** (AGAI, following Albrecht, Christianos and Schäfer): one-shot normal-form game, then repeated normal-form game ($n$ agents, 1 state), then stochastic game ($n$ agents, $m$ states, fully observable), then POSG ($n$ agents, $m$ states, partially observable). Each contains the previous one as a special case.
- **Special cases** (standard facts): a stochastic game with one agent is a [[Markov Decision Process]]; a POSG with one agent is a [[POMDP]]; a stochastic game with one state is a repeated normal-form game.
- **Examples:** the two-square vacuum world (one agent); level-based foraging, a grid world where agents collect apples, which becomes a POSG when each agent sees only the cells near it; autonomous driving; card games with private, hidden information.
- **Why it matters:** it is the standard model for multi-agent reinforcement learning, where each agent learns a [[Policy|policy]] while the others learn too.

## Connections

- Special case: [[Normal-Form Game]], [[Repeated Game]]
- Single-agent counterparts: [[Markov Decision Process]], [[POMDP]]
- Solution concepts: [[Best Response]], [[Nash Equilibrium]]
- Learning: [[Reinforcement Learning]]
- Setting: [[Multi-Agent System]]

## Appears In

- [[AGAI-L02 - Game Theory and Population Dynamics]]
