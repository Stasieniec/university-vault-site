---
type: concept
aliases: [Nash equilibria, pure Nash equilibrium, mixed Nash equilibrium]
course: [AGAI]
tags: [game-theory, multi-agent, exam-topic]
status: complete
---

# Nash Equilibrium

## Definition

> [!definition] Best response
> A **best response** is the action that maximises my payoff, **given** the action of my partner.

> [!definition] Nash equilibrium (pure)
> An outcome in which **each player's action is a best response to the other's**. No player can gain by changing its action **alone**. It describes what self-interested players can end up doing.

> [!definition] Mixed Nash equilibrium
> Players choose probability distributions over their actions. A profile of such mixed strategies is a Nash equilibrium if no player can raise its **expected** payoff by changing its own distribution while the others keep theirs.

## Intuition

> [!intuition] Nobody regrets their own choice
> Picture the players after the game, each told what the others did. If every one of them would say "given that, I would choose the same again", the outcome is a Nash equilibrium. It is a stability property: no one has a reason to deviate unilaterally. It says nothing about whether the outcome is good for the group, which is why the [[Prisoner's Dilemma]] equilibrium (both defect) is worse for both players than mutual cooperation.

> [!example] Finding pure equilibria in a 2x2 game
> For each of the four outcomes, check: is the row action a best response to the column action, and the column action a best response to the row action? In the Prisoner's Dilemma ($R=3, S=0, T=5, P=1$), defecting is the best response to both C ($5 > 3$) and D ($1 > 0$), so (D,D) is the only pure equilibrium. In the [[Stag Hunt]] ($R=4, S=0, T=3, P=2$), C is the best response to C ($4 > 3$) and D to D ($2 > 0$), so (C,C) and (D,D) are both equilibria.

## Mathematical Formulation

For a two-player game with payoff functions $u_1, u_2$, an action pair $(a_1^{*}, a_2^{*})$ is a pure Nash equilibrium if

$$u_1(a_1^{*}, a_2^{*}) \ge u_1(a_1, a_2^{*}) \;\; \forall a_1, \qquad u_2(a_1^{*}, a_2^{*}) \ge u_2(a_1^{*}, a_2) \;\; \forall a_2$$

where:
- $a_i$: an action of player $i$; $a_i^{*}$ the equilibrium action
- $u_i(a_1, a_2)$: player $i$'s payoff when the actions are $a_1, a_2$

**Mixed equilibrium in a symmetric 2x2 game.** Let the partner cooperate with probability $p$. My expected payoffs are

$$\pi_C(p) = pR + (1-p)S, \qquad \pi_D(p) = pT + (1-p)P$$

where $R, S, T, P$ are the payoffs for (C vs C), (C vs D), (D vs C), (D vs D). In a fully mixed equilibrium each player must be **indifferent** between C and D (otherwise it would play only the better one), so $\pi_C(p) = \pi_D(p)$, giving

$$p^{*} = \frac{P - S}{(R - T) + (P - S)}$$

For the Stag Hunt above, $p^{*} = 2/(1+2) = 2/3$: each hunter goes for the stag with probability $2/3$. The same $p^{*}$ is the belief threshold above which hunting stag is the best response, and the unstable interior rest point of the [[Replicator Dynamics]].

## Key Properties / Variants

- **Existence** (Nash, 1950; standard result beyond the course material): every finite game has at least one Nash equilibrium, possibly in mixed strategies. Pure equilibria need not exist (matching pennies has none).
- **Not necessarily efficient.** An equilibrium can be Pareto-dominated by another outcome: (D,D) in the Prisoner's Dilemma is worse for both than (C,C). Compare [[Pareto Optimality]].
- **Can be multiple.** The Stag Hunt has two pure equilibria and one mixed one. "Play a Nash equilibrium" does not say which will be reached; that is the **equilibrium selection** problem, where beliefs, risk and dynamics decide.
- **Dominant strategies.** If an action earns strictly more than every alternative whatever the other does (D in the Prisoner's Dilemma), every equilibrium uses it.
- **Stability under adaptation.** In the [[Replicator Dynamics]] of the Prisoner's Dilemma and the Stag Hunt, the stable rest points are exactly the strict pure equilibria, while the Stag Hunt's mixed equilibrium ($x = 2/3$) is an unstable rest point. An adaptive population therefore ends up at some equilibria and never at others. (Background beyond the course material: in general every stable rest point of the replicator dynamics is a Nash equilibrium and every strict equilibrium is stable, but a mixed equilibrium can be stable in other games, such as Hawk-Dove.)
- **Repeated play changes the equilibria.** When the same game is repeated with memory, strategies like Tit-for-Tat can sustain cooperation: see [[Repeated Game]].

## Connections

- Social-welfare counterparts: [[Pareto Optimality]] (and the social optimum, which maximises total payoff)
- Canonical games: [[Prisoner's Dilemma]], [[Stag Hunt]]
- Dynamics: [[Replicator Dynamics]], [[Repeated Game]]
- Setting: [[Multi-Agent System]] of [[Rational Agent|rational agents]]

## Appears In

- [[AGAI-Lab01 - Simple Agents, Complex Dynamics]]
- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
