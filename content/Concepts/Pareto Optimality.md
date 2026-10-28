---
type: concept
aliases: [Pareto optimal, Pareto efficiency, Pareto optimum]
course: [AGAI]
tags: [game-theory, multi-agent, exam-topic]
status: complete
---

# Pareto Optimality

## Definition

> [!definition] Pareto optimum
> An outcome is **Pareto optimal** if no other outcome makes one player better off **without making another worse off**. It describes outcomes that are not wasteful.

> [!definition] Social optimum
> An outcome that **maximises the sum of the players' payoffs** (total welfare). Every social optimum is Pareto optimal, but not every Pareto optimum is a social optimum.

## Intuition

> [!intuition] No free improvements left on the table
> Outcome $B$ **Pareto-dominates** outcome $A$ if nobody does worse under $B$ and at least one player does strictly better. Moving from $A$ to $B$ is then a free improvement: someone gains and nobody can object. An outcome is Pareto optimal when no such free improvement exists. That is a weak requirement. It rules out waste, but it says nothing about fairness or total welfare: an outcome where one player takes everything can be Pareto optimal, because any change would hurt that player.

Pareto optimality asks what is **desirable** for the group. The [[Nash Equilibrium]] asks what self-interested players will **actually do**. Social dilemmas are exactly the games where the two disagree.

## Mathematical Formulation

Outcome $o'$ Pareto-dominates outcome $o$ when

$$u_i(o') \ge u_i(o) \;\; \text{for all players } i \quad \text{and} \quad u_j(o') > u_j(o) \;\; \text{for some player } j$$

An outcome is Pareto optimal if no outcome Pareto-dominates it. A social optimum solves

$$o^{\text{soc}} = \arg\max_{o} \sum_{i} u_i(o)$$

where:
- $o, o'$: outcomes (one action per player)
- $u_i(o)$: the payoff of player $i$ in outcome $o$

**Why a social optimum is always Pareto optimal:** if $o'$ Pareto-dominated the social optimum, it would have a strictly larger payoff sum, contradicting maximality.

> [!example] Prisoner's Dilemma ($R=3, S=0, T=5, P=1$)
> | Outcome | Payoffs | Sum | Pareto optimal? |
> |---|---|---|---|
> | (C,C) | (3, 3) | 6 | yes, and the social optimum |
> | (C,D) | (0, 5) | 5 | yes |
> | (D,C) | (5, 0) | 5 | yes |
> | (D,D) | (1, 1) | 2 | no, (C,C) is better for both |
>
> The only Nash equilibrium, (D,D), is the only outcome that is **not** Pareto optimal. (C,D) is Pareto optimal even though one player is exploited: moving away from it would cost the defector something. So Pareto optimal does not mean good.

> [!example] Stag Hunt ($R=4, S=0, T=3, P=2$)
> (C,C) gives (4, 4), which is at least as good for both players as every other outcome and strictly better for at least one. It Pareto-dominates all of them, so it is the only Pareto optimum and the social optimum. It is also a Nash equilibrium, but so is (D,D), which is Pareto-dominated.

## Key Properties / Variants

- **Usually several Pareto optima.** Pareto dominance is only a partial order (two outcomes can each be better for a different player); there is typically a whole set of optima (the Pareto frontier), and it does not pick one.
- **Equilibria can be Pareto-inefficient.** The Prisoner's Dilemma is the standard case: individual best responses lead to an outcome everyone would like to leave together.
- **Coordination failure.** In the [[Stag Hunt]] the Pareto optimum is an equilibrium, yet agents may still miss it for lack of trust.
- **Used beyond game theory** (background): the same idea defines efficiency in economics and the trade-off frontier in multi-objective optimisation.

## Connections

- Compared with: [[Nash Equilibrium]] (stability versus efficiency)
- Canonical games: [[Prisoner's Dilemma]], [[Stag Hunt]]
- Setting: [[Multi-Agent System]]
- Long-run route to the efficient outcome: [[Repeated Game]]

## Appears In

- [[AGAI-Lab01 - Simple Agents, Complex Dynamics]]
- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
