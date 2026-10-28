---
type: concept
aliases: [PD, prisoners dilemma]
course: [AGAI]
tags: [game-theory, multi-agent, exam-topic]
status: complete
---

# Prisoner's Dilemma

## Definition

> [!definition] Prisoner's Dilemma (PD)
> A symmetric two-player game in which each player chooses **Cooperate (C)** or **Defect (D)**, with row-player payoffs $R$ (both cooperate), $S$ (I cooperate, partner defects), $T$ (I defect, partner cooperates) and $P$ (both defect), satisfying
> $$T > R > P > S$$

The names: **R**eward for mutual cooperation, **S**ucker's payoff, **T**emptation to defect, **P**unishment for mutual defection.

## Intuition

> [!intuition] Self-interest destroys cooperation
> Whatever my partner does, I do better by defecting: against a cooperator I get $T$ instead of $R$, against a defector $P$ instead of $S$. My partner reasons the same way. So we both defect and get $P$ each, although we both prefer $R$. Each player's reasoning is individually sound; the collective result is bad. That gap between individual rationality and collective welfare is what makes it a **dilemma of cooperation**.

Two motives push towards defection:
- **Greed**: $T > R$, defecting pays more when the partner cooperates.
- **Fear**: $P > S$, defecting pays more when the partner defects.

The PD has **both**, which makes D a dominant strategy. Its problem is one of **incentives**: even an agent certain its partner will cooperate is tempted to defect. Compare the [[Stag Hunt]], which has fear but no greed, and where the problem is trust.

> [!example] Two pricing bots
> | | partner C | partner D |
> |---|---|---|
> | **C** | 3, 3 | 0, 5 |
> | **D** | 5, 0 | 1, 1 |
>
> Two autonomous agents owned by different companies (pricing bots in one market, scheduling agents sharing a cluster) each maximise their owner's payoff. Game theory predicts (D,D), with payoff 1 each, where (C,C) would have given 3 each.

## Mathematical Formulation

Against a partner who cooperates with probability $p$ (or, in a population, a fraction $x$ of cooperators):

$$f_C - f_D = p(R - T) + (1 - p)(S - P)$$

where:
- $f_C = pR + (1-p)S$: expected payoff of cooperating
- $f_D = pT + (1-p)P$: expected payoff of defecting
- $R - T < 0$ by greed, and $S - P < 0$ by fear

Both terms are negative, so $f_C - f_D < 0$ for every $p$: defection always earns more. In the [[Replicator Dynamics]] this means the fraction of cooperators falls from any starting point. $x = 1$ (all cooperate) is a rest point but unstable, since a single defector earns more and spreads; $x = 0$ (all defect) is the stable rest point and corresponds to the Nash equilibrium (D,D).

## Key Properties / Variants

- **Unique Nash equilibrium (D,D)**, in dominant strategies. See [[Nash Equilibrium]].
- **(D,D) is the only outcome that is not Pareto optimal**; (C,C) is the social optimum. See [[Pareto Optimality]].
- **One-shot versus repeated.** Played once, defection is the rational choice. Played repeatedly with memory, reciprocal strategies such as Tit-for-Tat and Win-Stay Lose-Shift can make cooperation pay, and Axelrod's 1980 tournament was won by Tit-for-Tat. See [[Repeated Game]].
- **Extra condition for repeated play** (background beyond the course material): many definitions also require $2R > T + S$, so that taking turns exploiting each other pays less than steady mutual cooperation.
- **Populations.** Without memory or reciprocity, adaptive populations (imitation, simple reinforcement learning) drift to all-defect, matching the rational prediction.
- **Why it matters for AI agents.** Self-interested agents deployed by different owners can end up in PD-like situations: competing bots, negotiating LLM agents, shared resources.

## Connections

- Solution concepts: [[Nash Equilibrium]], [[Pareto Optimality]]
- Contrast: [[Stag Hunt]] (fear without greed; coordination problem)
- Escapes from the dilemma: [[Repeated Game]] (memory, reciprocity)
- Population view: [[Replicator Dynamics]]
- Setting: [[Multi-Agent System]], [[Rational Agent]]

## Appears In

- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
- [[AGAI-Lab01 - Simple Agents, Complex Dynamics]]
