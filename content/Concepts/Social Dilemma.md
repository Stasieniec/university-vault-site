---
type: concept
aliases: [social dilemma, social dilemmas, dilemma of cooperation, dilemma of coordination]
course: [AGAI]
tags: [game-theory, multi-agent, exam-topic]
status: complete
---

# Social Dilemma

## Definition

> [!definition] Social dilemma
> A **social dilemma** is a situation in which agents who each pursue their own interest reach an outcome that is worse for the group than another outcome they could have reached together. In game-theoretic terms: what the [[Nash Equilibrium]] predicts and what [[Pareto Optimality|Pareto optimality]] or welfare call desirable come apart.

## Intuition

> [!intuition] Individually sensible, collectively bad
> Each agent's choice is reasonable given what it expects from the others, and yet everyone ends up worse off. Each agent is already doing its best on its own terms, so a fix has to change the incentives (repetition, reputation, contracts) or the beliefs (communication, trust).

> [!example] Two kinds of dilemma
> | | Prisoner's Dilemma | Stag Hunt |
> |---|---|---|
> | Payoffs (CC, CD, DC, DD) | 3, 0, 5, 1 | 4, 0, 3, 2 |
> | Nash equilibria | (D, D) only | (C, C) and (D, D) |
> | Problem | cooperation: the only equilibrium is inefficient | coordination: the efficient outcome is an equilibrium, but so is a worse one |

## Mathematical Formulation

For a symmetric two-player game with actions C and D and row payoffs $R$ (C meets C), $S$ (C meets D), $T$ (D meets C), $P$ (D meets D), two motives can push agents to D:

$$\text{greed: } T > R, \qquad \text{fear: } P > S$$

where:
- greed: defecting pays more than cooperating when the partner cooperates
- fear: defecting pays more than cooperating when the partner defects

The [[Prisoner's Dilemma]] ($T > R > P > S$) has both, the [[Stag Hunt]] ($R > T > P > S$) only fear, and the Hawk-Dove or Chicken game ($T > R > S > P$) only greed. In all three, mutual cooperation $R$ beats mutual defection $P$.

**Donation game.** A common form of the Prisoner's Dilemma: cooperating costs $c$ and gives the partner $b$, with $b > c > 0$. Then $R = b - c$, $S = -c$, $T = b$, $P = 0$; D earns exactly $c$ more than C whatever the partner does.

## Key Properties / Variants

- **Equilibrium against welfare.** In the Prisoner's Dilemma the unique Nash equilibrium (D, D) is the only outcome that is not Pareto optimal; the social optimum is (C, C).
- **Populations.** Under the [[Replicator Dynamics]], the Prisoner's Dilemma goes to all-defect from any start, the Stag Hunt to all-cooperate or all-defect depending on whether cooperators start above the tipping point, and Hawk-Dove to a stable mix of both strategies.
- **Escapes.** Repetition with memory ([[Repeated Game]]: Tit-for-Tat, Win-Stay Lose-Shift), communication and reputation can make cooperation pay or make agents trust each other enough to coordinate.
- **For AI agents.** Self-interested agents owned by different parties (pricing bots, coding agents sharing a repository) can land in the same structures.

## Connections

- Games: [[Prisoner's Dilemma]], [[Stag Hunt]], [[Normal-Form Game]]
- Solution concepts in tension: [[Nash Equilibrium]], [[Pareto Optimality]]
- Dynamics and escapes: [[Replicator Dynamics]], [[Repeated Game]]
- Setting: [[Multi-Agent System]]

## Appears In

- [[AGAI-L02 - Game Theory and Population Dynamics]]
- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
- [[AGAI-Lab01 - Simple Agents, Complex Dynamics]]
