---
type: concept
aliases: [repeated games, iterated game, Tit-for-Tat, TFT, Win-Stay Lose-Shift, WSLS]
course: [AGAI]
tags: [game-theory, multi-agent, exam-topic]
status: complete
---

# Repeated Game

## Definition

> [!definition] Repeated game
> In a **repeated game** the same players play the same **stage game** (for example the [[Prisoner's Dilemma]]) for several rounds. A strategy is now a **rule that maps what happened before to an action**. The length of history the rule can use is its **memory**.

> [!definition] Tit-for-Tat (TFT)
> Cooperate in the first round; afterwards, **copy what the partner did in the previous round**.

> [!definition] Win-Stay, Lose-Shift (WSLS)
> Cooperate in the first round; afterwards, **repeat your previous action if it earned a "win"** (in the PD, $T$ or $R$, i.e. 5 or 3 points), **switch otherwise**.

## Intuition

> [!intuition] The shadow of the future
> In a one-shot Prisoner's Dilemma defection is the rational choice: there is no past to respond to and no future to protect. Once the game repeats and agents remember, what I do today shapes what my partner does tomorrow. A strategy that rewards cooperation and punishes defection makes exploiting it unprofitable over many rounds, so cooperation can pay even among self-interested agents.

The key ability is **conditioning on the partner**. Memory-zero strategies treat everyone the same: AllC cooperates with defectors and gets exploited. Memory-one strategies like TFT cooperate with cooperators and defect against defectors, so their cooperation rate depends on who they meet. That is what lets cooperation pay even when defectors are around.

## Mathematical Formulation

A **memory-one** strategy in the PD can be written as **five probabilities of cooperating**: in the first round, and after each previous outcome (my last action, partner's last action):

| Strategy | first | after CC | after CD | after DC | after DD |
|---|---|---|---|---|---|
| AllC | 1 | 1 | 1 | 1 | 1 |
| AllD | 0 | 0 | 0 | 0 | 0 |
| Random | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| TFT | 1 | 1 | 0 | 1 | 0 |
| WSLS | 1 | 1 | 0 | 0 | 1 |

Memory-zero strategies (AllC, AllD, Random) are the special case where the last four numbers are equal. Reading WSLS: after CC I got 3, a win, so stay with C; after CD I got 0, a loss, so shift to D; after DC I got 5, a win, so stay with D; after DD I got 1, a loss, so shift to C.

A tournament score is the average payoff per round against every strategy in the field, including a copy of itself:

$$\text{score}(i) = \frac{1}{|\mathcal{S}|} \sum_{j \in \mathcal{S}} \frac{1}{n} \sum_{t=1}^{n} u\big(a_i^{(t)}, a_j^{(t)}\big)$$

where:
- $\mathcal{S}$: the set of strategies in the tournament
- $n$: number of rounds per match
- $a_i^{(t)}$: the action strategy $i$ plays in round $t$
- $u(\cdot, \cdot)$: the stage-game payoff of the first player

## Key Properties / Variants

- **Axelrod's tournament (1980).** Robert Axelrod invited game theorists to submit strategies for a round-robin PD tournament of 200 rounds per match. Tit-for-Tat, submitted by Anatol Rapoport, won.
- **Length matters.** In a one-round tournament AllD comes out on top. As the number of rounds grows, AllD's one-off gain from exploiting a reciprocator in round 1 is outweighed by mutual defection afterwards, while reciprocators earn $R$ with each other every round, and AllD falls behind.
- **AllD profits from naive cooperators.** In a field containing AllC and Random, much of AllD's score is the exploitation payoff $T = 5$ taken from them; its success depends on who else is in the population.
- **TFT's properties** (Axelrod's analysis, background beyond the course material): it is "nice" (never defects first), retaliatory, forgiving (returns to cooperation as soon as the partner does), and clear.
- **Noise breaks TFT.** With execution errors (an agent sometimes plays the opposite of what it intended), two TFT players get stuck echoing each other's defections: one accidental D starts an alternating C/D pattern that lasts until another error breaks it.
- **WSLS corrects errors.** Two WSLS players recover from a single error: after the mistake (D,C) the defector "won" and stays D while the victim "lost" and shifts to D; the resulting (D,D) is a loss for both, so both shift back to C. WSLS has a weakness of its own: against AllD it alternates between cooperating and defecting, so it is exploited every other round.
- **Memory as agent design.** These strategies are simple reflex rules over a one-round memory (see [[Agent Program]]); the point is that very simple agents produce rich dynamics.

## Connections

- Stage game: [[Prisoner's Dilemma]] (also applies to the [[Stag Hunt]])
- Changes which outcomes are sustainable compared with the one-shot [[Nash Equilibrium]]; reaches the [[Pareto Optimality|Pareto-optimal]] (C,C)
- Population version: [[Replicator Dynamics]] over strategies such as AllD, TFT and WSLS
- Setting: [[Multi-Agent System]]

## Appears In

- [[AGAI-Lab01 - Simple Agents, Complex Dynamics]]
