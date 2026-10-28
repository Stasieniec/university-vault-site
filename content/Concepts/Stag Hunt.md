---
type: concept
aliases: [stag hunt game]
course: [AGAI]
tags: [game-theory, multi-agent, exam-topic]
status: complete
---

# Stag Hunt

## Definition

> [!definition] Stag Hunt (SH)
> A symmetric two-player game with actions **C** (hunt the stag) and **D** (hunt a hare), using the same payoff names as the [[Prisoner's Dilemma]] ($R$, $S$, $T$, $P$), but ordered
> $$R > T > P > S$$

A stag is worth a lot but can only be caught if **both** hunters go for it. A hare can be caught alone, and is worth a little less if the other hunter also chases hares in the same field.

## Intuition

> [!intuition] Cooperation is safe only if you trust your partner
> Mutual cooperation is the best outcome for both, and it is stable: if my partner hunts the stag, so should I ($R > T$). But hunting stag alone leaves me with nothing ($S$), while a hare is guaranteed. So whether I should cooperate comes down to whether I believe the other will. The Stag Hunt is a **dilemma of coordination**, or of **trust**.

Of the two motives for defecting, the Stag Hunt has **fear** ($P > S$) but **no greed** ($T < R$). That is the contrast with the Prisoner's Dilemma, which has both: there the problem is incentives; here it is trust.

> [!example] Two coding agents
> | | partner C | partner D |
> |---|---|---|
> | **C** | 4, 4 | 0, 3 |
> | **D** | 3, 0 | 2, 2 |
>
> Two coding agents can coordinate changes to shared files to implement a big feature (C), or each work on an independent, low-impact task for a smaller but safe reward (D). The feature only lands if both coordinate.

## Mathematical Formulation

An agent who believes its partner cooperates with probability $p$ compares

$$\pi_C(p) = pR + (1-p)S, \qquad \pi_D(p) = pT + (1-p)P$$

Cooperating is the best response when $\pi_C(p) > \pi_D(p)$, that is, when the belief exceeds the threshold

$$p^{*} = \frac{P - S}{(R - T) + (P - S)}$$

where:
- $P - S$: what you lose by cooperating when your partner does not (the fear motive)
- $R - T$: what you gain by cooperating when your partner also does
- $p^{*}$: how much trust is needed before hunting the stag is worthwhile; larger $p^{*}$ means more trust required

For $R=4, S=0, T=3, P=2$: $p^{*} = 2/(1 + 2) = 2/3$.

**Which equilibrium is played more often?** With beliefs spread uniformly between 0 and 1, an agent cooperates with probability $1 - p^{*}$. So (C,C) is the more frequent outcome when $p^{*} < 1/2$, which is when $R - T > P - S$: the gain from successful cooperation outweighs the loss from being let down. With the numbers above, $R - T = 1 < P - S = 2$, so (D,D) wins: (C,C) occurs with probability $(1/3)^2 \approx 0.11$, (D,D) with $(2/3)^2 \approx 0.44$, and the rest is miscoordination. Making the hare more attractive (raising $T$) shrinks $R - T$, raises $p^{*}$ and makes (C,C) rarer still. (Background beyond the course material: game theorists call the equilibrium favoured by this comparison the **risk-dominant** one.)

## Key Properties / Variants

- **Three Nash equilibria**: (C,C) and (D,D) in pure strategies, plus a mixed one where each cooperates with probability $p^{*}$. See [[Nash Equilibrium]].
- **(C,C) is the only Pareto optimum** and the social optimum; it Pareto-dominates (D,D). See [[Pareto Optimality]].
- **"Rational agents will coordinate on (C,C) because it is better for both" is false.** Being better for both does not make an outcome the one reached; a rational agent cooperates only if its belief exceeds $p^{*}$.
- **Population dynamics.** In the [[Replicator Dynamics]], $x = 0$ and $x = 1$ are stable and the interior rest point $x = p^{*}$ is unstable. A population with more than $p^{*}$ cooperators goes to all-cooperate; with fewer, to all-defect. With $p^{*} = 2/3$, a population starting at 60% cooperators still collapses to all-defect, even though most agents cooperate at the start.
- **Same threshold in both models.** In a randomly mixed population the chance that your partner cooperates is the fraction of cooperators $x$, which plays the role of the belief $p$. So the rational threshold and the evolutionary tipping point coincide.
- **What changes the outcome**: communication, reputation or memory can raise beliefs above $p^{*}$ and let agents coordinate on the stag.

## Connections

- Contrast: [[Prisoner's Dilemma]] (greed and fear versus fear only)
- Solution concepts: [[Nash Equilibrium]], [[Pareto Optimality]]
- Population view: [[Replicator Dynamics]]
- Setting: [[Multi-Agent System]]

## Appears In

- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
- [[AGAI-Lab01 - Simple Agents, Complex Dynamics]]
