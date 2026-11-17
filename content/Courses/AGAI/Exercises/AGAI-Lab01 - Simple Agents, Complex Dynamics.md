---
type: exercise
course: 5204AGAI6Y
week: 1
source: "Lab 1 notebook (werkcollege, 28 Oct 2026)"
concepts:
  - "[[Nash Equilibrium]]"
  - "[[Pareto Optimality]]"
  - "[[Prisoner's Dilemma]]"
  - "[[Stag Hunt]]"
  - "[[Repeated Game]]"
  - "[[Replicator Dynamics]]"
  - "[[Reward Hacking]]"
  - "[[Value Iteration]]"
  - "[[Markov Decision Process]]"
  - "[[Agent Program]]"
status: complete
---

# AGAI-Lab01: Simple Agents, Complex Dynamics

> [!abstract] In short
> In a one-shot [[Prisoner's Dilemma]], greed and fear both push to D, so self-interested agents defect. In the [[Stag Hunt]] only fear does, so the outcome depends on what each agent believes its partner will do. In repeated games, memory lets cooperation pay. Under the [[Replicator Dynamics|replicator equation]], a Prisoner's Dilemma population always ends all-D, and a Stag Hunt population tips at 2/3 cooperators, the same threshold a rational agent's belief needs.
>
> Feeds mock Q6 (Lab Q2 is the same exercise; the mock version with its key is in [[AGAI-L02 - Game Theory and Population Dynamics|Lecture 2]]) and Q4 and Q5. The belief threshold returns in the open question on Stag Hunt Q-learners (mock Q12). Part III (the vacuum world and reward hacking) and the signalling game at the end of Part II are not midterm material and are left out.

**Concepts tested:** [[Nash Equilibrium]], [[Pareto Optimality]], [[Prisoner's Dilemma]], [[Stag Hunt]], [[Repeated Game]], [[Replicator Dynamics]]

## Part I: one-shot and repeated games

### The model

Every game is a **symmetric two-player, two-action game**: each player picks C or D at the same time, and four numbers give the row player's payoff (the column player is paid the same way):

| | partner plays C | partner plays D |
|---|---|---|
| **play C** | $R$ (reward) | $S$ (sucker) |
| **play D** | $T$ (temptation) | $P$ (punishment) |

- **Prisoner's Dilemma (PD):** $T > R > P > S$; the lab uses $R = 3, S = 0, T = 5, P = 1$.
- **Stag Hunt (SH):** $R > T > P > S$; the lab uses $R = 4, S = 0, T = 3, P = 2$. A stag (C) is caught only if both hunt it; a hare (D) can be caught alone.

Best response, Nash equilibrium, Pareto optimality and the social optimum are defined in [[AGAI-L02 - Game Theory and Population Dynamics|Lecture 2]]. With four cells, all are found by checking each cell.

### Nash, Pareto and social optimum in the PD (Q2)

| | col C | col D |
|---|---|---|
| **row C** | (3, 3), total 6 | (0, 5), total 5 |
| **row D** | (5, 0), total 5 | (1, 1), total 2 |

**(a) Is the Nash equilibrium (D, D) Pareto optimal, or the social optimum?** Neither. (C, C) gives both players 3 instead of 1, so (D, D) is Pareto dominated, and its total (2) is the lowest of all four cells.

**(b) Why do self-interested agents end at (D, D) although both prefer (C, C)?** The row player's best response is D whatever the column player does (5 > 3 and 1 > 0): it does not depend on the partner at all. The same holds for the column player. At (C, C) each player gains by switching alone, so preferring (C, C) jointly does not help.

**(c) (C, D) is Pareto optimal. Is it a good outcome?** No. It is Pareto optimal only because any move away from it hurts the defector, who has the top payoff 5. It is unequal (0 against 5) and wastes welfare (5 against 6). Pareto optimality only rules out gains that cost nobody anything; the social optimum (largest sum) is much stronger: of the three Pareto-optimal cells, only (C, C) qualifies.

### Greed and fear (Q3)

Two motives can push a player towards D:

| motive | condition | PD (3, 0, 5, 1) | SH (4, 0, 3, 2) |
|---|---|---|---|
| **greed**: D pays more against a cooperator | $T > R$ | yes | no |
| **fear**: D pays more against a defector | $P > S$ | yes | yes |

In the PD both motives point to D, so D is dominant: even an agent sure that its partner cooperates defects. More trust cannot fix that; the payoffs must change (punishment, contracts, repetition). The PD is an **incentive** problem with one pure equilibrium, (D, D).

In the SH there is no greed: an agent sure that its partner hunts stag hunts stag too, so (C, C) is an equilibrium, and so is (D, D). Only fear, doubt about the partner, pushes to D. The SH is a **trust and coordination** problem.

### Beliefs decide the Stag Hunt (Q4, Q5)

An agent holds a belief $p$, the probability that its partner cooperates. Its expected payoffs are $\pi_C(p) = pR + (1-p)S$ and $\pi_D(p) = pT + (1-p)P$, and C is the best response when $p$ exceeds

$$p^* = \frac{P - S}{(R - T) + (P - S)}$$

where $P - S$ is what you lose by cooperating with a defector (fear) and $R - T$ is what you gain by cooperating with a cooperator. For the lab's SH, $p^* = 2/3$: C needs a belief above two thirds.

**With no information** (each agent's belief drawn uniformly from 0 to 1), only a third of agents hunt stag, so (C, C) happens in only about 1 encounter in 9, the rarest outcome. Raising $T$ from 3 to 4 makes $p^* = 1$, and nobody hunts stag.

**Which equilibrium is played more often:** (C, C) wins when $R - T > P - S$, the gain from cooperating with a cooperator beats the loss from cooperating with a defector. In the lab's SH fear (2) beats gain (1), so (D, D) wins.

**The flaw in "rational agents always coordinate on (C, C), because it is better for both":** each agent chooses alone and best-responds to its own belief. (C, C) being better for both does not make C the best response unless the agent believes its partner plays C with probability above $p^*$. Anything that raises shared beliefs changes the outcome: communication ("I will hunt stag"), repeated meetings that build up experience, or reputation.

### Repeated games: memory lets cooperation pay (Q6, Q7)

In a [[Repeated Game|repeated game]] the same two agents play the PD for several rounds, in a round-robin tournament where everyone meets everyone. **Memory-0** strategies (AllC, AllD, Random) ignore the history; **memory-1** strategies look at the last round. **TFT** copies the partner's last action. **WSLS** repeats its action after a win (3 or 5) and switches after a loss (0 or 1).

- **Conditioning is what makes cooperation pay against a defector.** AllC is exploited every round. TFT is exploited only in round 1, then defects back, while still earning 3 per round with cooperators. WSLS is weaker: against AllD, mutual defection (1) counts as a loss, so it switches back to C and is exploited every other round.
- **AllD can still win** the 50-round tournament of AllC, AllD, Random, TFT and WSLS. Its score comes from exploitable partners: AllC hands it 5 every round, Random and WSLS about half the time. Against TFT it gets barely more than mutual defection.
- **Without exploitable partners** (only AllD, TFT and WSLS), TFT beats AllD from 4 rounds on: AllD's edge is a one-off first-round gain, while TFT and WSLS earn 3 per round from each other.

So rankings depend on the field and on the number of rounds.

## Part II: replicator dynamics

### The replicator equation

In Part I agents calculated. Here they adapt: a strategy that earns more than average spreads, because agents copy successful peers or successful agents are deployed more. In a large population with a fraction $x$ of cooperators, paired at random, a partner cooperates with probability $x$:

$$f_C(x) = xR + (1-x)S, \qquad f_D(x) = xT + (1-x)P, \qquad \dot x = x\big(f_C - \bar f\big) = x(1-x)(f_C - f_D)$$

The equation describes imitation of successful peers and a simple reinforcement learner (Cross learning) with a small learning rate. Rest points and stability are defined in [[AGAI-L02 - Game Theory and Population Dynamics|Lecture 2]].

### The Prisoner's Dilemma population (Q9)

**(a) Factorisation.** $f_C - \bar f = f_C - x f_C - (1-x) f_D = (1-x)(f_C - f_D)$; multiply by $x$.

**(b) Why C always shrinks.**

$$f_C(x) - f_D(x) = x(R - T) + (1-x)(S - P)$$

Greed makes $R - T$ negative and fear makes $S - P$ negative, so for every $x$ this is an average of two negative numbers. $\dot x < 0$ everywhere between 0 and 1: a population starting with 90% cooperators still ends at 0.

**(c) Rest points.** $x = 0$ (all D) is stable and matches the unique Nash equilibrium (D, D). $x = 1$ (all C) is a rest point only because there is nobody to copy: a single defector earns $T = 5$ against the cooperators' $R = 3$, so it spreads. It is unstable.

### The Stag Hunt population (Q10)

**(a)** $f_C - f_D = x(4 - 3) + (1-x)(0 - 2) = 3x - 2$: negative below 2/3, zero at 2/3, positive above.

**(b)** Rest points: $x = 0$ (all hare, stable), $x = 2/3$ (unstable), $x = 1$ (all stag, stable). The interior point is the mixed Nash equilibrium, and an adapting population does not end there: slightly more stag hunters make stag pay more, which attracts more, and the reverse below. It is the tipping point between the two outcomes.

**(c) Why the tipping point equals $p^*$.** Under random matching, the probability that my partner cooperates is the population share $x$. So $f_C(x)$ and $f_D(x)$ are Part I's expected payoffs with the share in place of the belief, and the rest point solves the same equation as the rational agent's indifference: $x^* = p^* = 2/3$.

**(d) "If the majority cooperates, cooperation must win."** False: from 60% stag hunters the population falls to 0. At $x = 0.6$ a stag hunter earns $4 \cdot 0.6 = 2.4$ and a hare hunter $3 \cdot 0.6 + 2 \cdot 0.4 = 2.6$. What spreads is what earns more given the current mix; "everyone is better off if everyone cooperates" compares whole outcomes, which no adapting agent does. The bar is 2/3 instead of 1/2 because fear ($P - S = 2$) outweighs gain ($R - T = 1$).

## What to be able to answer

- Pure Nash equilibria, Pareto optima and the social optimum of a symmetric 2x2 game, and why the PD's equilibrium is inefficient.
- Greed ($T > R$) and fear ($P > S$), which games have which, and incentive against trust problems.
- The belief threshold $p^*$ and the rule $R - T > P - S$.
- Why memory-1 strategies make cooperation pay, and why AllD's success depends on the field.
- The PD and SH replicator outcomes, and why the SH tipping point equals $p^*$.

## Exam questions

> [!exam]- A symmetric game has R = 5, S = 0, T = 3, P = 1. (a) Which motives (greed, fear) does it have, and what kind of game is it? (b) Find the pure Nash equilibria and the social optimum. (c) Above which belief that the partner cooperates is C the best response? (d) What happens under the replicator dynamics from 50% cooperators?
> **Key points:** no greed (3 < 5), fear (1 > 0): a Stag Hunt; pure NE (C, C) and (D, D), social optimum (C, C) with total 10; $p^* = 1/3$; from 50%, above the tipping point 1/3, so cooperators grow to all C.
>
> **(a)** $T = 3 < R = 5$: no greed. $P = 1 > S = 0$: fear. $R > T > P > S$, so it is a Stag Hunt, a trust problem.
>
> **(b)** At (C, C) deviating gives 3 < 5; at (D, D) deviating gives 0 < 1; at (C, D) the cooperator gains by switching (1 > 0). Pure equilibria: (C, C) and (D, D). Totals 10, 3, 3, 2, so (C, C) is the social optimum (and the only Pareto-optimal cell).
>
> **(c)** $p^* = \frac{P - S}{(R-T) + (P-S)} = \frac{1}{2 + 1} = \frac13$. Here the gain (2) beats the fear (1), so (C, C) is the equilibrium played more often by agents with uniform beliefs.
>
> **(d)** $f_C - f_D = x(R - T) + (1-x)(S - P) = 2x - (1 - x) = 3x - 1$, positive above $1/3$. At $x = 1/2$, C earns 2.5 and D earns 2, so cooperators grow until everyone cooperates. Below 1/3 the population would go to all D.

> [!exam]- In a large population playing a Prisoner's Dilemma (T > R > P > S) under the replicator dynamics: (a) write $f_C - f_D$ in terms of the payoffs and the cooperator share $x$, (b) explain why cooperators die out from any start, and (c) explain why all-C is a rest point but not a stable one.
> **Key points:** $f_C - f_D = x(R - T) + (1-x)(S - P)$; greed makes the first term negative and fear the second, so $\dot x < 0$ for every interior $x$; all-D is the stable outcome and matches the NE (D, D); all-C rests only because nobody defects, and one defector earns T > R and spreads.
>
> **(a)** A partner cooperates with probability $x$, so $f_C = xR + (1-x)S$ and $f_D = xT + (1-x)P$, giving $f_C - f_D = x(R - T) + (1-x)(S - P)$.
>
> **(b)** Greed ($T > R$) makes $R - T < 0$ and fear ($P > S$) makes $S - P < 0$. The difference is an average of two negative numbers, so D earns more against every mix, and $\dot x = x(1-x)(f_C - f_D) < 0$ between 0 and 1. Even 90% cooperators decline to 0, the Nash equilibrium (D, D).
>
> **(c)** At $x = 1$ there are no defectors to copy, so $\dot x = 0$. But a single defector among cooperators earns $T$ against their $R < T$, so its share grows: a small push sends the population away, which makes the rest point unstable.

> [!exam]- In a repeated Prisoner's Dilemma ($R = 3, S = 0, T = 5, P = 1$), explain why memory matters for cooperation, why AllD can win a round robin with AllC, AllD, Random, TFT and WSLS, and why TFT beats AllD when only AllD, TFT and WSLS take part in a long game.
> **Key points:** memory-0 strategies cannot react to the partner; TFT is exploited only in round 1 and earns 3 per round with cooperators; AllD's score comes from exploiting AllC, Random and WSLS; with no exploitable partners AllD's one-off first-round gain is diluted, and TFT wins from 4 rounds.
>
> **Memory.** AllC, AllD and Random play the same way whoever they meet, so an unconditional cooperator is exploited every round. TFT and WSLS condition on the last round: TFT cooperates with cooperators and defects against AllD from round 2, capping its loss at one round.
>
> **Why AllD wins the five-strategy field.** It earns 5 per round from AllC and about 3 from Random and from WSLS, which counts mutual defection as a loss and returns to C every other round. Against TFT it earns little more than 1. Its win is paid for by exploitable partners.
>
> **Three-strategy field.** Without them, AllD gets 5 once against TFT and then 1 per round, while TFT and WSLS earn 3 per round from each other. From 4 rounds on that one-off gain is too diluted, and TFT ranks first.

## Flashcards

> [!card]- What payoff ordering defines a Prisoner's Dilemma?
> $T > R > P > S$.

> [!card]- What payoff ordering defines a Stag Hunt?
> $R > T > P > S$.

> [!card]- Which motives push to D in the Prisoner's Dilemma, and which in the Stag Hunt?
> Prisoner's Dilemma: greed ($T > R$) and fear ($P > S$). Stag Hunt: fear only.

> [!card]- Why is the Prisoner's Dilemma an incentive problem and the Stag Hunt a trust problem?
> In the PD, D is best even against a sure cooperator; in the SH only doubt about the partner pushes to D.

> [!card]- What is the threshold belief $p^*$ for playing C in a symmetric 2x2 game?
> $p^* = (P - S) / ((R - T) + (P - S))$.

> [!card]- What is $p^*$ in the Stag Hunt with $R = 4, S = 0, T = 3, P = 2$?
> 2/3.

> [!card]- When do agents with uniform beliefs play (C, C) more often than (D, D) in a Stag Hunt?
> When $R - T > P - S$: the gain beats the fear.

> [!card]- Why does "(C, C) is better for both, so rational agents play it" fail in the Stag Hunt?
> Each agent best-responds to its own belief; C pays only if that belief exceeds $p^*$.

> [!card]- Why does Tit-for-Tat do better against AllD than AllC does?
> It conditions on the partner's last move, so it is exploited only in round 1.

> [!card]- Why does AllD exploit WSLS every other round?
> WSLS counts mutual defection as a loss and switches back to C.

> [!card]- Why can AllD win a tournament that includes AllC and Random?
> Its points come from exploiting unconditional or random cooperators.

> [!card]- In a Prisoner's Dilemma population, why is $f_C - f_D$ negative for every $x$?
> It equals $x(R - T) + (1-x)(S - P)$, and greed and fear make both terms negative.

> [!card]- Why does the Stag Hunt's replicator tipping point equal the threshold belief $p^*$?
> Under random matching the cooperator share is the probability the partner cooperates.

> [!card]- True or false: a Stag Hunt population (R = 4, S = 0, T = 3, P = 2) starting with 60% stag hunters ends with everyone hunting stag.
> **False.** At 60%, hare earns 2.6 against stag's 2.4, so stag hunting dies out.

> [!card]- True or false: making Prisoner's Dilemma agents sure that their partner cooperates makes them cooperate.
> **False.** Greed remains: against a sure cooperator, D still pays $T > R$.

> [!card]- Multiple selection: which strategies cooperate at different rates depending on the opponent? (A) AllC (B) TFT (C) Random (D) WSLS
> **B and D.** Both are memory-1. A and C are memory-0 and cannot react.

## Links

- **Course:** [[AGAI - Overview|Agentic AI overview]]
- **Lecture:** [[AGAI-L02 - Game Theory and Population Dynamics]]
- **Concepts:** [[Nash Equilibrium]] · [[Pareto Optimality]] · [[Prisoner's Dilemma]] · [[Stag Hunt]] · [[Repeated Game]] · [[Replicator Dynamics]] · [[Reward Hacking]] · [[Value Iteration]] · [[Markov Decision Process]] · [[Bellman Optimality Equation]] · [[Reward Signal]] · [[Agent Program]] · [[Multi-Agent System]]
- **Source:** `Lab1-AgenticAI-2026.ipynb`, Agentic AI 2026/2027 (UvA MSc AI), Lab 1, "Simple agents, Complex dynamics", werkcollege 28 Oct 2026. The notebook ships without outputs; numbers come from re-running its models.
