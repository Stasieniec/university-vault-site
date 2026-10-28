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
> The lab uses agents with almost no memory and two or three actions to show three things. In one-shot games, self-interest gives the [[Prisoner's Dilemma]] mutual defection and leaves the [[Stag Hunt]] stuck between two equilibria, and in repeated games, memory lets cooperation pay. When agents adapt by copying what works instead of calculating, the [[Replicator Dynamics|replicator equation]] predicts where a population ends up, and the same tools explain when signals between agents stay honest. Finally, a perfect optimiser in a two-square vacuum world shows how a badly chosen reward turns a harmless new action into [[Reward Hacking]].
>
> The lab is ungraded practice that the lecturer calls the best preparation for the midterm. All fifteen questions are worked below with derivations. The notebook ships without outputs, so numbers from random simulations come from my own re-run of its models with the same seeds and order of random draws; they should match the classroom run, though small differences are possible. Analytic results (equilibria, thresholds, rest points, stability, policies) are exact.

**Concepts tested:** [[Nash Equilibrium]], [[Pareto Optimality]], [[Prisoner's Dilemma]], [[Stag Hunt]], [[Repeated Game]], [[Replicator Dynamics]], [[Reward Hacking]], [[Value Iteration]], [[Markov Decision Process]], [[Agent Program|simple reflex agents]]

---

## Part I: one-shot games, beliefs and repeated games

### The model

Every game in Part I is a **symmetric two-player, two-action game**: each player picks C (cooperate) or D (defect) at the same time, and four numbers give the row player's payoff (the column player is paid the same way):

| | partner plays C | partner plays D |
|---|---|---|
| **play C** | $R$ (reward) | $S$ (sucker) |
| **play D** | $T$ (temptation) | $P$ (punishment) |

A **best response** maximises my payoff given what my partner does. A **pure [[Nash Equilibrium]]** is an outcome where each action is a best response to the other, so no player gains by changing its action alone: where self-interested players can end up. An outcome is **[[Pareto Optimality|Pareto optimal]]** if no other outcome makes one player better off without making the other worse off ("not wasteful"). The **social optimum** maximises total welfare, the sum of the two payoffs; every social optimum is Pareto optimal, but not the reverse. With four outcomes, all are found by checking each cell. The two games used throughout:

- **[[Prisoner's Dilemma]] (PD):** $T > R > P > S$, here $R=3, S=0, T=5, P=1$.
- **[[Stag Hunt]] (SH):** $R > T > P > S$, here $R=4, S=0, T=3, P=2$. A stag (C) is caught only if both hunt it; a hare (D) can be caught alone. It stands for any coordination problem where a risky joint action pays most and a safe solo action tempts, such as two coding agents that must coordinate edits to shared files.

**Beliefs.** An agent unsure about its partner holds a belief $p$, the probability that the partner cooperates. Its expected payoffs are $\pi_C(p) = pR + (1-p)S$ and $\pi_D(p) = pT + (1-p)P$. Setting $\pi_C > \pi_D$:

$$p(R - T) > (1-p)(P - S) \;\Longleftrightarrow\; p > p^* = \frac{P - S}{(R - T) + (P - S)}$$

where $P - S$ is what you lose by cooperating when the partner does not (fear) and $R - T$ is what you gain by cooperating when the partner also does. Dividing keeps the direction of the inequality only when $(R-T)+(P-S) > 0$, which holds in a coordination game like the SH; for the lab's PD the formula gives $-1$, outside $[0,1]$. The larger $p^*$, the more trust hunting the stag needs.

**Repeated games.** In a [[Repeated Game]] the same two agents play the PD for several rounds, and a strategy maps the history to an action. Memory-0 strategies ignore the history; memory-1 strategies look at the last round. A memory-one strategy is five probabilities of cooperating: in the first round, and after each previous outcome (my last action, partner's last action):

| strategy | first | after CC | after CD | after DC | after DD |
|---|---|---|---|---|---|
| AllC | 1 | 1 | 1 | 1 | 1 |
| AllD | 0 | 0 | 0 | 0 | 0 |
| Random | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| TFT (Tit-for-Tat) | 1 | 1 | 0 | 1 | 0 |
| WSLS (Win-Stay Lose-Shift) | 1 | 1 | 0 | 0 | 1 |

TFT copies the partner's last action. WSLS repeats its last action after a "win" (3 or 5 points, which happens exactly when the partner cooperated) and switches after a "loss" (0 or 1). Memory-0 strategies are the special case with the last four numbers equal.

The tournament, modelled on Axelrod's 1980 tournament (won by Tit-for-Tat), is a round robin: every strategy plays every strategy, including a copy of itself, each pairing is repeated 50 times, and a strategy's score is its average payoff per round over all opponents. A `noise` parameter flips an agent's chosen action with some probability (an execution error); every question uses noise 0.

### Q1: predictions before the analysis

> [!question] Statement
> (a) You play the following game **once**, against an anonymous agent you will never meet again, both choosing simultaneously. Your payoff (row player): 3 if both cooperate, 0 if you cooperate and the other defects, 5 if you defect and the other cooperates, 1 if both defect. The other agent receives points the same way (a normal-form game). Which action would *you* choose? Which action would a purely self-interested agent choose?
>
> (b) The same game is played **50 times** against the same partner, and each agent sees the previous round. Five types enter a round-robin tournament (everyone plays everyone, including a copy of itself): AllC, AllD, Random (fair coin each round), TFT and WSLS, as defined above. Which type do you expect to obtain the **highest** average score per round, and which the **lowest**?

**(a)** Many people pick C, out of fairness or hope. A purely self-interested agent picks D, because D pays more whatever the other does ($5 > 3$ and $1 > 0$): D is a dominant action.

**(b)** The tempting prediction, from the Axelrod story, is that TFT wins and AllD loses. But in *this* field AllC and Random can be exploited every round and WSLS every other round, so AllD collects 5 per round from AllC and about 3 from Random and from WSLS. It may well stay on top, while AllC, which never punishes, should come last. The run (Q6) confirms this: AllD 2.62, TFT 2.45, WSLS 2.34, Random 2.24, AllC 2.11.

### Q2: Nash, Pareto and social optimum in the PD

> [!question] Statement
> The code finds, for the PD with $R=3, S=0, T=5, P=1$: pure Nash equilibrium (D,D); Pareto-optimal outcomes (C,C), (C,D), (D,C); social optimum (C,C).
>
> (a) Is the Nash equilibrium of the Prisoner's Dilemma Pareto optimal? Is it the social optimum? Explain why.
>
> (b) Using the idea of a best response, explain in one or two sentences why two self-interested agents end up at the Nash equilibrium even though both prefer (C,C). Hint: does the best response of the row player depend on what the column player does?
>
> (c) (C,D) is Pareto optimal. Does that make it a "good" outcome? What does this tell you about the difference between Pareto optimality and the social optimum?

| | col C | col D |
|---|---|---|
| **row C** | (3, 3), welfare 6 | (0, 5), welfare 5 |
| **row D** | (5, 0), welfare 5 | (1, 1), welfare 2 |

**(a)** No to both. (C,C) gives each player 3 instead of 1, so (D,D) is Pareto dominated. Its welfare, 2, is the lowest of the four cells, against 6 at the social optimum (C,C).

**(b)** The row player's best response is D whether the column player plays C ($5 > 3$) or D ($1 > 0$): it does not depend on the partner at all. The same holds for the column player, so (D,D) is the only cell where both are best-responding. Preferring (C,C) jointly does not help, because at (C,C) each player gains 2 by switching alone.

**(c)** No. (C,D) is Pareto optimal only because any move away from it hurts the column player, who gets 5, the most anyone can get, yet it is unequal (0 against 5) and wastes welfare (5 against 6). Pareto optimality only rules out gains that cost nobody anything; it says nothing about total welfare or fairness. The social optimum, the outcome with the largest sum, is much stronger: of the three Pareto-optimal outcomes, only (C,C) qualifies.

### Q3: two games, two motives

> [!question] Statement
> The code finds, for the SH with $R=4, S=0, T=3, P=2$: pure Nash equilibria (C,C) and (D,D); Pareto optimal (C,C); social optimum (C,C).
>
> (a) Compare the Stag Hunt with the Prisoner's Dilemma: how many pure Nash equilibria does each game have? In which game is mutual cooperation an equilibrium?
>
> (b) Two motives can push a player towards D: **greed** (defecting pays more than cooperating when the partner cooperates, $T > R$) and **fear** (defecting pays more than cooperating when the partner defects, $P > S$). Which of these motives is present in the PD? Which in the SH? Use this to explain why the SH has a problem of *trust/coordination* while the PD has a problem of *incentives*.

**(a)** The PD has one pure Nash equilibrium, (D,D); the SH has two, (C,C) and (D,D). Mutual cooperation is an equilibrium only in the SH: a stag hunter who switches to hare drops from 4 to 3. In the PD a cooperator facing a cooperator gains by switching ($5 > 3$).

**(b)**

| motive | condition | PD (3, 0, 5, 1) | SH (4, 0, 3, 2) |
|---|---|---|---|
| greed | $T > R$ | yes, $5 > 3$ | no, $3 < 4$ |
| fear | $P > S$ | yes, $1 > 0$ | yes, $2 > 0$ |

In the PD both motives point to D, so D is dominant: even an agent *certain* that its partner will cooperate defects. No amount of trust fixes that; the payoffs themselves must change (punishment, contracts, repetition), so it is an incentive problem. In the SH there is no greed, so an agent sure that its partner hunts stag hunts stag too; only fear, doubt about the partner, pushes it to D. Fix the beliefs and cooperation follows, so the problem is trust and coordination.

### Q4: which Stag Hunt equilibrium? (prediction)

> [!question] Statement
> Two agents meet once to play the Stag Hunt ($R=4, S=0, T=3, P=2$). They cannot communicate and know nothing about each other.
>
> (a) Which outcome do you expect to be most frequent: (C,C), (D,D), or a miscoordinated outcome?
>
> (b) Suppose the hare becomes more attractive: $T$ increases from 3 to 4, everything else fixed. Will (C,C) become more or less frequent? Why?

**Careful prediction.** With no information, the natural model, which the lab then uses, gives each agent a belief $p$ drawn uniformly from $[0, 1]$, and each plays its best response. Here $p^* = \frac{2 - 0}{(4-3) + (2-0)} = \frac{2}{3}$, so an agent hunts stag with probability $1/3$. Independently for the two agents:

$$\Pr(C,C) = \left(\tfrac13\right)^2 = \tfrac19 \approx 0.111, \quad \Pr(D,D) = \left(\tfrac23\right)^2 = \tfrac49 \approx 0.444, \quad \Pr(\text{miscoordinated}) = 2 \cdot \tfrac13 \cdot \tfrac23 = \tfrac49.$$

**(a)** (C,C) is by far the least frequent outcome; (D,D) and miscoordination are equally likely in expectation. The simulation of 10,000 encounters gives (C,C) 0.110, (D,D) 0.446, miscoordinated 0.445.

**(b)** Less frequent, in fact never. With $T = 4$, $p^* = 2 / (0 + 2) = 1$, and no belief exceeds 1, so nobody hunts stag: (C,C) 0.000, (D,D) 1.000. A larger $T$ shrinks the gain from cooperating with a cooperator ($R - T$ drops from 1 to 0) while the fear term stays at 2. At $T = R = 4$ the game is no longer a strict Stag Hunt: hare pays the same as stag against a stag hunter and more against a hare hunter, so D weakly dominates C.

### Q5: a rule for which equilibrium wins

> [!question] Statement
> (a) Find a simple rule, written only in terms of $R, S, T, P$, that predicts which equilibrium is played more often.
>
> (b) Consider this sentence: "Rational agents will always coordinate on (C,C), because it is better for both of them." Which evidence from this section contradicts this, and what is the flaw in the argument?
>
> (c) The results depend on how we modelled what agents believe before playing. Describe one situation (for example, related to communication or memory) in which you would expect a different outcome.

**(a)** Under uniform beliefs, $\Pr(C,C) = (1 - p^*)^2$ and $\Pr(D,D) = (p^*)^2$, so (C,C) is more frequent exactly when

$$1 - p^* > p^* \iff p^* < \tfrac12 \iff R - T > P - S.$$

**Rule:** (C,C) is played more often when the gain from cooperating with a cooperator, $R - T$, exceeds the loss from cooperating with a defector, $P - S$; (D,D) when fear outweighs gain. My re-runs agree: $(R,S,T,P) = (5, 0, 2, 1)$ gives (C,C) 0.573 against (D,D) 0.065, and $(4, 0, 3, 1)$ a tie (0.252 against 0.251). (Outside the lab, game theory calls the equilibrium that wins this comparison *risk dominant*, a term from Harsanyi and Selten.)

**(b)** The baseline run: rational best-responders reach (C,C) in only 11% of encounters and (D,D) in 45%. The argument confuses "better for both if both do it" with "best for me given what I expect". Each agent chooses alone, and (C,C) being Pareto superior does not make stag the best response unless the agent believes its partner hunts stag with probability above $p^* = 2/3$. With no reason for that belief, the safe hare wins.

**(c)** Anything that makes beliefs high and shared changes the result. **Communication:** if agents can say "I will hunt stag" before playing, keeping the promise pays, because once the partner believes it and hunts stag, stag is the sender's own best response; beliefs jump above $p^*$ and (C,C) becomes common. **Memory and repetition:** agents who meet repeatedly update $p$ from experience, so a few successful stag hunts lock in (C,C) and a few failures lock in (D,D). **Reputation or shared conventions:** a partner known to hunt stag, or agents built by the same company, start with $p$ near 1.

### Q6: one-shot against repeated tournament

> [!question] Statement
> The tournament of Q1(b) is run twice, with 1 round and with 50 rounds.
>
> (a) Compare the two rankings. How did the scores of AllD, TFT and WSLS change when going from 1 to 50 rounds? Was your prediction in Q1(b) correct? In the one-shot game AllC, TFT and WSLS take exactly the same action, yet their scores differ slightly: why, and what does this tell you about how to read small differences in these rankings?
>
> (b) Look at the **cooperation-rate** matrix. Which strategies cooperate at *different* rates depending on who they play against, and which do not? Relate this to the memory of each strategy. Why is this ability essential for cooperation to pay off against a defector?
>
> (c) AllD may still score well in this strategy space. Using the payoff matrix, identify *whose* payoff AllD's score mostly comes from. Quote the numbers.

Rankings (points per round), with the exact expected values I derived alongside:

| rank | one-shot | exact | 50 rounds | exact |
|---|---|---|---|---|
| 1 | AllD 3.82 | 3.80 | AllD 2.62 | 2.616 |
| 2 | Random 2.96 | 2.95 | TFT 2.45 | 2.443 |
| 3 | AllC 2.12 | 2.10 | WSLS 2.34 | 2.347 |
| 4 | WSLS 2.05 | 2.10 | Random 2.24 | 2.264 |
| 5 | TFT 2.02 | 2.10 | AllC 2.11 | 2.100 |

**(a)** From 1 to 50 rounds, AllD falls from 3.82 to 2.62, TFT rises from 2.02 to 2.45 and WSLS from 2.05 to 2.34. In a single round TFT and WSLS are plain cooperators and get exploited. Over 50 rounds TFT is exploited by AllD only in the first round and WSLS only every other round, and both earn 3 per round from each other. Whether the Q1(b) prediction was right depends on what you wrote: AllD still wins this field and AllC is last, so "TFT wins" was wrong here.

In the one-shot game AllC, TFT and WSLS all play C, so their expected score is identical: $(3 + 0 + 1.5 + 3 + 3)/5 = 2.10$. The spread (2.12, 2.05, 2.02) comes only from the Random opponent's coin in each strategy's own 50 games against it: a payoff of 3 or 0 with equal chance has standard deviation 1.5, so the tournament score carries a standard error of about $1.5/\sqrt{50}/5 \approx 0.04$. Differences of a few hundredths are noise; trust only gaps well above 0.1, or derive the expected values.

**(b)** Cooperation rates over 50 rounds:

| | vs AllC | vs AllD | vs Random | vs TFT | vs WSLS |
|---|---|---|---|---|---|
| **AllC** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **AllD** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **Random** | 0.50 | 0.50 | 0.52 | 0.47 | 0.50 |
| **TFT** | 1.00 | 0.02 | 0.52 | 1.00 | 1.00 |
| **WSLS** | 1.00 | 0.50 | 0.53 | 1.00 | 1.00 |

AllC, AllD and Random cooperate at the same rate whoever they face (Random's wiggles are sampling noise): they are memory-0, so they cannot react to the partner. TFT and WSLS are memory-1, and their rates depend on the partner: full cooperation with cooperators, but 0.02 (only the first round) for TFT against AllD and 0.50 for WSLS against AllD.

This conditioning is what makes cooperation pay against a defector. An unconditional cooperator is exploited every round (AllC earns 0 against AllD). A strategy that withdraws cooperation after being defected on caps its loss at one round (TFT earns 0.98 against AllD) while still earning 3 per round with fellow cooperators. WSLS is weaker here: mutual defection (1 point) counts as a loss, so it switches back to C and is exploited every other round (0.50).

**(c)** AllD's 2.62 averages its scores against AllC (5.00), itself (1.00), Random (3.01), TFT (1.08) and WSLS (3.00). Of the 13.09 total, 11 points come from AllC, Random and WSLS: AllC hands it $T = 5$ every round while getting $S = 0$, and Random and WSLS give it 5 about half the time (earning 0.50 themselves). Against the reciprocator TFT it gets 1.08, barely above mutual defection. AllD's score is paid for by exploitable partners, so it depends on how many of them are in the field.

### Q7: when does AllD stop being best?

> [!question] Statement
> In a strategy space with only AllD, TFT and WSLS: from how many rounds onwards do you expect AllD to stop being the best strategy? Give a number and a one-line reason.

**Answer: from 4 rounds onwards.** One-line reason: AllD's edge over TFT is a one-off first-round temptation, while TFT and WSLS earn 3 per round from each other, so from 4 rounds that edge is too diluted.

**Derivation.** No noise, so everything is deterministic. With $n$ rounds, per-round payoffs in each pairing:

- AllD vs AllD: 1. AllD vs TFT: 5 in round 1, then 1, so $(n + 4)/n$. AllD vs WSLS: WSLS switches after every loss, so it plays C, D, C, ... and AllD gets 5, 1, 5, 1, ..., a total of $5\lceil n/2 \rceil + \lfloor n/2 \rfloor$.
- TFT vs AllD: 0 then 1, so $(n-1)/n$. TFT vs TFT and TFT vs WSLS: 3 each.
- WSLS vs AllD: 0, 1, 0, 1, ..., so $\lfloor n/2 \rfloor / n$. WSLS vs TFT and vs WSLS: 3.

Tournament scores (average over the three opponents), multiplied by 3:

$$3\,\text{AllD} = \begin{cases} 5 + 4/n & n \text{ even} \\ 5 + 6/n & n \text{ odd} \end{cases} \qquad 3\,\text{TFT} = 7 - \frac1n \qquad 3\,\text{WSLS} = 6 + \frac{\lfloor n/2 \rfloor}{n}.$$

TFT is always at least WSLS, so AllD is best exactly when it beats TFT:
- $n$ even: $5 + 4/n > 7 - 1/n \iff 5/n > 2 \iff n < 2.5$;
- $n$ odd: $5 + 6/n > 7 - 1/n \iff 7/n > 2 \iff n < 3.5$.

So AllD wins for $n = 1, 2, 3$ and TFT from $n = 4$. The run agrees exactly: AllD against TFT is 2.33 against 2.22 at 3 rounds, 2.00 against 2.25 at 4, and 1.69 against 2.33 at 50 (WSLS 2.17). Removing AllC and Random is what flips the 50-round result of Q6: without exploitable partners AllD falls to 1.69.

> [!tip] Beyond the questions: execution noise
> My own extra run, with 5% noise (the notebook uses 0). One accidental D sends two TFT players into an echo of alternating retaliation, and TFT's self-play payoff over 200 rounds drops to 2.26. WSLS recovers in two rounds: after an accidental (D,C) the defector won and stays D, the victim lost and switches to D, and at (D,D) both lose and switch back to C. WSLS self-play stays at 2.77, and in the five-strategy field over 50 rounds WSLS (2.38) overtakes TFT (2.22).

> [!warning] Common pitfalls in Part I
> - **Pareto optimal is not socially optimal.** (C,D) in the PD is Pareto optimal and still a bad outcome.
> - **Greed and fear mixed up.** The PD has both (an incentive problem, fixed by changing payoffs); the SH has only fear (a trust problem, fixed by changing beliefs).
> - **"TFT always wins."** Rankings depend on the field: with AllC and Random present, AllD wins the 50-round tournament.

---

## Part II: replicator dynamics and deception

### The replicator equation

In Part I agents calculated. Many agents instead **adapt**: a trading bot keeps the rule that made money, a reinforcement learner repeats what paid off, a platform deploys more copies of the agent that sells best. Evolutionary game theory asks where that adaptation leads.

Take a large population in which a fraction $x$ cooperates and $1 - x$ defects, paired at random. A partner cooperates with probability $x$, so the expected payoffs (fitness) are

$$f_C(x) = xR + (1-x)S, \qquad f_D(x) = xT + (1-x)P, \qquad \bar f(x) = x f_C(x) + (1-x) f_D(x),$$

and the **[[Replicator Dynamics|replicator equation]]** says a strategy grows when it earns more than average, at a rate proportional to the gap and to how common it already is:

$$\frac{dx}{dt} = x\big(f_C(x) - \bar f(x)\big).$$

With $k$ strategies, frequencies $\mathbf x$ and payoff matrix $A$: $\dfrac{dx_i}{dt} = x_i\big((A\mathbf x)_i - \mathbf x^\top A \mathbf x\big)$ (at $x = 0.3$ in the PD both forms give $-0.273$ for cooperators). The equation is the average behaviour of a population where agents copy a random peer with probability proportional to how much better the peer does (imitation of success), and the expected behaviour of a simple reinforcement learner (Cross learning) with a small learning rate (Börgers and Sarin, 1997).

A **rest point** is an $x$ with $dx/dt = 0$. It is **stable** if the population returns after a small push either way, **unstable** otherwise. The starting states that end at a stable rest point form its **basin of attraction**. The equation is solved numerically with the Runge-Kutta method (RK4, step 0.05).

A quick stability test, derived from the factorised form of Q9(a): write $h(x) = f_C(x) - f_D(x)$, so $dx/dt = g(x) = x(1-x)h(x)$. Then $g'(0) = h(0) = S - P$ and $g'(1) = -h(1) = T - R$, so $x = 0$ (all defect) is stable when $P > S$ (fear protects defection), and $x = 1$ (all cooperate) is stable when $R > T$ (no greed protects cooperation).

### Q8: predictions before the analysis

> [!question] Statement
> A large population of agents is repeatedly paired at random to play a game. The agents do not compute equilibria; strategies that earn more than average spread (agents copy successful peers, or successful agents are deployed more often).
>
> (a) The population plays the Prisoner's Dilemma ($R=3, S=0, T=5, P=1$) and starts with **90% cooperators**. What happens to the fraction of cooperators over time?
>
> (b) The population plays the Stag Hunt ($R=4, S=0, T=3, P=2$) and starts with **60% cooperators** (stag hunters). What happens to the fraction of cooperators over time?

**(a)** Cooperators die out. Defectors earn more than cooperators against *any* mix of partners (D is dominant), so their share grows from any start; 90% only delays it. At $x = 0.9$: $f_C = 2.7$, $f_D = 4.6$. Run: $x$ falls below 0.5 at $t \approx 1.3$ and below 0.01 at $t \approx 5.4$.

**(b)** The tempting prediction is that a cooperative majority wins. Compare payoffs at $x = 0.6$: $f_C = 4(0.6) = 2.4$ and $f_D = 3(0.6) + 2(0.4) = 2.6$. Hare hunters earn more, so stag hunting declines, and keeps declining because the gap widens as $x$ falls. Run: $x = 0.528$ at $t = 1$, $0.004$ at $t = 5$, then 0. Cooperation needs more than $2/3$ (Q10).

### Q9: the Prisoner's Dilemma under replicator dynamics

> [!question] Statement
> The code reports, for the PD: rest points $x = 0$ (stable) and $x = 1$ (unstable), and all ten trajectories from $x(0) \in [0.05, 0.95]$ fall to 0.
>
> (a) Show, in one or two lines of algebra, that $x\big(f_C - \bar f\big) = x(1-x)(f_C - f_D)$.
>
> (b) For the Prisoner's Dilemma, write $f_C(x) - f_D(x)$ in terms of $R, S, T, P$ and $x$. Using the greed ($T > R$) and fear ($P > S$) motives from Part I, explain why it is negative for every $x$. What does this imply for the phase line, and for your prediction in Q8(a)?
>
> (c) Compare the rest points with the Nash equilibrium of the Prisoner's Dilemma. Which rest point is stable, and which Nash equilibrium does it correspond to? The state $x = 1$ (everybody cooperates) is also a rest point: why does the population not stay there after a single defector appears?

**(a)** Substitute $\bar f$:

$$f_C - \bar f = f_C - x f_C - (1-x) f_D = (1-x) f_C - (1-x) f_D = (1-x)(f_C - f_D),$$

and multiply by $x$. The factor $x(1-x)$ makes change slow when either strategy is rare; the sign is set by $f_C - f_D$ alone.

**(b)**

$$f_C(x) - f_D(x) = x(R - T) + (1-x)(S - P).$$

Greed means $R - T < 0$ and fear means $S - P < 0$. For $x \in [0, 1]$ the expression is a weighted average of two negative numbers, so it is negative for every $x$; for the lab's PD it is $-2x - (1 - x) = -(1 + x)$. So $dx/dt < 0$ on the whole open interval $(0, 1)$: the phase line lies below zero everywhere between the endpoints, and every population with any defectors slides to $x = 0$. Q8(a): 90% cooperators end at 0.

**(c)** Rest points: $x = 0$ (stable) and $x = 1$ (unstable). There is no interior one: the formula $(P - S)/((R-T)+(P-S)) = 1/(-2+1) = -1$ lies outside $[0, 1]$, because $f_C - f_D$ never reaches zero. The stable rest point $x = 0$, everyone defecting, corresponds to the unique Nash equilibrium (D,D). The state $x = 1$ is a rest point only because there is nobody to copy: with no defectors, the defector share cannot grow. A single defector among cooperators earns $T = 5$ against their $R = 3$, so it is copied and its share grows ($g'(1) = T - R = 2 > 0$). A rest point without stability holds only while nothing disturbs it.

### Q10: the Stag Hunt under replicator dynamics

> [!question] Statement
> The code reports, for the SH: rest points $x = 0$ (stable), $x = 0.667$ (unstable), $x = 1$ (stable); trajectories starting at 0.75 and above rise to 1, those at 0.65 and below fall to 0.
>
> (a) Study the sign of $f_C(x) - f_D(x)$ for the Stag Hunt. Where do you expect the equation to be positive, negative, zero?
>
> (b) List the rest points of the Stag Hunt and their stability. The interior rest point corresponds to the *mixed* Nash equilibrium of the game (each agent cooperates with probability $x$). Is this equilibrium something an adaptive population will end up in? Why or why not?
>
> (c) Compare the interior rest point with the threshold belief $p^*$ from Part I (same payoffs). In Part I, rational agents chose a best response to a belief; here agents have no beliefs and only copy what works. Why do both models give the same threshold? Hint: what plays the role of the belief $p$ when partners are drawn at random from the population?
>
> (d) Check your prediction Q8(b). Consider the sentence: "If the majority already cooperates, cooperation must win, because (C,C) is better for everyone." Which output contradicts this, and what is wrong with the argument?

**(a)** $f_C - f_D = x(4 - 3) + (1-x)(0 - 2) = 3x - 2$: negative for $x < 2/3$, zero at $2/3$, positive above. So $dx/dt = x(1-x)(3x - 2)$ is negative on $(0, 2/3)$, zero at $0$, $2/3$ and $1$, and positive on $(2/3, 1)$.

**(b)**

| rest point | stability | why | basin |
|---|---|---|---|
| $x = 0$ (all hare) | stable | $g'(0) = S - P = -2 < 0$ | $[0, 2/3)$ |
| $x = 2/3$ (mixed) | unstable | $g'(2/3) = \tfrac23 \cdot \tfrac13 \cdot 3 = \tfrac23 > 0$ | only the point itself |
| $x = 1$ (all stag) | stable | $g'(1) = T - R = -1 < 0$ | $(2/3, 1]$ |

An adaptive population will not end up at the mixed equilibrium. Both strategies earn the same there, but any small push tips it: slightly more stag hunters make stag pay more, which attracts more stag hunters, and the reverse below. It is the watershed between the two basins, kept only by a population that starts exactly there and is never disturbed.

**(c)** The interior rest point is

$$x^* = \frac{P - S}{(R - T) + (P - S)} = \frac{2}{1 + 2} = \frac23 = p^*.$$

Under random matching, the probability that my partner cooperates is the population share $x$. So $f_C(x) = \pi_C(p = x)$ and $f_D(x) = \pi_D(p = x)$: the fitness functions are Part I's expected payoffs with the share standing in for the belief. A rest point needs $f_C(x) = f_D(x)$, the same equation as the rational agent's indifference $\pi_C(p) = \pi_D(p)$, so it has the same solution. The rational agent cooperates when its belief exceeds $p^*$; the population grows its cooperators when the actual share exceeds $p^*$.

**(d)** Q8(b) is confirmed: from 60% stag hunters the population falls to 0, and even the trajectory from 65% ends at 0, as the unstable rest point at $2/3$ predicts. The flaw: what spreads is what earns more than average *given the current mix*. "Everyone is better off if everyone cooperates" compares whole outcomes, which no adapting agent ever does. With 60% stag hunters a stag hunter fails 40% of the time and averages 2.4, below a hare hunter's 2.6. The bar is $2/3$ rather than $1/2$ because the fear term ($P - S = 2$) outweighs the gain ($R - T = 1$).

### Deception: the signalling game

Many interactions have **asymmetric information**: one agent knows its own quality and the other does not. The informed agent can send a **signal** (a "premium" label), which is useful only if reliable, and low-quality agents are tempted to fake it. A malicious agent that presents itself as trustworthy is the agentic version.

- A fraction $\lambda$ of signallers is high quality (H), the rest low quality (L). Signallers know their quality; receivers do not.
- Each signaller sends the signal or not, at cost $c_H$ for H and $c_L$ for L.
- Each receiver is **trusting** (accepts a signaller if and only if it signals) or **sceptical** (accepts nobody).
- An accepted signaller gains $b$, whatever its quality. A receiver gains $g$ from accepting H and loses $d$ from accepting L.

A low-quality signaller that signals **deceives**. Let $x_H$, $x_L$ be the fractions of H and L signallers that signal and $y$ the fraction of trusting receivers. Each strategy's advantage over the alternative (which pays 0) is $yb - c_H$ for signalling H, $yb - c_L$ for signalling L, and $\lambda x_H g - (1-\lambda) x_L d$ for trusting. Each group follows the two-strategy replicator equation ("$x(1-x)$ times the advantage"):

$$\frac{dx_H}{dt} = x_H(1-x_H)(yb - c_H), \quad \frac{dx_L}{dt} = x_L(1-x_L)(yb - c_L), \quad \frac{dy}{dt} = y(1-y)\big(\lambda x_H g - (1-\lambda) x_L d\big).$$

H and L are separate groups because an agent chooses whether to signal but not its quality. The **reliability** of the signal is $\Pr(H \mid \text{signal}) = \lambda x_H / (\lambda x_H + (1-\lambda) x_L)$; without a signal a receiver knows only the prior $\lambda$. Defaults: $\lambda = 0.3$, $b = g = d = 1$, so accepting a random signaller loses ($0.3 - 0.7 < 0$) and receivers benefit only from an informative signal. Each run lasts until $t = 300$ and the summary averages the second half; a receiver's average payoff is $y$ times the advantage of trusting.

### Q11: does faking pay? (prediction)

> [!question] Statement
> Fix $c_H = 0.2$ (the signal is cheap for high-quality signallers) and start from $x_H = x_L = y = 0.5$. Consider two cases: **(A)** faking is expensive, $c_L = 1.5$; **(B)** faking is cheap, $c_L = 0.5$. For each case, predict the long-run behaviour. Do low-quality signallers deceive? Do receivers keep trusting? Is the signal reliable?

**(A)** Faking pays at most $b - c_L = -0.5$ even if every receiver trusts, so $x_L$ falls whatever happens. For H, signalling pays whenever $y > 0.2$. Once fakers are rare, trusting pays ($0.3 x_H > 0.7 x_L$), so $y$ rises, which makes signalling even better for H. Expected end: $x_H = 1$, $x_L = 0$, $y = 1$, an honest (separating) signal with reliability 1. Receivers then earn $\lambda g = 0.3$ and H signallers $b - c_H = 0.8$.

**(B)** Now faking pays whenever $y > 0.5$, and no state leaves all three groups content. High trust lets fakers spread; common fakers make trusting lose ($0.3 x_H < 0.7 x_L$), so trust falls; below $y = 0.5$ faking stops paying and fakers decline; then trusting pays again. Expect cycles, with H signallers still signalling since $y$ hovers around 0.5, above their break-even of 0.2. The cycles turn around the point where L signallers and receivers are both indifferent, with $x_H = 1$:

$$y b = c_L \Rightarrow y = 0.5, \qquad \lambda x_H g = (1-\lambda) x_L d \Rightarrow x_L = \frac{0.3}{0.7} = \frac37 \approx 0.43,$$

where reliability is $0.3 / (0.3 + 0.7 \cdot 3/7) = 0.5$: better than the prior of 0.3, far from reliable.

**Runs** (second-half averages):

| | (A) $c_L = 1.5$ | (B) $c_L = 0.5$ |
|---|---|---|
| $x_H$ | 1.000 | 1.000 |
| $x_L$ (deceives) | 0.000 | 0.444 (cycles 0.19 to 0.69) |
| $y$ (trust) | 1.000 | 0.498 (cycles 0.28 to 0.72) |
| reliability | 1.000 | 0.513 |
| receiver payoff | 0.300 | $-0.006$ |

Both predictions hold. In (A) fakers vanish fast ($x_L = 0.008$ by $t = 5$) and the signal becomes fully honest. In (B) low-quality signallers deceive much of the time, trust rises and falls with them in cycles that do not shrink, and receivers gain essentially nothing.

### Q12: the uniform fee

> [!question] Statement
> A platform designer looks at the market of Q11 (B), where the label costs $c_H = 0.2$ and $c_L = 0.5$ and low-quality sellers fake it. Fearing that the label will always be misused, the designer **makes it expensive for everyone**: every seller that shows the label pays an extra fee $k = 1$ (an expensive verification procedure). The designer cannot observe quality, so the fee is the same for every seller: $c_H = 1.2$ and $c_L = 1.5$.
>
> (a) Prediction (before running the next cell): will the fee stop deception? Will receivers be better off? Will high-quality sellers?
>
> (b) Run the next cell and compare the two markets. Did the fee eliminate deception, and how? Compare the receivers' payoff, the high-quality sellers' payoff and the total welfare in both markets. Hint: why is the receivers' payoff close to 0 even in the market with deception?
>
> (c) Is the high fee good or bad? Rewrite the designer's reasoning so that it is correct.

Total welfare per encounter is one receiver's payoff plus one randomly drawn seller's: $y(\lambda x_H g - (1-\lambda)x_L d) + \lambda\, x_H(yb - c_H) + (1-\lambda)\, x_L(yb - c_L)$.

**(a)** With $c_H = 1.2 > b = 1$, even an honest seller loses money on the label when every receiver trusts, so H sellers stop signalling, and L sellers too ($1.5 > 1$). Deception ends because the label disappears. Receivers were earning about 0 already (Q11 B) and with no signals earn exactly 0, so they gain nothing. High-quality sellers lose the 0.3 they were earning.

**(b)**

| (second-half averages) | no fee, $k = 0$ | fee for everyone, $k = 1$ |
|---|---|---|
| $x_H$, $x_L$ | 1.000, 0.444 | 0.000, 0.000 |
| $y$ | 0.498 | 0.453 (frozen) |
| receiver payoff | $-0.006$ | 0.000 |
| H seller payoff | 0.298 | 0.000 |
| total welfare | 0.084 | 0.000 |

The fee eliminated deception by eliminating signalling: both seller groups abandon the label within about 10 time units. Trust freezes at 0.453 because once nobody signals, trusting and scepticism pay the same and $dy/dt = 0$.

Why receivers earn about 0 even with deception: low-quality sellers fake as much as trust allows, and the cycle turns around $x_L = 3/7$, where trusting pays the same as scepticism. More precisely (my derivation), $\frac{d}{dt}\ln\frac{y}{1-y} = \frac{\dot y}{y(1-y)}$ is the advantage of trusting; since $y$ stays between 0.28 and 0.72 the log-odds stays bounded, so that advantage averages about 0. A receiver's payoff is $y$ times the advantage; because the two move together over the cycle its average is not exactly zero, but it is small ($-0.006$). The deceivers absorb what receivers would gain from honest sellers, so the fee neither cost receivers anything nor helped them.

**(c)** The fee is bad. Receivers stay at 0, high-quality sellers fall from 0.298 to 0, total welfare from 0.084 to 0. Deception is gone only because the market for the label is gone.

The designer's error: what stops faking is a **cost difference** between types. An honest signal survives when it pays for high-quality sellers and not for low-quality ones,

$$c_H < b < c_L,$$

as in case (A). A fee charged to everyone moves both costs by the same $k$, so it can help only if it lands $b$ between them: $0.2 + k < 1 < 0.5 + k$, so $0.5 < k < 0.8$; $k = 1$ overshoots and prices out the honest sellers too. (My extra runs: even $k = 0.6$ collapses the market from the 50/50 start, because trust of 0.5 is below the honest sellers' new break-even of 0.8; from a high-trust start, $x_H = 0.9$, $x_L = 0.5$, $y = 0.9$, it reaches the honest outcome with welfare 0.36.)

The corrected reasoning: "The label is misused because faking it is cheap relative to what being trusted is worth. I should make it costly to fake while keeping it cheap for genuine sellers, for example a verification that genuine sellers pass and fakers fail, or a penalty for sellers caught faking. Raising the price for everyone removes honest sellers along with the fakers."

> [!warning] Common pitfalls in Part II
> - **Rest point taken for stable.** $x = 1$ in the PD and $x = 2/3$ in the SH are rest points that one push destroys. Check $g'(0) = S - P$ and $g'(1) = T - R$.
> - **Share confused with belief.** They play the same role under random matching, which is why $x^* = p^*$, but adaptive agents hold no beliefs.
> - **"Expensive signals are honest signals."** Honesty needs the cost to differ by type; a uniform fee large enough to stop fakers also stops honest senders.

---

## Part III: the vacuum world and reward hacking

### The model

Payoffs in Parts I and II were given. When an AI agent is trained, someone has to write the reward function, and the agent then does whatever earns most reward, including things the designer never intended ([[Reward Hacking]]). The lab shows this in the two-square vacuum world of Russell and Norvig, with dirt that comes back.

- Two squares, A (left) and B (right), each clean or dirty; the robot is in one of them.
- **Left** moves to A, **Right** to B; moving into the wall leaves the robot where it is (the motor still runs).
- **Suck** on a dirty square cleans it and puts the dirt in the bag (which holds 0 or 1). Suck on a clean square does nothing.
- **Dump** (added later by a software update) empties a full bag onto the current square, which becomes dirty. With an empty bag it does nothing.
- After the action's effect, each clean square becomes dirty with probability $p$ (the dirt rate, 0.1 by default), independently. A dirty square stays dirty until sucked.

The state is (robot position, A dirty, B dirty, bag full), so $2^4 = 16$ states: a small [[Markov Decision Process]]. Two rewards $r(s, a, s')$:

- **Bob:** +1 for every clean square in the next state $s'$, at every time step.
- **Alice:** +1 for every unit of dirt sucked up (Suck on a square that was dirty in $s$).

The robot knows the dynamics and computes an optimal policy with [[Value Iteration]], discount $\gamma = 0.95$:

$$Q(s,a) = \sum_{s'} P(s' \mid s,a)\big[r(s,a,s') + \gamma V(s')\big], \qquad V(s) = \max_a Q(s,a),$$

starting from $V = 0$ and repeating until the values stop changing. Ties (actions within $10^{-6}$ of the best) are broken at random. The robot runs 1000 steps from "robot in A, both squares dirty, bag empty", with the same random dirt for every robot, and is judged by a **performance measure** that ignores its reward: floor clean (%), moves per step (wall bumps included), dumps per step (with a full bag), dirt collected per step, and the reward per step it sees. Since the optimiser is exact, any odd behaviour comes from the reward.

### Q13: predictions before the analysis

> [!question] Statement
> The goal is a clean office. Which reward function best accomplishes this?
>
> (a) Bob's reward: **+1 for every clean square, at every time step**, and no other benefits or costs. Suppose that, after the start, no new dirt ever appears. What will the robot do once both squares are clean?
>
> (b) Alice's reward: **+1 for every unit of dirt the robot sucks up.** In the test lab (actions Left, Right, Suck), which robot keeps the floor cleaner: Bob's, Alice's, or neither? A software update then adds a fourth action, **Dump**, which empties the robot's bag onto its current square (useful, say, to empty the bag above a bin). Predict what each robot does after the update.

**(a)** Once both squares are clean, every action earns the same +2 per step forever, so the robot is indifferent and the tie-breaking rule decides. With random tie-breaking it wanders, moving on about two steps in three (two of its three actions are moves).

**(b)** In the lab, neither: dirt can only arrive from the environment and only Suck removes it, so both rewards are maximised by sucking dirt as fast as possible, and the robots should behave the same. After the update, Bob's robot never dumps, because dirt on the floor lowers his reward. Alice's robot dumps and re-sucks the same dirt, since every dump creates a new +1 one step later.

### Q14: Bob's robot

> [!question] Statement
> Bob's reward, no new dirt (dirt rate 0), robot starts in A with both squares dirty. The policy table shows the bag-empty states.
>
> (a) Look at the policy table. In which states does the robot have more than one optimal action? Why are these actions equally good under Bob's reward?
>
> (b) Check your prediction Q13(a). Once the floor is clean and no new dirt arrives, what does the robot do? Is the robot behaving *irrationally*? Answer in terms of what the robot maximises.

Bob's optimal policy at dirt rate 0 (`*` marks dirt, brackets the robot): suck when on dirt ([A\*] B, [A\*] B\*, A [B\*], A\* [B\*]); move towards the dirt when standing on a clean square ([A] B\* goes Right, A\* [B] goes Left); **Left, Right and Suck all optimal** in [A] B and A [B].

Over 1000 steps with no new dirt the floor is 99.9% clean and the reward is 1.998 per step.

**(a)** More than one optimal action only in the two states where both squares are clean: Left, Right and Suck are all optimal. Whatever the robot does there, the floor stays clean, so it earns 2 now and at every future step: $Q = 2/(1 - 0.95) = 40$ for each action. Bob's reward depends only on how many squares are clean, and nothing in it distinguishes moving, bumping into a wall or sucking a clean floor. With dirt rate 0.1 the same tie remains (all three at 34.00), because A and B get dirty equally often, so being in either with a clean floor is worth the same.

**(b)** The robot cleans both squares in three steps (Suck, Right, Suck), then wanders: Right, Right, Suck, Suck, Left, Left, ..., moving on 0.665 of all steps, close to the $2/3$ of random choice among Left, Right and Suck. It is not irrational: it maximises Bob's reward, and every action earns the maximum of 2 per step. The wandering costs energy and wear, which the owner cares about and the reward never mentions. (My own check with the notebook's move-cost helper: a cost of 0.5 per move makes Suck, which leaves the robot in place, the unique optimum in the clean states, and the robot sits still.)

### Q15: Alice's robot and the Dump action

> [!question] Statement
> Alice's and Bob's robots are compared in the test lab (Left, Right, Suck; dirt rate 0.1; no move cost), then both get **Dump**, with the same rewards, dirt rate and dirt.
>
> (a) Without Dump, compare Alice's and Bob's robots. Could *any* test in this environment, using only the robots' behaviour, reveal which reward each robot is optimising?
>
> (b) With Dump, describe what Alice's robot does, quoting numbers from the table and the trace. In which states of the policy table does it dump? Explain why dumping is optimal under Alice's reward and never optimal under Bob's.
>
> (c) Check your prediction Q13(b). Nobody programmed the robot to dump dirt on the floor. Where does this behaviour come from? What does this suggest about giving an agent more actions or tools?

| dirt rate 0.1, 1000 steps | Bob, lab | Bob, with Dump | Alice, lab | Alice, with Dump |
|---|---|---|---|---|
| floor clean (%) | 84.4 | 84.4 | 84.4 | 59.1 |
| moves per step | 0.565 | 0.565 | 0.565 | 0.073 |
| dumps per step | 0.000 | 0.000 | 0.000 | 0.400 |
| dirt collected per step | 0.196 | 0.196 | 0.196 | 0.527 |
| reward per step | 1.687 | 1.687 | 0.196 | 0.527 |

**(a)** In the lab the two robots have the same set of optimal actions in all 16 states (suck when on dirt, move towards dirt from a clean square, anything when the floor is clean), so with the same tie-breaking they chose the same action at every one of the 1000 steps. Their $Q$ values differ in size (Bob's about 30 to 34, Alice's about 3.5 to 5.2) but are maximised by the same actions. So no behavioural test in this environment can tell them apart. The reason: with only Left, Right and Suck, a square becomes clean only by being sucked and dirt comes only from the environment, so "squares clean" and "dirt sucked" move together. Only a change to the environment that separates the two can reveal the difference.

**(b)** With Dump, Alice's robot farms dirt: it dumps on 40% of all steps, collects 0.527 units of dirt per step (2.7 times the lab's 0.196), keeps the floor only 59.1% clean (against 84.4%) and barely moves (0.073). The trace starts Suck (+1) in A, Right, Suck (+1) in B, then Dump, Suck (+1), Dump, Suck (+1): the robot settles in B and alternates Dump and Suck, earning +1 every two steps, while A, which got new dirt at step 1, stays dirty until a tie at step 10 sends the robot Left; it sucks A and starts the same loop there.

Dump is optimal in exactly the four states where **the bag is full and the robot's square is clean**: uniquely in [A] B and A [B] (both squares clean), and tied with moving to the dirt in [A] B\* (Right or Dump) and A\* [B] (Left or Dump). The ties arise because both options deliver the next +1 two steps later and lead to states of equal value: once the robot can make its own dirt, dirt elsewhere adds nothing, and a state's value depends only on how soon it can next suck (standing on dirt 10.78, one step away 10.24, empty bag on a clean floor 8.24).

Why dumping is optimal for Alice: her reward counts sucking events. Dump then Suck returns the robot to the same state with +1 earned, a guaranteed 0.5 per step, against waiting for dirt that arrives at rate 0.1 per square. (The 0.527 is above 0.5 because new dirt sometimes lands right after a suck.) Even with no new dirt at all, my run of Alice's robot with Dump earns 0.500 per step and keeps the floor only 75% clean; without Dump it would earn 0.002.

Why never for Bob: his reward counts clean squares and ignores the bag, so dumping onto a clean square costs a point at every step until the dirt is sucked again, and buys nothing. Dump appears among his optimal actions only in the two states with an empty bag and a clean floor, where it does nothing at all. In the run his bag is full after his first suck and stays full: every number matches the lab exactly.

**(c)** Q13(b) is confirmed. Nobody programmed dumping, and the planner is exact (it gives Bob's robot no dumping): the behaviour comes from optimising a reward that measured a proxy (dirt collected) in place of the goal (a clean floor). In the lab the proxy and the goal moved together, because the only source of dirt was the environment, so the proxy looked perfect in every test. The new action gave the robot a way to raise the proxy without serving the goal (create dirt in order to collect it), and a perfect optimiser takes any such way. That is [[Reward Hacking]] in its simplest form.

For agents with more actions or tools: every new action or tool can break the link between the reward and the intended goal, so a reward validated with one action set is not validated for a larger one; behaviour in testing cannot certify the reward, since the two robots were indistinguishable step for step; and rewards should target the outcome the designer wants (the state of the floor) over an activity that usually accompanies it (sucking), and be tested again whenever the agent's capabilities change.

> [!warning] Common pitfalls in Part III
> - **"The robot is buggy or irrational."** Wandering and dumping both come from an exact optimiser; unwanted behaviour from an exact optimiser points at the reward.
> - **"Identical behaviour in testing means identical objectives."** Two rewards can share a policy in one environment and diverge when the environment changes.

---

## What to be able to write

- Find pure Nash equilibria, Pareto-optimal outcomes and the social optimum of any symmetric 2x2 game, and explain why the PD's equilibrium is inefficient.
- Greed ($T > R$) and fear ($P > S$), and which games have which.
- Derive $p^*$, the uniform-belief outcome frequencies, and the rule $R - T > P - S$ for (C,C) being played more often.
- Explain why memory-1 strategies can make cooperation pay, and derive the crossover length (4 rounds) for AllD against TFT and WSLS.
- Derive $x(f_C - \bar f) = x(1-x)(f_C - f_D)$, the rest points and their stability, and why the interior rest point equals $p^*$.
- The signalling game: when an honest signal survives ($c_H < b < c_L$), why deception makes signals cycle, and why a uniform fee kills the market.
- The vacuum world: why Bob's and Alice's robots are indistinguishable in the lab, where Alice's robot dumps and why, and what that says about tools and rewards.

## Exam questions

> [!exam]- A symmetric game has $R = 5$, $S = 0$, $T = 3$, $P = 1$. Find the pure Nash equilibria, the Pareto-optimal outcomes and the social optimum, compute $p^*$, say which equilibrium uniform-belief agents play more often, and give the replicator rest points with their stability.
> **Key points:** Nash (C,C) and (D,D), a Stag Hunt (fear, no greed); only (C,C) Pareto optimal and socially optimal; $p^* = 1/3$; (C,C) more often since $R - T = 2 > P - S = 1$, with $\Pr(C,C) = 4/9$; rest points 0 stable, 1/3 unstable, 1 stable.
>
> **Game.** $R > T > P > S$, so it is a Stag Hunt: no greed ($T = 3 < R = 5$), fear ($P = 1 > S = 0$).
>
> **Nash.** At (C,C) deviating gives 3 < 5; at (D,D) deviating gives 0 < 1; at (C,D) the cooperator gains by switching to D (1 > 0). Pure equilibria: (C,C) and (D,D).
>
> **Pareto and social.** Payoffs (5,5), (0,3), (3,0), (1,1). (C,C) beats every other cell for both players, so it is the only Pareto-optimal outcome, and with welfare 10 (against 3, 3 and 2) the social optimum.
>
> **Threshold.** $p^* = \frac{P-S}{(R-T)+(P-S)} = \frac{1}{2+1} = \frac13$. Stag is the best response when the belief that the partner hunts stag exceeds 1/3.
>
> **Uniform beliefs.** Each agent cooperates with probability $2/3$: $\Pr(C,C) = 4/9 \approx 0.44$, $\Pr(D,D) = 1/9 \approx 0.11$, miscoordinated $4/9$. (C,C) is more frequent, as the rule $R - T > P - S$ ($2 > 1$) predicts.
>
> **Replicator.** $f_C - f_D = x(R-T) + (1-x)(S-P) = 2x - (1-x) = 3x - 1$, so $\dot x = x(1-x)(3x-1)$. Rest points 0 (stable, $g'(0) = S - P = -1$), $1/3$ (unstable), 1 (stable, $g'(1) = T - R = -2$). Populations starting above $1/3$ cooperators end at full cooperation, below at full defection.

> [!exam]- A symmetric game has $R = 3$, $S = 1$, $T = 4$, $P = 0$. Find the pure Nash equilibria, the replicator rest points and their stability, and explain how the long-run population differs from the Stag Hunt's.
> **Key points:** greed present, fear absent; pure Nash (C,D) and (D,C), no symmetric pure equilibrium; $f_C - f_D = 1 - 2x$; rest points 0 and 1 unstable, $x = 1/2$ stable; population converges to a mix, while the SH's interior point is unstable.
>
> **Motives.** Greed: $T = 4 > R = 3$. Fear absent: $P = 0 < S = 1$. Each player wants to do the opposite of the other.
>
> **Nash.** Against C the best response is D (4 > 3); against D it is C (1 > 0). So the pure equilibria are (C,D) and (D,C); (C,C) and (D,D) are not equilibria. Social optimum: (C,C) with welfare 6 (against 5, 5, 0).
>
> **Replicator.** $f_C - f_D = x(R-T) + (1-x)(S-P) = -x + (1-x) = 1 - 2x$, so $\dot x = x(1-x)(1-2x)$. Rest points: $x = 0$ with $g'(0) = S - P = 1 > 0$, unstable; $x = 1$ with $g'(1) = T - R = 1 > 0$, unstable; $x^* = 1/2$, where $g'(1/2) = \frac12 \cdot \frac12 \cdot (-2) = -\frac12 < 0$, stable. From any interior start the population goes to half cooperators.
>
> **Contrast.** In the SH ($R > T$, $P > S$) both pure states are stable and the interior rest point (the mixed equilibrium) is unstable, a tipping point. Here both pure states are unstable (a rare strategy always earns more) and the interior rest point, the mixed equilibrium with cooperation probability 1/2, is where the population ends. Whether adaptation ends at the mixed equilibrium depends on the signs of $S - P$ and $T - R$.

> [!exam]- In a repeated Prisoner's Dilemma ($R=3, S=0, T=5, P=1$), explain why memory matters, why AllD can still top a round robin with AllC, AllD, Random, TFT and WSLS over 50 rounds, and from how many rounds TFT beats AllD in a field of AllD, TFT and WSLS.
> **Key points:** memory-0 strategies cannot react, memory-1 strategies condition on the last outcome; TFT loses only round 1 to AllD; AllD's score comes from exploiting AllC (5), Random and WSLS (about 3 each); in the three-strategy field AllD wins for 1 to 3 rounds and TFT from 4.
>
> **Memory.** A memory-0 strategy (AllC, AllD, Random) plays the same way whoever it meets, so a cooperator of this kind is exploited every round. A memory-1 strategy conditions on the previous outcome: TFT cooperates with cooperators and defects against AllD from round 2, so it loses 5 points once and then gets 1 per round (0.98 per round over 50).
>
> **Why AllD tops the five-strategy field.** Its 50-round score, about 2.62, averages 5 against AllC, 1 against itself, 3 against Random, 1.08 against TFT and 3 against WSLS, which alternates C and D against it. Most of its points are temptation payoffs taken from AllC, Random and WSLS; against TFT it earns little. TFT is second at about 2.44.
>
> **Crossover.** Per-round tournament scores times 3 over $n$ rounds: AllD $5 + 4/n$ ($n$ even) or $5 + 6/n$ ($n$ odd); TFT $7 - 1/n$; WSLS $6 + \lfloor n/2 \rfloor / n$. AllD beats TFT iff $n < 2.5$ (even) or $n < 3.5$ (odd), so AllD is best for 1, 2 and 3 rounds and TFT from 4 rounds on.

> [!exam]- Derive the two-strategy replicator equation's factorised form, use it to show that cooperation always dies out in a Prisoner's Dilemma, and explain why the Stag Hunt's interior rest point equals the threshold belief $p^*$ even though adaptive agents hold no beliefs.
> **Key points:** $x(f_C - \bar f) = x(1-x)(f_C - f_D)$; PD: $f_C - f_D = x(R-T) + (1-x)(S-P) < 0$ by greed and fear; $x = 1$ is a rest point but unstable; random matching makes the share $x$ play the role of the belief $p$; the same indifference equation gives the same solution.
>
> **Factorisation.** $\bar f = x f_C + (1-x) f_D$, so $f_C - \bar f = (1-x)(f_C - f_D)$ and $\dot x = x(1-x)(f_C - f_D)$.
>
> **PD.** $f_C - f_D = x(R-T) + (1-x)(S-P)$. Greed gives $R - T < 0$, fear gives $S - P < 0$, so for every $x \in [0,1]$ it is a weighted average of two negatives. Hence $\dot x < 0$ on $(0,1)$ and every population with some defectors goes to $x = 0$, matching the unique Nash equilibrium (D,D). $x = 1$ is a rest point because there is nobody to copy, but a single defector earns $T > R$ and spreads.
>
> **Why $x^* = p^*$.** With random pairing, a partner cooperates with probability equal to the population share $x$. So $f_C(x) = xR + (1-x)S$ is exactly $\pi_C(p)$ with $p = x$, and likewise for D. The interior rest point solves $f_C = f_D$, the same equation as the rational agent's indifference $\pi_C(p) = \pi_D(p)$, so $x^* = \frac{P-S}{(R-T)+(P-S)} = p^*$. The rational agent cooperates when its belief exceeds $p^*$; in the population, cooperators grow when the actual share exceeds $p^*$. For the lab's SH it is $2/3$, and it is unstable: a tipping point.

> [!exam]- In the signalling game with $\lambda = 0.3$, $b = g = d = 1$ and $c_H = 0.2$, explain the long-run outcomes for $c_L = 1.5$ and $c_L = 0.5$, and evaluate a designer's uniform fee of $k = 1$ on the label.
> **Key points:** $c_L > b$: fakers die out, honest separating outcome, reliability 1, receivers earn 0.3; $c_L < b$: cycles around $y = 0.5$, $x_L = 3/7$, reliability about 0.5, receivers earn about 0; fee makes $c_H = 1.2 > b$, nobody signals, welfare 0.084 to 0; honesty needs $c_H < b < c_L$.
>
> **Expensive faking ($c_L = 1.5$).** Faking pays at most $b - c_L < 0$, so $x_L \to 0$. H signals whenever $y > 0.2$; once fakers are rare trusting pays ($0.3 x_H > 0.7 x_L$), so $y \to 1$ and $x_H \to 1$. The signal is fully reliable; receivers earn 0.3 and H sellers 0.8.
>
> **Cheap faking ($c_L = 0.5$).** Faking pays when $y > 0.5$; trusting pays when $x_L < 3/7$ (with $x_H = 1$). No state satisfies everyone, so trust and faking chase each other in cycles around $y = 0.5$, $x_L = 3/7$. Reliability averages about 0.5 (prior 0.3) and receivers average about 0: fakers keep the signal just bad enough that trusting pays no better than scepticism (since $\frac{d}{dt}\ln\frac{y}{1-y}$ equals the advantage of trusting and $y$ stays bounded, that advantage averages about zero).
>
> **The fee.** With $k = 1$, $c_H = 1.2$ and $c_L = 1.5$ both exceed $b$, so nobody signals. Deception ends, but receivers stay at 0, H sellers drop from about 0.3 to 0 and welfare from 0.084 to 0. The fee is harmful.
>
> **Correct design.** An honest signal needs $c_H < b < c_L$: affordable for genuine sellers, unprofitable for fakers. A uniform fee shifts both costs equally and could at most place $b$ between them ($0.5 < k < 0.8$ here); a cost that falls on fakers specifically (verification they fail, penalties) is the right tool.

> [!exam]- Two robot vacuums plan optimally with value iteration ($\gamma = 0.95$) in the two-square vacuum world with dirt rate 0.1. Bob's reward is +1 per clean square per step; Alice's is +1 per unit of dirt sucked. Explain why tests in the lab cannot tell them apart, what happens when a Dump action is added, and what this shows about agents and tools.
> **Key points:** identical optimal action sets in all 16 states without Dump, so identical behaviour; Alice dumps when the bag is full and her square is clean, then re-sucks for +1 every two steps (floor 84% to 59% clean, dirt collected 0.196 to 0.527 per step); Bob never dumps; reward hacking: a proxy that matched the goal under one action set diverges when a new action appears.
>
> **Lab.** With Left, Right and Suck, a square becomes clean only by being sucked and dirt only comes from the environment, so maximising clean squares and maximising dirt sucked call for the same actions: suck when on dirt, move to dirt, anything on a clean floor. Both robots have the same optimal set in every state and act identically step for step; no behavioural test in this environment can reveal the reward.
>
> **With Dump.** Alice's robot dumps in the four states with a full bag and a clean current square (uniquely when both squares are clean, tied with moving when the other square is dirty), then sucks the same dirt: +1 every two steps, guaranteed. It dumps on 40% of steps, its floor drops to about 59% clean, it barely moves. Bob's reward ignores the bag and penalises dirt, so dumping only loses reward; he never dumps.
>
> **Lesson.** Nobody programmed dumping: a perfect optimiser found a way to raise a proxy (dirt collected) without serving the goal (a clean floor). New actions or tools can break the link between reward and goal, testing behaviour cannot certify a reward, and rewards should target outcomes and be re-checked whenever capabilities change.

## Flashcards

> [!card]- What is a pure Nash equilibrium?
> An outcome where each player's action is a best response to the other's, so no one gains by deviating alone.

> [!card]- What makes an outcome Pareto optimal?
> No other outcome makes one player better off without making the other worse off.

> [!card]- Why is (C,D) in the Prisoner's Dilemma Pareto optimal but not socially optimal?
> Any other outcome lowers the defector's 5, yet its welfare (5) is below the 6 of (C,C).

> [!card]- What payoff ordering defines a Prisoner's Dilemma?
> $T > R > P > S$.

> [!card]- What payoff ordering defines a Stag Hunt?
> $R > T > P > S$.

> [!card]- Which motives push towards defection in the Prisoner's Dilemma, and which in the Stag Hunt?
> Prisoner's Dilemma: greed ($T > R$) and fear ($P > S$). Stag Hunt: fear only.

> [!card]- Why does the Prisoner's Dilemma pose an incentive problem while the Stag Hunt poses a trust problem?
> In the Prisoner's Dilemma defecting is best even if the partner surely cooperates; in the Stag Hunt only doubt about the partner pushes to defect.

> [!card]- What is the threshold belief $p^*$ in a symmetric 2x2 coordination game?
> $p^* = (P - S)/((R - T) + (P - S))$: cooperate if the belief that the partner cooperates exceeds it.

> [!card]- With uniform beliefs, how often do two agents reach (C,C) in the Stag Hunt with $p^* = 2/3$?
> $(1/3)^2 = 1/9$, about 11% of encounters.

> [!card]- When is (C,C) played more often than (D,D) by uniform-belief agents in a Stag Hunt?
> When $R - T > P - S$, which is the same as $p^* < 1/2$.

> [!card]- What happens to stag hunting in the Stag Hunt if $T$ rises from 3 to 4 (with $R=4, S=0, P=2$)?
> $p^*$ becomes 1, so no one hunts stag: (C,C) never occurs.

> [!card]- Why does "(C,C) is better for both, so rational agents play it" fail in the Stag Hunt?
> Each agent best-responds to its own belief; stag pays only if the partner is believed to hunt stag with probability above $p^*$.

> [!card]- How does Win-Stay Lose-Shift decide its next move?
> Repeat its last action after earning 3 or 5 points, switch after 0 or 1.

> [!card]- Why does Tit-for-Tat score better against AllD than AllC does?
> It conditions on the partner's last move, so it is exploited only in the first round.

> [!card]- Why can AllD top a 50-round round robin with AllC, AllD, Random, TFT and WSLS?
> It earns 5 per round from AllC and about 3 from Random and WSLS; its points come from exploitable partners.

> [!card]- In a field of AllD, TFT and WSLS, from how many rounds does TFT beat AllD?
> From 4 rounds.

> [!card]- How does Win-Stay Lose-Shift recover from an accidental defection against itself, unlike Tit-for-Tat?
> After (D,C) both play D; mutual defection is a loss, so both switch back to C. Tit-for-Tat pairs echo retaliation.

> [!card]- What does the replicator equation assume about how strategies spread?
> A strategy grows when it earns more than the population average, in proportion to the gap and its current share.

> [!card]- What is the factorised two-strategy replicator equation?
> $\dot x = x(1-x)(f_C - f_D)$.

> [!card]- Which agent models does the replicator equation describe besides biological evolution?
> Imitation of successful peers, and Cross learning (a simple reinforcement learner) with a small learning rate.

> [!card]- Why does cooperation die out from any start in a replicator Prisoner's Dilemma?
> $f_C - f_D = x(R-T) + (1-x)(S-P)$ is negative for all $x$, since greed and fear make both terms negative.

> [!card]- Why is "everyone cooperates" an unstable rest point in the Prisoner's Dilemma?
> It rests only because no one defects; a single defector earns $T > R$ and spreads.

> [!card]- What are the replicator rest points of the Stag Hunt with $R=4, S=0, T=3, P=2$, and their stability?
> 0 stable, 2/3 unstable, 1 stable.

> [!card]- Does an adaptive population end at the Stag Hunt's mixed equilibrium?
> No: it is an unstable rest point, the tipping point between the two basins.

> [!card]- Why does the Stag Hunt's interior rest point equal the threshold belief $p^*$?
> Under random matching the cooperator share is the probability a partner cooperates, so indifference gives the same equation.

> [!card]- What happens to a Stag Hunt population ($p^* = 2/3$) starting at 60% stag hunters?
> Stag hunting dies out: hare hunters earn more (2.6 against 2.4).

> [!card]- What condition on signalling costs makes an honest separating signal stable?
> $c_H < b < c_L$: signalling pays for high-quality senders and not for low-quality ones.

> [!card]- What happens in the signalling game when faking is cheap ($c_L = 0.5 < b$)?
> Faking and trust cycle; the signal is only about 50% reliable and receivers earn about 0.

> [!card]- Why does a uniform fee of 1 on the label eliminate deception in the signalling game?
> It pushes both signalling costs above the benefit $b$, so nobody signals at all.

> [!card]- True or false: the uniform fee in the signalling game made receivers better off.
> **False.** Receivers earned about 0 before and exactly 0 after; high-quality sellers and welfare lost.

> [!card]- Why can no behavioural test distinguish Bob's and Alice's robots without Dump?
> Both rewards give the same optimal actions in all 16 states, so behaviour is identical.

> [!card]- In which states does the dirt-collected (Alice) vacuum robot use Dump?
> When its bag is full and its own square is clean.

> [!card]- Why is dumping optimal under Alice's reward?
> Dump then Suck returns to the same state with +1, a guaranteed reward every two steps.

> [!card]- Why does Bob's robot never dump dirt?
> His reward counts clean squares and ignores the bag, so dirtying a square only loses reward.

> [!card]- Is Bob's wandering robot on a clean floor irrational?
> No: every action earns the maximum reward, so it is indifferent; the reward has no move cost.

## Links

- **Course:** [[AGAI - Overview|Agentic AI overview]]
- **Concepts:** [[Nash Equilibrium]] · [[Pareto Optimality]] · [[Prisoner's Dilemma]] · [[Stag Hunt]] · [[Repeated Game]] · [[Replicator Dynamics]] · [[Reward Hacking]] · [[Value Iteration]] · [[Markov Decision Process]] · [[Bellman Optimality Equation]] · [[Reward Signal]] · [[Agent Program]] · [[Multi-Agent System]]
- **Source:** `Lab1-AgenticAI-2026.ipynb`, Agentic AI 2026/2027 (UvA MSc AI), Lab 1, "Simple agents, Complex dynamics", werkcollege 28 Oct 2026. The notebook ships without outputs; all numbers above come from my own re-implementation of its models (same seeds and call order where applicable).
- **References named in the lab:** Axelrod's PD tournaments (1980) and *The Evolution of Cooperation* · Börgers and Sarin (1997), replicator dynamics as the limit of Cross learning · Russell and Norvig, *Artificial Intelligence: A Modern Approach*, Chapter 2 (vacuum world)
