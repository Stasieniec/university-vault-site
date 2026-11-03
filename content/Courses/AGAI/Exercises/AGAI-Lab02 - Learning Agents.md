---
type: exercise
course: 5204AGAI6Y
week: 2
source: "Lab 2 notebook (werkcollege, 4 Nov 2026)"
concepts:
  - "[[Q-Learning]]"
  - "[[Independent Learning]]"
  - "[[Stag Hunt]]"
  - "[[Prisoner's Dilemma]]"
  - "[[Nash Equilibrium]]"
  - "[[Epsilon-Greedy Policy]]"
  - "[[Exploration vs Exploitation]]"
  - "[[Social Dilemma]]"
  - "[[Tabular RL]]"
  - "[[Function Approximation]]"
  - "[[Deep Reinforcement Learning]]"
  - "[[PPO]]"
  - "[[Actor-Critic]]"
status: complete
---

# AGAI-Lab02: Learning Agents

> [!abstract] In short
> Agents now learn from their own rewards by trial and error, each independently, and the lab asks whether such learners reach a good equilibrium and whether a more powerful learner helps. In the [[Stag Hunt]], two independent [[Q-Learning|Q-learners]] mostly lock into the safe (D,D): initial values decide the equilibrium, and since one agent's exploration is noise in the other's rewards, exploration knocks them out of (C,C) but almost never out of (D,D). In the Coin Game, a grid-world social dilemma, tabular Q-learners learn to collect coins but never to leave the other's alone, so their total reward stays near that of random agents. Irrelevant inputs blow up the Q-table and ruin learning; a neural-network learner ([[PPO]]) shrugs them off but still defects. Function approximation fixes generalisation, and the conflict of incentives remains.
>
> The lab is ungraded practice that the lecturer calls the best preparation for the midterm. The notebook ships without outputs, so simulation numbers come from my own re-implementation of its models with the lab's parameters (own random numbers, own PPO code); they show the expected behaviour but can differ in detail from a classroom run. Analytic results are exact.

**Concepts tested:** [[Q-Learning]], [[Independent Learning]], [[Epsilon-Greedy Policy]], [[Exploration vs Exploitation]], [[Stag Hunt]], [[Prisoner's Dilemma]], [[Nash Equilibrium]], [[Social Dilemma]], [[Tabular RL]], [[Function Approximation]], [[PPO]], [[Actor-Critic]], [[REINFORCE]]

---

## Part I: independent Q-learning in the Stag Hunt

### The model

The game is Lab 1's [[Stag Hunt]] with $R=4$ (both hunt stag), $S=0$ (I hunt stag alone), $T=3$ (I hunt hare while the other hunts stag) and $P=2$ (both hunt hare). C is stag, D is hare. It has two pure [[Nash Equilibrium|Nash equilibria]], (C,C) and (D,D), and Lab 1's threshold belief is

$$p^* = \frac{P - S}{(R - T) + (P - S)} = \frac{2}{1 + 2} = \frac23:$$

stag is the best response only if the partner hunts stag with probability above 2/3. Two Lab 1 benchmarks come back: rational agents with uniformly random beliefs reached (C,C) in 11% of encounters, and under replicator dynamics a population went to all-stag only from starting shares above 2/3, so from one third of the starting states.

**The learners.** The stage game repeats unchanged and nothing is remembered, so there is a single state. Each agent $i$ keeps two numbers, $Q_i(C)$ and $Q_i(D)$, its estimates of what each action earns, and uses stateless Q-learning:

$$Q_i(a_i) \leftarrow Q_i(a_i) + \alpha\,\big(r_i - Q_i(a_i)\big)$$

where:
- $a_i$ is the action agent $i$ just played, and only its value is updated
- $r_i$ is the payoff it just received
- $\alpha$ is the **learning rate**, the fraction of the prediction error $r_i - Q_i(a_i)$ that is corrected (0.1 throughout)

Regular [[Q-Learning]] learns $Q(s,a)$ per state and adds a look-ahead term to the target, $r + \gamma \max_{a'} Q(s', a')$, which credits an action for the state it leads to ($\gamma$ is the [[Discount Factor]]). With one state, $\max_{a'} Q(s', a')$ is the same number whichever action was played, so it shifts both targets equally and never changes which action looks better. Dropping it (equivalently $\gamma = 0$) leaves the rule above.

Actions are [[Epsilon-Greedy Policy|$\varepsilon$-greedy]]: with probability $\varepsilon$ a uniformly random action, otherwise the action with the larger $Q$-value, ties broken at random. So an agent whose greedy action is C plays D with probability $\varepsilon/2$.

Each agent learns from its own action and reward only and treats the partner as part of the environment: [[Independent Learning]] (Lecture 3, [[AGAI-L03 - Intro to Multi-Agent Reinforcement Learning]]). Because the partner is learning too, the payoff of each action keeps changing: a moving target.

**What is measured.** Each experiment runs 500 independent pairs for 3000 rounds. The outcome of a run is its **greedy joint action**, what the two would play now without exploration: (C,C), (D,D) or miscoordinated. An agent counts as a stag hunter only if $Q(C) > Q(D)$ strictly, so a tie counts as hare.

### Q1: predictions before the runs

> [!question] Statement
> (a) Two independent Q-learners play the Stag Hunt ($R=4, S=0, T=3, P=2$) 3000 times in a row. They start knowing nothing (all value estimates equal to 0), and choose a random action (explore) 10% of the time. The experiment is repeated 500 times. In what fraction of the runs do you expect them to end at (C,C)? Choose one: less than 10%, 10-40%, 40-70%, more than 70%.
>
> (b) Now suppose both agents start *at* (C,C): they already hunt stag and believe that this pays off. Will they keep hunting stag if they explore 10% of the time? And if they explore 50% of the time?

**(a)** Agents that know nothing start by playing at random, which is like holding the belief $p = 1/2 < p^*$. Against a coin-flipping partner D earns more than C (2.5 against 2, Q2(b)), and the replicator benchmark puts $x = 1/2$ in the basin of all-hare. So (C,C) should be rare: **less than 10%**. My runs give 5% to 10% across seeds (mean 6.6% over ten).

**(b)** A rational agent would stay: against a partner who explores 10% of the time, C earns 3.8 and D 2.95 (Q4). The learners mostly do not: at $\varepsilon = 0.1$ about 80% of runs leave (C,C) within 3000 rounds, at $\varepsilon = 0.5$ almost all within a few dozen rounds.

### Q2: the baseline run

> [!question] Statement
> Both agents start with $Q = (0, 0)$, $\alpha = 0.1$, $\varepsilon = 0.1$, in the Stag Hunt and, as a sanity check, in the Prisoner's Dilemma ($R=3, S=0, T=5, P=1$).
>
> (a) In what fraction of the runs did the learners end at each outcome? Compare with Lab 1, where rational agents with uniform beliefs reached (C,C) in 11% of the encounters and the replicator dynamics reached all-C from 1/3 of the initial states. Did learning help coordination?
>
> (b) Early in learning, which action earns more for the row agent, and why? Hint: what does the partner do in the first rounds, and what do C and D earn against such a partner? Explain why the learners lock into (D,D) so quickly.
>
> (c) In the Prisoner's Dilemma the learners always end at (D,D). Is this a coordination failure in the same sense as in the Stag Hunt? Use the notion of a dominant action and/or Nash equilibrium to answer.

Share of runs per greedy outcome after a given round (one run of 500 pairs; miscoordination is at most 1% in the Stag Hunt and 7% in the Prisoner's Dilemma):

| round | 1 | 100 | 500 | 1000 | 3000 |
|---|---|---|---|---|---|
| SH (C,C) | 0.28 | 0.29 | 0.29 | 0.24 | 0.10 |
| SH (D,D) | 0.72 | 0.71 | 0.71 | 0.76 | 0.90 |
| PD (D,D) | 0.72 | 0.78 | 1.00 | 1.00 | 1.00 |

**(a)** At round 3000 the Stag Hunt learners end at (C,C) in 5% to 10% of runs (across seeds) and at (D,D) in the rest; the Prisoner's Dilemma learners end at (D,D) in every run. Learning did not help: the learners do worse than Lab 1's rational agents (11%) and far worse than the replicator dynamics (one third). In round 1 about 28% of runs settle at (C,C) and the rest at (D,D); after that exploration slowly knocks (C,C) runs over to (D,D) (Q4), and none come back.

**(b)** In the first rounds all $Q$-values are 0, so both agents break ties at random: the partner is a coin flip. Against it

$$\mathbb{E}[r \mid C] = \tfrac12 \cdot 4 + \tfrac12 \cdot 0 = 2, \qquad \mathbb{E}[r \mid D] = \tfrac12 \cdot 3 + \tfrac12 \cdot 2 = 2.5.$$

D earns more on average, and, more important for a learner, never less than 2, while C earns 0 half the time. Round 1 in detail:

- Both played C: both earned 4, $Q(C) = 0.4 > 0$, so both now prefer C. Probability 1/4.
- Both played D: both earned 2, $Q(D) = 0.2 > 0$, so both prefer D.
- One of each: the D player earned 3 and prefers D. The C player earned 0, so it is still tied; as soon as its tie-break picks D it earns 2 and prefers D.

So a run reaches (C,C) essentially only if both play C in round 1, probability 1/4 (28% in my run; a few mixed runs also meet at C in round 2 when the D player explores). Lock-in is quick because of a feedback loop: once my partner plays D, C pays me 0, so I never see evidence that C could pay 4, and my D in turn makes C pay 0 for my partner. $Q(C)$ can only rise when both play C in the same round, which during exploration happens in $(\varepsilon/2)^2 = 0.25\%$ of rounds.

**(c)** No. In the Prisoner's Dilemma D is dominant ($T = 5 > R = 3$ and $P = 1 > S = 0$), so (D,D) is the unique Nash equilibrium. There was no other equilibrium to coordinate on: the learners found the only one, and their values reflect it (final average $Q(D) \approx 1.2 = 0.95 \cdot 1 + 0.05 \cdot 5$). The bad outcome is a [[Social Dilemma]], an incentive problem: (C,C) Pareto dominates (D,D), yet each agent does better by defecting whatever the other does. In the Stag Hunt (C,C) is itself an equilibrium, which the learners could have reached and mostly missed: that is a coordination failure.

### Q3: initial values as beliefs

> [!question] Statement
> An agent that believes its partner hunts stag with probability $b_i$ starts with $Q_i(C) = b_i R + (1-b_i)S$ and $Q_i(D) = b_i T + (1-b_i)P$. Three pairs of beliefs are compared: both optimistic (0.8, 0.8), both pessimistic (0.4, 0.4), and an optimist meeting a sceptic (0.9, 0.5), with $\varepsilon = 0.01$.
>
> (a) Which pairs of initial beliefs lead to (C,C)? Compare with the threshold $p^* = 2/3$ of Lab 1.
>
> (b) An optimist (belief 0.9) meets a sceptic (belief 0.5). Which equilibrium do they reach? Who adapts to whom, and how many rounds does it take? Use the update rule: what happens to the optimist's $Q(C)$ in the first rounds?
>
> (c) How would these results change if you increase $\varepsilon$, and why? Test it with $\varepsilon = 0.1$.

| belief $b$ | $Q(C)$ | $Q(D)$ | $Q(C) - Q(D)$ | starts with |
|---|---|---|---|---|
| 0.9 | 3.6 | 2.9 | +0.7 | C |
| 0.8 | 3.2 | 2.8 | +0.4 | C |
| 0.5 | 2.0 | 2.5 | −0.5 | D |
| 0.4 | 1.6 | 2.4 | −0.8 | D |

**(a)** Both optimistic ends at (C,C) in 100% of runs, both pessimistic at (D,D) in 100%, and the optimist with the sceptic at (D,D) in 99% to 99.8%. The link to $p^*$ is exact:

$$Q(C) - Q(D) = b(R - T) - (1 - b)(P - S) = b - 2(1-b) = 3b - 2 > 0 \iff b > \tfrac23 = p^*.$$

So an agent's first greedy action is Lab 1's best response to its belief. When both start on the same side, play confirms the beliefs (two stag hunters earn 4, two hare hunters earn 2 and barely try C), so with little exploration equilibrium selection moves into the initialisation.

**(b)** They reach (D,D): **the optimist adapts to the sceptic**, and the sceptic never moves. With exploration switched off the update rule gives this trace. C earns the optimist 0 against the sceptic's D, so each C play multiplies its $Q(C)$ by $1 - \alpha = 0.9$.

| round | optimist | its $Q(C)$, $Q(D)$ | sceptic | its $Q(C)$, $Q(D)$ |
|---|---|---|---|---|
| 1 | C, earns 0 | 3.24, 2.90 | D, earns 3 | 2.00, 2.55 |
| 2 | C | 2.916, 2.90 | D | 2.00, 2.595 |
| 3 | C | 2.624, 2.90 | D | 2.00, 2.636 |
| 4 | D, earns 2 | 2.624, 2.81 | D, earns 2 | 2.00, 2.572 |
| 7 | D | 2.624, 2.590 | D | 2.00, 2.417 |
| 8 | C again | 2.362, 2.590 | D | 2.00, 2.475 |

The optimist hunts stag for three rounds and switches in round 4: $3.6 \cdot 0.9^3 = 2.62 < 2.9$, while $3.6 \cdot 0.9^2 = 2.92$ is still above. The switch is not yet final. At (D,D) its $Q(D)$ falls towards $P = 2$ while its $Q(C)$ is stale (not updated while it plays D) and still above 2, so it relapses to C in rounds 8, 14 and 25, each time earning 0 and losing another 10%. After the sixth stag hunt $Q(C) = 3.6 \cdot 0.9^6 = 1.91 < 2$, below anything $Q(D)$ can reach, and from round 26 the pair stays at (D,D).

The sceptic never plays C, so its $Q(C) = 2.0$ is never updated, and its D earns 3 or 2, which keeps $Q(D)$ between 2 and 3 and so above $Q(C)$. At $\varepsilon = 0.01$ exploration is too rare to show it that stag pays 4 before the optimist gives up. The agent whose current action is punished does the adapting.

**(c)** With $\varepsilon = 0.1$, the two optimists end at (C,C) in only 18% to 22% of runs (against 100%), while the other two pairs stay at (D,D) in at least 99%. More exploration does not rescue the pairs at (D,D) and destroys most of the optimists' (C,C): the partner's exploratory D gives a stag hunter 0, which hits $Q(C)$ hard, while a hare hunter loses little when the partner explores (Q4).

### Q4: exploration as noise in the other's rewards

> [!question] Statement
> Both agents start at (C,C) with the values of belief $b = 0.9$, $Q = (3.6, 2.9)$, and $\alpha = 0.1$. The exploration rate is $\varepsilon = 0.01, 0.05, 0.1$ or $0.5$.
>
> (a) Both agents start at (C,C), which is a Nash equilibrium. For which settings does (C,C) survive 3000 rounds? Against a partner that explores 10% of the time, C is still the best response (expected payoff 3.8, against 2.95 for D): why do the learners nevertheless leave (C,C)? Hint: what does $Q(C)$ actually estimate when $\alpha = 0.1$?
>
> (b) Why is the opposite escape, from (D,D) to (C,C), much less likely?

Expected payoffs against an exploring partner at (C,C), and my runs (first exit: median round over runs that left):

| $\varepsilon$ | C pays | D pays | (C,C) at 3000 | first exit |
|---|---|---|---|---|
| 0.01 | 3.98 | 2.995 | 100% | almost none leave |
| 0.05 | 3.90 | 2.975 | 81% to 84% | round 1300 |
| 0.1 | 3.80 | 2.95 | 17% to 20% | round 480 |
| 0.5 | 3.00 | 2.75 | 0.2% to 0.4% | round 12 |

**(a)** (C,C) survives at $\varepsilon = 0.01$, mostly survives at 0.05, is mostly lost at 0.1 and entirely at 0.5, although C stays the best response in every row.

Unrolling the update shows what $Q(C)$ is:

$$Q_t(C) = \alpha \sum_{k \ge 0} (1-\alpha)^k\, r_{t-k} \quad (\text{plus a vanishing term from the initial value}),$$

where $r_{t-k}$ is the payoff of the $k$-th most recent C play: an **exponentially weighted average of the recent payoffs of C**. The weights $0.1, 0.09, 0.081, \dots$ halve about every 6.6 plays, so $Q(C)$ remembers roughly the last $1/\alpha = 10$ payoffs. With a constant learning rate it estimates "what C paid lately", a noisy quantity, and never settles on the expected value 3.8.

At $\varepsilon = 0.1$ the partner plays D in 5% of rounds, and each such round pays me 0 and cuts my $Q(C)$ by 10%. From 3.8, three zeros give $3.42 \to 3.08 \to 2.77$, below $Q(D) \approx 2.95$. Three zeros within a short stretch are enough, and over 3000 rounds, with each agent exposed to the other's exploration, such a streak is likely. For a payoff with variance $\sigma^2$ the weighted average has standard deviation $\sigma \sqrt{\alpha / (2 - \alpha)}$; with $\sigma^2 = 16 \cdot 0.05 \cdot 0.95 = 0.76$ that is 0.20 around 3.80 (my runs: mean 3.80, standard deviation 0.20, minimum about 2.0).

Once one agent's $Q(C)$ dips below its $Q(D)$, it plays D, and the partner earns 0 every round and follows within about three rounds. Both may retry stag a few times while their stale $Q(C)$ is above 2, as the optimist did in Q3(b), and some pairs re-coordinate, but each failed retry costs another 10% until (D,D) holds. At $\varepsilon = 0.5$ C leads by only 0.25 and a single zero ($3.0 \to 2.7 < 2.75$) is enough, so (C,C) falls within a dozen rounds.

So the learners leave a Nash equilibrium because one agent's exploration is noise in the other's rewards, and a constant learning rate makes each agent react to that noise as if the partner had changed.

**(b)** At (D,D) with $\varepsilon = 0.1$, C pays $0.05 \cdot 4 = 0.2$ on average and D pays $0.05 \cdot 3 + 0.95 \cdot 2 = 2.05$. For C to become greedy, $Q(C)$ must climb from about 0.2 to above 2:

- $Q(C)$ is updated only when I explore C (5% of rounds), and pays 4 only if the partner explores C in the same round (probability 0.05). The other 95% of my C plays pay 0 and pull it back down.
- Each success moves $Q(C)$ only 10% of the way to 4: $0.2 \to 0.58 \to 0.92 \to 1.23 \to 1.51 \to 1.76 \to 1.98 \to 2.18$, about seven successes with no failure in between.
- Even then both must switch together: one that switches alone earns 0 against a hare hunter and switches back.

Starting at (D,D) (both pessimistic) with $\varepsilon = 0.1$, my runs saw one transition to (C,C) in 500 runs of 3000 rounds, and it fell back. Structurally these are the Stag Hunt's basins: leaving (C,C) needs the partner to look like a defector more than $1 - p^* = 1/3$ of the time, leaving (D,D) needs it to look like a cooperator more than $p^* = 2/3$ of the time. In Lab 1's terms (D,D) is risk-dominant ($P - S = 2 > R - T = 1$), and noisy learners drift to it and stay.

> [!warning] Common pitfalls in Part I
> - **"C is the best response, so learners stay at (C,C)."** A constant-rate learner tracks recent payoffs and reads a streak of exploratory defections as a change in the partner.
> - **"More exploration helps coordination."** In the Stag Hunt it destroys (C,C) and leaves (D,D) intact.
> - **Calling the Prisoner's Dilemma result a coordination failure.** It has one equilibrium, and the learners found it.

---

## Part II: the Coin Game with tabular Q-learning

### The model

The **Coin Game** (Lerer and Peysakhovich, 2017) turns the repeated Prisoner's Dilemma into a grid world. Two agents, red and blue, move on a $3 \times 3$ grid that always holds exactly one coin, red or blue.

- The grid is a **torus** (walking off one edge enters the opposite one), so every cell is at most two moves away.
- Actions: up, down, left, right. Both agents move at the same time and may share a cell.
- An agent picks up the coin by moving onto its cell; both can take the same coin in the same step. A new coin of random colour then appears on a random cell not occupied by an agent.
- An episode lasts 16 steps; then agents and coin are placed at random again.
- Each agent observes four one-hot $3 \times 3$ grids (36 numbers): its own position, the other's position, the coin if it has its own colour, the coin if it has the other's colour.

A pickup pays the picker +1 whatever the coin's colour, and taking the other player's coin costs that player 2:

| pickup | picker | coin's owner | total |
|---|---|---|---|
| own coin | +1 | (the picker) | +1 |
| other's coin | +1 | −2 | −1 |

Going for your own coins and leaving the other's alone is the cooperative pattern; taking every reachable coin is defection. Both are patterns of movement the agents must discover from positions.

**Three measures**, averaged over the games of an update: **individual performance** (coin pickups per episode, both agents together, any colour; random agents get about 3.7), **restraint** (the share of pickups of the picker's own colour: 0.5 for colour-blind agents, 1 for full restraint) and **welfare** (total reward per episode, red plus blue). From the table,

$$\text{welfare} = \text{pickups} \times (2 \times \text{restraint} - 1),$$

so welfare needs both skill and restraint, and colour-blind experts score about 0.

**The learners.** Each agent runs regular [[Q-Learning]] with a table ([[Tabular RL]]):

$$Q(s, a) \leftarrow Q(s, a) + \alpha\,\big(r + \gamma \max_{a'} Q(s', a') - Q(s, a)\big)$$

where $s$ is the state read from the observation (own cell, other's cell, coin cell, whether the coin is mine: $9 \cdot 9 \cdot 9 \cdot 2 = 1458$ rows), $a$ one of the four moves and $s'$ the next state, with $\alpha = 0.1$, $\gamma = 0.95$, $\varepsilon = 0.1$ and all values starting at 0. The $\gamma \max$ term now matters: it credits a move for bringing the agent closer to a coin. Each agent treats the other as part of the environment. Training runs 32 games in parallel; one update is one 16-step episode in each, so 512 decisions per agent.

The lab's text speaks of 300 updates, but the training function's default, which its cells run, is 1000. I report values at update 300 and at the end of the 1000-update run.

### Q5: predictions before the runs

> [!question] Statement
> (a) Two independent Q-learners are trained with their own rewards. After training, what share of each agent's pickups do you expect to be coins of its *own* colour?
>
> (b) How large will the total reward per episode (red + blue) be, compared with two agents that move at random?

**(a)** About half. Nothing in an agent's own reward pays for restraint (Q6(b)), so a self-interested learner takes whichever coin it can reach, and colours are random.

**(b)** About the same as random agents, close to 0. The learners will pick up many more coins, but colour-blind pickups add nothing to the total on average (welfare formula). Both predictions hold (Q6).

### Q6: what the Q-learners learn

> [!question] Statement
> (a) Describe the three panels (individual performance, restraint, welfare). How did their individual performance evolve? Did they show restraint? Compare their total reward with that of random agents.
>
> (b) Use the payoffs to explain why an agent that maximises its own reward picks up every coin it can reach, whatever its colour. What is the expected total reward per pickup when both agents do this?
>
> (c) Why can independent Q-learning, by itself, not find the outcome in which both agents only pick up their own coins? Relate your answer to the Prisoner's Dilemma and to the phrase "treats the other agent as part of the environment".

| | pickups | own share | welfare |
|---|---|---|---|
| random agents | 3.6 to 3.8 | 0.49 to 0.52 | −0.1 to +0.1 |
| update 100 | 12.1 | 0.60 | +2.4 |
| update 300 | 14.0 | 0.57 | +2.0 |
| update 1000 | 16.9 | 0.52 | +0.6 |

(Per episode, averaged over 20 updates of one run; two more seeds give 16.8 to 17.3 pickups, share 0.52 and welfare +0.7 at update 1000.)

**(a)** Individual performance rises steadily from the random level to about 17 coins per episode for the pair after 1000 updates (4.6 times random), and is still creeping up. Restraint stays near 0.5: a small bump to about 0.6 around updates 100 to 200 fades back to 0.52. Welfare follows the formula, $16.9 \times (2 \cdot 0.52 - 1) \approx 0.7$, so the total ends barely above random agents' (about 0) with more than four times the pickups. The agents got good at the individual task and no better at the social one.

The early bump is not cooperation. A likely reason, read off the payoffs: taking a coin rather than letting the other take it is worth $1 - (-2) = 3$ to me for my own coin and $1 - 0 = 1$ for the other's. So a selfish learner races harder for its own coins, and once both reach every coin fast, that preference matters less.

**(b)** Compare taking a coin with leaving it, from my own reward only. Taking it pays +1 now, whatever its colour. Leaving the other's coin pays me 0, and if the other then takes it I still get 0. The −2 my taking causes lands in the other's reward, which never enters my $Q$-values. So in every state taking is at least as good for me as leaving: it plays the role of D in the Prisoner's Dilemma, a dominant action. When both agents do it, half the pickups are own coins (+1 in total) and half the other's ($+1 - 2 = -1$):

$$\mathbb{E}[\text{total reward per pickup}] = \tfrac12(+1) + \tfrac12(-1) = 0.$$

(A simultaneous pickup of one coin also totals 0: $+1 + 1 - 2$ over two pickups.)

**(c)** Two reasons work together. First, each Q-learner maximises its own return, and for that return restraint is costly: it gives up +1 now and protects only the other's reward. It could pay only if it changed what the other does later (reciprocity, as Tit-for-Tat in Lab 1). Second, independent Q-learning **treats the other agent as part of the environment**: the other's behaviour is just part of the transitions and rewards, and the state holds positions and the coin, with no record of whether the other has been restrained. So neither agent can reward restraint or punish a defector; in Lab 1's terms both are memory-0 with respect to the partner, and against any fixed behaviour of the other, taking every coin is the best response. The result is mutual defection, as with the Part I learners in the Prisoner's Dilemma. Restraint would need rewards that include the other's welfare, memory of the other's behaviour, or learners that account for how the other learns.

### Q7: irrelevant lamps (prediction)

> [!question] Statement
> Each agent's observation gets $k$ **lamps**: signals switched on or off at random at every step, with no effect on the game. For a table, two observations differing in one lamp are different states: $1458 \cdot 2^k$ rows. Before running: how do you expect the individual performance of the Q-learners (coin pickups per episode) to change with 4 and with 8 lamps? Would the lamps bother a human player?

The best policy ignores the lamps, but a table cannot know that: it learns each lamp pattern separately. With a fixed amount of experience, 4 lamps should slow learning noticeably and 8 lamps (256 copies of every state) should leave the agents close to random. A human would not be bothered: after a few steps they would see that the lamps change whatever anyone does and ignore them, because people generalise across situations that differ only in irrelevant details.

### Q8: why the table breaks

> [!question] Statement
> Tabular Q-learners are trained with 0, 4 and 8 lamps and the same training budget. After training, the code reports the fraction of states visited during training and the share of decisions, in new games, taken in never-visited states (where all values are 0, so the agent acts at random).
>
> (a) How does the individual performance change with the number of lamps? Quote the numbers.
>
> (b) **State space.** How many rows does the Q-table have with $k$ lamps? Compute $N$ for $k = 0, 4, 8$.
>
> (c) **Training budget.** The budget is counted in decisions per agent (environment steps): $B = \texttt{n\_updates} \times \texttt{n\_envs} \times \texttt{episode\_length}$. Compute $B$ for the runs above (read the three numbers from the code). How many decisions per state does it give on average, $B / N$, for $k = 0, 4, 8$?
>
> (d) **Expected number of visited states.** Assume each of the $B$ decisions lands in a state drawn uniformly at random from the $N$ states, independently.
> 1. Show that the probability that a given state is never visited is $(1 - 1/N)^B \approx e^{-B/N}$, and hence that the expected number of visited states is $N\big(1 - (1 - 1/N)^B\big) \approx N(1 - e^{-B/N})$.
> 2. Compute it for $k = 0, 4, 8$ and compare the fraction visited$/N$ with the printed fraction. Your prediction should be too high. Not all $9 \cdot 9 \cdot 9 \cdot 2$ position states can occur: where can the coin *not* be when an agent makes a decision? Count the states that can occur and redo the computation with them.
> 3. Under the same assumption, what share of the test decisions should fall in never-visited states? Compare with the printed share.
>
> (e) Use (c) and (d) to explain the result of (a).
>
> (f) The best policy is the same with and without lamps. Why can a table not discover this? What would a learner need in order to be unaffected by the lamps?

**(a)** Pickups per episode (random agents: about 3.7):

| lamps | update 300 | update 1000 |
|---|---|---|
| 0 | 14.0 | 16.9 |
| 4 | 9.9 | 12.8 |
| 8 | 4.7 | 6.6 |

(Three seeds agree within about 1.5 pickups at update 300 and 0.5 at update 1000.) Four lamps cost about a quarter of the performance; eight leave the agents barely above random after 300 updates and far below the lamp-free learners after 1000. With lamps, restraint stays between 0.47 and 0.56; the 4-lamp learners end with share 0.56 and welfare about +1.5, which matches the early phase in which the lamp-free learners also had a share above 0.5 (Q6(a)): they are slower along the same path.

**(b)** $N = 1458 \cdot 2^k$: **1,458** rows for $k = 0$, **23,328** for $k = 4$, **373,248** for $k = 8$.

**(c)** From the code: 1000 updates, 32 games, 16 steps, so

$$B = 1000 \times 32 \times 16 = 512{,}000 \text{ decisions per agent,}$$

and $B / N = 351$ for $k = 0$, $21.9$ for $k = 4$ and $1.37$ for $k = 8$. (With the 300 updates the text mentions, $B = 153{,}600$ and $B/N = 105$, $6.6$ and $0.41$.)

**(d) 1.** One decision misses a given state with probability $1 - 1/N$, so $B$ independent decisions all miss it with probability $(1 - 1/N)^B = e^{B \ln(1 - 1/N)} \approx e^{-B/N}$, using $\ln(1 - 1/N) \approx -1/N$ for large $N$. Each state is visited with probability $1 - (1 - 1/N)^B$, and by linearity of expectation (summing over the $N$ states) the expected number of visited states is $N\big(1 - (1-1/N)^B\big) \approx N(1 - e^{-B/N})$.

**(d) 2.** With all $N$ states the predicted visited fractions are $1 - e^{-351} \approx 1$, $1 - e^{-21.9} \approx 1$ and $1 - e^{-1.372} = 0.746$. My runs print 0.790, 0.790 and 0.648, lower in every case.

When an agent decides, the coin is never on an agent's cell: an agent standing on it would have picked it up, and new coins go on free cells. If the agents are on different cells ($9 \cdot 8 = 72$ ordered pairs) the coin has 7 possible cells; if they share one (9 cases) it has 8:

$$N'_0 = (9 \cdot 8 \cdot 7 + 9 \cdot 8) \times 2 = (504 + 72) \times 2 = 1152 \text{ reachable position states},$$

and $N'_k = 1152 \cdot 2^k$: 1,152, 18,432 and 294,912, so only $1152/1458 = 79.0\%$ of the rows can ever be visited. Then $B/N' = 444$, $27.8$ and $1.736$; the expected visited share of reachable states is 1, 1 and $1 - e^{-1.736} = 0.824$, and of all rows $0.790$, $0.790$ and $0.790 \cdot 0.824 = 0.651$. Printed: 0.790, 0.790, 0.648. The corrected model fits almost exactly (my runs visited every reachable state for 0 and 4 lamps, and 82.0% for 8).

**(d) 3.** A test decision lands in a uniform reachable state, never visited with probability $e^{-B/N'}$: $e^{-444} \approx 0$, $e^{-27.8} \approx 10^{-12}$, and $e^{-1.736} = 0.176$ for 8 lamps. Printed: 0.000, 0.000 and 0.174. (With the 300-update budget the 8-lamp prediction would be $e^{-0.521} = 59\%$.)

**(e)** The budget is fixed while the table grows by $2^k$, so experience per row shrinks by the same factor. With 0 lamps each reachable state gets about 444 decisions, enough to learn its values. With 4 lamps every state is visited, but only about 28 times across four actions, and each update corrects 10% of the error while bootstrapping from under-trained next-state values, so values are only partly learned. With 8 lamps the average state is seen 1.7 times: 17% of test decisions land in a row never seen (random action), and most of the rest in rows seen once or twice, with values near 0. That is close to random play, hence 6.6 pickups.

**(f)** A table has one independent row per observation: nothing learned in one row is used in any other, so it cannot notice that 256 rows differing only in the lamps call for the same action. It must learn the same policy 256 times over, from 1/256 of the data each time. To be unaffected, a learner needs **generalisation** across states: a representation that shares parameters between similar observations and can learn to give the lamps no weight, such as [[Function Approximation]] with a neural network ([[Neural Network Function Approximation]]) or with features that leave the lamps out.

---

## Part III: deep RL with PPO

### The model

Deep reinforcement learning replaces the table with a neural network that takes the observation (36 numbers plus lamps) as input; all states share the weights, so learning in one state carries over to similar ones. The lab uses [[PPO]] (proximal policy optimisation), a [[Policy Gradient Methods|policy gradient]] method: it learns the policy itself and changes its parameters to make actions with higher long-term reward more likely.

**Actor and critic.** PPO is an [[Actor-Critic]] method with two separate networks, each with two hidden layers of 64 tanh units:

- the **actor** $\pi(a \mid s; \phi)$ outputs one preference per action; a softmax turns them into probabilities ([[Softmax Policy]]) and the agent **samples** its action;
- the **critic** $V(s; \theta)$ outputs one number, an estimate of the return from state $s$.

**Advantage.** The actor makes action $a_t$ more likely in proportion to its [[Advantage Function|advantage]], how much better it did than the critic expected:

$$\text{Adv}(s_t, a_t) = r_t + \gamma V(s_{t+1}; \theta) - V(s_t; \theta)$$

This is the one-step estimate. The code mixes horizons with [[Generalized Advantage Estimation]]: weight $\lambda = 0$ gives the one-step estimate, $\lambda = 1$ the whole rest of the episode (Monte Carlo); here $\lambda = 0.95$. The critic is trained towards the target $y_t = \text{Adv}(s_t, a_t) + V(s_t; \theta)$.

**Clipped actor loss.** Each batch, collected by the **behaviour policy** $\pi_\beta$ (the policy before the update), is used for several gradient steps. The [[Importance Sampling|importance sampling weight]] $\rho = \pi(a \mid s; \phi) / \pi_\beta(a \mid s)$ measures how far the policy has moved since, and the loss clips it:

$$L(\phi) = -\min\big(\rho\,\text{Adv},\ \text{clip}(\rho, 1-\epsilon, 1+\epsilon)\,\text{Adv}\big), \qquad \epsilon = 0.2.$$

Beyond $1 \pm \epsilon$ the clipped term stops rewarding further change, so each update stays in a **trust region** around $\pi_\beta$.

**Full update.** Each minibatch minimises the actor loss, minus $c_e = 0.01$ times the policy's [[Entropy]] (entropy regularisation), plus $c_v = 0.5$ times the critic loss $(y_t - V(s_t; \theta))^2$. Adam with learning rate 0.003 runs 4 epochs of 4 minibatches over each batch of 512 decisions (16 gradient steps), with advantages normalised to mean 0 and standard deviation 1 within the batch and the gradient norm clipped at 0.5. Then the batch is thrown away (PPO is on-policy). Without a critic the advantage becomes the Monte Carlo return $u_t$ (the discounted sum of rewards from $t$ to the end of the episode), which is [[REINFORCE]].

In the Coin Game each agent runs its own PPO (independent PPO), with the other agent treated as part of the environment, exactly as the Q-learners did.

### Q9: the pieces of PPO

> [!question] Statement
> (a) What does the critic add, compared with an algorithm that uses the Monte Carlo return of the whole episode (REINFORCE)?
>
> (b) The Q-learners needed $\varepsilon$-greedy exploration. Where does a PPO agent's exploration come from, and which term of the loss keeps it from disappearing too early?

**(a)** REINFORCE pushes every action up in proportion to the whole return that followed it. In a lucky episode even bad moves are reinforced, and the return mixes the effects of many later actions, coin placements and the other agent's moves, so the gradient estimate has high variance. The critic adds:

- a **[[Baseline]]**: subtracting $V(s_t)$ asks whether the action did better than usual from this state, which removes the part of the outcome that does not depend on the action and lowers the variance;
- **bootstrapping**: with $\lambda < 1$ the advantage uses $V(s_{t+1})$ in place of the rest of the episode, so an action is judged mainly by its immediate consequences. This sharpens credit assignment and lowers variance further, at the price of some bias when the critic is wrong.

(The code's batch normalisation of advantages subtracts the batch mean, a crude constant baseline, so its REINFORCE variant is not entirely without one.)

**(b)** From the policy itself: the agent samples from the actor's probabilities, so every action with non-zero probability keeps being tried. Training pushes good actions towards probability 1, which would end exploration; the **entropy regularisation term** ($-c_e$ times the policy's entropy) rewards spread-out probabilities, so the policy does not become deterministic too early.

### Q10: PPO against the table

> [!question] Statement
> Each agent runs its own PPO learner with the same budget as the tabular learners, without lamps and with 8 lamps.
>
> (a) Without lamps, which learner reaches the higher individual performance after 300 updates, and which one learns faster? Which one needs more computation?
>
> (b) With 8 lamps, compare the two learners. Why is PPO hardly affected? Relate your answer to Q8(f).
>
> (c) Look at the restraint and outcome panels. Does PPO solve the social dilemma? What is the problem that a better function approximator does *not* solve?
>
> (d) Give one situation in which you would still choose a table rather than a neural network.

Pickups per episode at updates 300 and 1000, own share and welfare at update 1000 (PPO: two seeds of my own implementation; its level depends on implementation details more than the table's, so the comparisons matter more than its exact numbers):

| learner, lamps | @300 | @1000 | share | welfare |
|---|---|---|---|---|
| table, 0 | 14.0 | 16.9 | 0.52 | +0.6 |
| table, 8 | 4.7 | 6.6 | 0.48 | −0.3 |
| PPO, 0 | 11.5 to 12.6 | 12.2 to 13.2 | 0.52 to 0.53 | +0.5 to +0.7 |
| PPO, 8 | 11.6 to 12.4 | 12.8 to 13.3 | 0.51 | +0.2 to +0.3 |

**(a)** Without lamps the table wins in my runs on both counts: it is ahead after 50 updates (10.7 pickups against 6.8 to 8.1), after 300 and after 1000. PPO reaches 11 to 12 pickups by update 200 and then plateaus with an almost deterministic policy (average entropy falls from 1.37 nats, near the maximum $\ln 4 = 1.39$, to 0.05 to 0.11 by update 300). The lamp-free game is the best case for a table: 1,152 reachable states, each visited about 444 times. PPO needs much more computation: every decision passes through two networks and each batch is used for 16 gradient steps through both, while the table changes one entry per decision. The notebook's timing comments put a PPO run at several times a tabular one; in my code it was 125 s against 8 s.

**(b)** With 8 lamps PPO does as well as without them, while the table drops to 6.6, so PPO now leads by about a factor two. This is the generalisation Q8(f) asked for. The network has one set of weights for all observations, so experience under one lamp pattern is used under every other. The lamps carry no information about the return, so the gradient pushes their weights without a consistent direction, they get no lasting influence, and behaviour depends on the positions alone. The network hardly grows: 8 more inputs add 512 weights to each first layer (the actor goes from 6,788 to 7,300 parameters), while the table goes from 5,832 entries to 1,492,992.

**(c)** No. At every stage of training restraint stays between 0.48 and 0.56 and welfare between about −0.1 and +1.2 per episode, close to the Q-learners. The rewards can also be lopsided: in one run red averaged −1.0 per episode and blue +1.7. A better function approximator solves a problem of representation, learning a good policy from fewer, more varied data. The dilemma is a problem of incentives: under each agent's own reward, taking every reachable coin is the best response to anything the other does (Q6(b)), and independent PPO optimises exactly that reward with the other treated as part of the environment. A more capable optimiser of the same selfish objective finds the same defection.

**(d)** When the state space is small and discrete and every state is visited often, as in the Part I stage game or the lamp-free Coin Game, where the table beat PPO with a fraction of the computation. Or when the values must be exact and inspectable, or the setting needs the convergence guarantee tabular Q-learning has in a stationary single-agent problem.

### Q11: wrap-up

> [!question] Statement
> (a) Give two experiments in this lab in which a *faster or more capable* learner led to a *worse*, or no better, collective outcome. Quote the numbers.
>
> (b) Complete the sentence, with evidence from Parts II and III: "Deep RL solved ..., but not ...".
>
> (c) Suppose the two agents of the Stag Hunt in Part I were PPO learners instead of Q-learners. Would the problem of equilibrium selection disappear? Why (not)?

**(a)**

1. **More training in the Coin Game.** The tabular Q-learners' welfare peaked at about +2.4 to +2.7 per episode around updates 100 to 200 (12 to 14 pickups, own share 0.60) and fell to +0.6 by update 1000 (16.9 pickups, share 0.52): better at the individual task, worse collectively, ending barely above random agents (about 0) with 4.6 times their pickups.
2. **PPO against the table with 8 lamps.** PPO is the more capable learner, 12.8 to 13.3 pickups against 6.6, and its collective outcome is no better: own share 0.51 against 0.48, welfare +0.2 to +0.3 against −0.3, all close to 0.

**(b)** "Deep RL solved **generalisation**: 8 irrelevant lamps left PPO at its lamp-free level (12.8 to 13.3 pickups) while they cut the table from 16.9 to 6.6, **but not the social dilemma**: with either learner the own share stays near 0.5 and welfare near 0, because each agent's own reward makes taking every coin the best response."

**(c)** No. Equilibrium selection comes from the game together with learners that each follow their own payoff, and a network changes neither; with a single state there is nothing to generalise over. For a two-action softmax policy with logit difference $z$, $\pi_C = \sigma(z)$ and the expected payoff is $J = \pi_C f_C + (1-\pi_C) f_D$, where $f_C, f_D$ are the expected payoffs of C and D against a partner who plays C with probability $q$. Since $d\sigma/dz = \sigma(1-\sigma)$,

$$\frac{\partial J}{\partial z} = \pi_C(1 - \pi_C)\,(f_C - f_D), \qquad f_C - f_D = q(R-T) + (1-q)(S-P) = 3q - 2.$$

Like Lab 1's replicator equation $\dot x = x(1-x)(f_C - f_D)$, this pushes towards C exactly when $q > 2/3$. A fresh network gives about $\pi_C = 1/2$, so each agent sees $q \approx 1/2$, its gradient points towards D, and the pair drifts to (D,D), as the replicator dynamics do from $x = 1/2$; both starting above $2/3$ would lead to (C,C). Both pure equilibria stay stable ($f_C - f_D = 1$ near $q = 1$, $-2$ near $q = 0$), so the outcome depends on initialisation and noise. Sampled actions and the entropy bonus keep the policies stochastic, which acts like exploration and, as in Q4, threatens (C,C) more than (D,D). Clipping limits the size of each step and leaves its direction unchanged.

> [!warning] Common pitfalls in Parts II and III
> - **Reading an own share above 0.5 as cooperation.** Own coins are worth more to a selfish agent (taking one also prevents a −2).
> - **Counting all $9 \cdot 9 \cdot 9 \cdot 2$ rows as visitable.** Only 1,152 of the 1,458 position states can occur.
> - **"A better learner fixes the dilemma."** Function approximation fixes representation; the dilemma comes from the rewards.

---

## What to be able to write

- The stateless Q-learning update, why the $\gamma \max$ term can be dropped, and why a constant learning rate makes $Q$ an exponentially weighted average of recent payoffs.
- Why learners from $Q = (0,0)$ reach (C,C) essentially only if both play C in round 1, and then lock into (D,D).
- Why the Prisoner's Dilemma result is a social dilemma and the Stag Hunt result a coordination failure.
- $Q(C) - Q(D) = 3b - 2$, and the optimist-sceptic trace.
- Why exploration knocks learners out of (C,C) and almost never out of (D,D).
- The Coin Game's rewards, the welfare formula, and why taking every coin is dominant with expected total 0 per pickup.
- State counts, reachable states, the budget $B$, and the uniform-visit model with its fit.
- The pieces of PPO and what the critic adds over REINFORCE.
- "Deep RL solved generalisation, but not the social dilemma", and why PPO would not remove equilibrium selection in the Stag Hunt.

## Exam questions

> [!exam]- Two independent stateless Q-learners ($\alpha = 0.1$, $\varepsilon$-greedy) play the Stag Hunt with $R=4$, $S=0$, $T=3$, $P=2$, starting from $Q = (0,0)$. Explain which equilibrium they usually reach and why, and how this compares with rational agents and with replicator dynamics.
> **Key points:** random play at the start acts like belief 1/2 < $p^* = 2/3$; D earns 2.5 against 2 and never below 2; (C,C) essentially only if both play C in round 1 (probability 1/4); feedback lock-in at (D,D), then exploration erodes the remaining (C,C); final (C,C) under 10%, below rational agents' 11% and the replicator's one third.
>
> **Start.** All values are 0, so both agents break ties at random and each faces a coin-flipping partner. Then C earns $\frac12 \cdot 4 = 2$ and D earns $\frac12 \cdot 3 + \frac12 \cdot 2 = 2.5$ on average, and D never earns less than 2 while C earns 0 half the time. The start is like holding belief $1/2$, below $p^* = (P-S)/((R-T)+(P-S)) = 2/3$.
>
> **Round 1 decides.** If both play C, both get 4 and $Q(C) = 0.4 > 0$: both prefer C. If either plays D, the D player's $Q(D)$ becomes positive, and a C player facing D earns 0, stays tied and drifts to D as soon as its tie-break picks D (2 against 0). So (C,C) is reached with probability about $1/4$; my runs give 28% after the first rounds.
>
> **Lock-in.** Once the partner plays D, C pays 0, so $Q(C)$ never gets evidence of the 4 it could earn; $Q(C)$ rises only when both play C together, $(\varepsilon/2)^2 = 0.25\%$ of rounds. Meanwhile the partner's exploratory defections push (C,C) runs over to (D,D) (each zero cuts $Q(C)$ by 10%), and nothing comes back. After 3000 rounds (C,C) holds in roughly 5% to 10% of runs.
>
> **Comparison.** Rational agents with uniform beliefs reached (C,C) in 11% of encounters, and replicator dynamics reached all-C from one third of starting states. Learning did not help: the learners start below $p^*$ and noise drives them towards the risk-dominant (D,D) ($P - S = 2 > R - T = 1$).

> [!exam]- Two stateless Q-learners in the Stag Hunt ($R=4, S=0, T=3, P=2$, $\alpha = 0.1$) start at (C,C) with $Q = (3.6, 2.9)$. Explain why they leave (C,C) at $\varepsilon = 0.1$ although C is the best response, and why the escape from (D,D) to (C,C) almost never happens.
> **Key points:** C pays 3.8 and D 2.95 against an exploring partner; $Q$ is an exponentially weighted average of recent payoffs (memory about 10 plays); three zeros take $Q(C)$ from 3.8 to 2.77 < 2.95; the partner then earns 0 and follows; from (D,D) C pays 4 only on joint exploration (0.25% of rounds), about seven successes in a row needed, both must switch at once; basins of $1/3$ against $2/3$.
>
> **Best response.** The partner plays D in $\varepsilon/2 = 5\%$ of rounds. C earns $0.95 \cdot 4 = 3.8$, D earns $0.95 \cdot 3 + 0.05 \cdot 2 = 2.95$, so a rational agent stays.
>
> **What $Q$ estimates.** Unrolling the update gives $Q_t(C) = \alpha \sum_k (1-\alpha)^k r_{t-k}$: an exponentially weighted average of recent payoffs, with weights halving about every 6.6 plays. With a constant $\alpha$ it tracks what C paid lately, with standard deviation $\sigma\sqrt{\alpha/(2-\alpha)} = 0.20$ around 3.8.
>
> **Escape.** Each exploratory D of the partner pays me 0 and cuts $Q(C)$ by 10%: $3.8 \to 3.42 \to 3.08 \to 2.77 < 2.95$. A short run of three such zeros, likely somewhere in 3000 rounds, flips my greedy action to D; then the partner earns 0 every round and flips within about three rounds. In my runs only 17% to 20% of pairs are at (C,C) after 3000 rounds (100% at $\varepsilon = 0.01$, 0.4% or less at $\varepsilon = 0.5$). One agent's exploration is noise in the other's rewards, and the learner reads it as a change in the partner.
>
> **No way back.** At (D,D), C pays $0.05 \cdot 4 = 0.2$ on average and D 2.05. $Q(C)$ changes only when I explore C, and pays 4 only if the partner explores C in the same round, 0.25% of rounds; failures are 19 times as common. Climbing from 0.2 to above 2 takes about seven successes in a row, and both agents must switch together or the switcher earns 0. Structurally, leaving (C,C) needs the partner to look like a defector more than $1 - p^* = 1/3$ of the time, leaving (D,D) needs it to look like a cooperator more than $p^* = 2/3$ of the time.

> [!exam]- In the Coin Game a pickup pays the picker +1 whatever the coin's colour, and taking the other player's coin costs that player 2. Explain why two independent Q-learners collect many coins but do not show restraint, derive the expected total reward per pickup, and relate this to the Prisoner's Dilemma.
> **Key points:** taking any reachable coin is at least as good for my own reward in every state; the −2 lands in the other's reward, which my values never see; welfare = pickups × (2 × share − 1); expected total per pickup 0 for colour-blind agents; independent learners treat the other as environment, no memory of its behaviour, no reciprocity; same structure as mutual defection.
>
> **Dominance.** Taking a coin pays me +1. Leaving the other's coin pays me 0 now, and if the other then takes it I still get 0; the cost of my taking it, −2, goes to the other. So taking every reachable coin is a dominant behaviour for my own reward, the analogue of D.
>
> **Total per pickup.** An own-colour pickup adds +1 to the total; a pickup of the other's coin adds $+1 - 2 = -1$. With colour-blind agents half of the pickups are of each kind, so the expected total is $\frac12 - \frac12 = 0$, and welfare $= \text{pickups} \times (2 \times \text{restraint} - 1) \approx 0$ however skilled they are.
>
> **Results.** My Q-learners went from the random level (3.7 pickups) to about 17 pickups per episode, with own-colour share 0.52 and welfare about +0.6, against about 0 for random agents.
>
> **Why learning cannot fix it.** Independent Q-learning treats the other agent as part of the environment and maximises its own return. Restraint costs +1 and benefits only the other, unless it changes the other's future behaviour. But the state holds only positions and the coin, with no record of the other's past restraint, so neither agent can reciprocate or punish; both are memory-0 with respect to the partner, and against any fixed behaviour of the other, taking every coin is the best response. As in the one-shot Prisoner's Dilemma, individually rational learning ends in mutual defection.

> [!exam]- A tabular Q-learner in the $3 \times 3$ Coin Game gets $k$ irrelevant random lamps in its observation and a budget of $B = 512{,}000$ decisions. Compute the number of rows, the reachable states and the expected share of visited states and of test decisions in unvisited states for $k = 0, 4, 8$, and explain what this means for learning.
> **Key points:** $N = 1458 \cdot 2^k$ (1,458; 23,328; 373,248); coin never on an agent's cell, so $N' = 1152 \cdot 2^k$; never-visited probability $(1-1/N')^B \approx e^{-B/N'}$; $B/N' = 444$, 27.8, 1.74; for $k=8$ about 82% of reachable states visited and 17.6% of test decisions unseen; a table does not generalise, a function approximator can ignore the lamps.
>
> **Rows.** Own cell, other's cell, coin cell (9 each), colour (2), lamp pattern ($2^k$): $N = 1458 \cdot 2^k$.
>
> **Reachable.** At decision time the coin is never on an agent's cell. Agents on different cells: $9 \cdot 8$ ordered pairs, 7 coin cells; on the same cell: 9 pairs, 8 coin cells. $(504 + 72) \cdot 2 = 1152$, so $N' = 1152 \cdot 2^k$ (79% of the rows).
>
> **Uniform-visit model.** A given state is missed by one decision with probability $1 - 1/N'$, by all $B$ independent decisions with probability $(1 - 1/N')^B \approx e^{-B/N'}$; expected visited states $N'(1 - e^{-B/N'})$. With $B/N' = 444$, 27.8 and 1.736: all reachable states are visited for $k = 0$ and 4, and $1 - e^{-1.736} = 82.4\%$ for $k = 8$. A uniform test decision falls in an unvisited state with probability $e^{-B/N'}$: about 0, about $10^{-12}$, and 17.6%. My runs: 82.0% visited and 17.4% unseen for $k = 8$.
>
> **Meaning.** With a fixed budget, experience per row falls by $2^k$. At $k = 4$ every state is seen, but only about 28 times across four actions, so values are partly learned (pickups 12.8 against 16.9 at $k = 0$). At $k = 8$ the agent acts at random in 17% of states and on barely trained values in most others (6.6 pickups, random agents 3.7). The table cannot notice that rows differing in the lamps need the same action; a learner that shares parameters across observations (a neural network, as in PPO) can give the lamps no weight.

> [!exam]- Describe the components of PPO (actor, critic, advantage, clipped objective, entropy regularisation), say what the critic adds over REINFORCE, and explain why independent PPO in the Coin Game handles irrelevant inputs but still fails the social dilemma.
> **Key points:** actor softmax policy sampled for exploration; critic $V(s)$ as baseline and bootstrap for $\text{Adv} = r + \gamma V(s') - V(s)$ (GAE with $\lambda$); clipped ratio $\rho$ keeps updates in a trust region; entropy bonus prevents premature determinism; shared weights generalise across lamp patterns; own-colour share about 0.5 and welfare about 0 because each agent's reward makes taking every coin the best response.
>
> **Actor.** A network $\pi(a \mid s; \phi)$ outputs action preferences turned into probabilities by a softmax; the agent samples, which is where exploration comes from.
>
> **Critic and advantage.** A second network $V(s; \theta)$ estimates the return. The advantage $\text{Adv}(s_t,a_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$ says how much better the action did than expected; generalized advantage estimation mixes horizons with $\lambda$ (0 one-step, 1 Monte Carlo). Compared with REINFORCE, which pushes each action by the whole return that followed, the critic provides a baseline and bootstrapping, so the gradient has much lower variance and credit goes to the action's own consequences, at the cost of some bias.
>
> **Clipped objective.** Each batch is reused for several epochs. The ratio $\rho = \pi(a\mid s;\phi)/\pi_\beta(a\mid s)$ measures how far the policy moved; the loss $-\min(\rho\,\text{Adv}, \text{clip}(\rho, 1-\epsilon, 1+\epsilon)\,\text{Adv})$ stops rewarding change beyond $1 \pm \epsilon$, a trust region around the behaviour policy.
>
> **Entropy.** The loss subtracts $c_e$ times the policy entropy, keeping probabilities spread so exploration does not stop too early.
>
> **Coin Game.** The network shares weights across all observations, and random lamps carry no information about returns, so PPO learns to ignore them: in my runs 12.8 to 13.3 pickups with 8 lamps, as without, against 6.6 for the table. But each PPO agent still optimises its own reward with the other treated as environment, so taking every coin stays the best response: own-colour share about 0.51 and welfare near 0. Deep RL fixes representation and generalisation; the dilemma lives in the incentives.

> [!exam]- Would replacing the Stag Hunt's Q-learners with PPO learners remove the problem of equilibrium selection? Use the expected policy gradient of a two-action softmax policy.
> **Key points:** one state, nothing to generalise; $\partial J/\partial z = \pi_C(1-\pi_C)(f_C - f_D)$ with $f_C - f_D = 3q - 2$; like the replicator equation, it favours C only when $q > 2/3$; from $\pi = 1/2$ both drift to D; both pure equilibria stable, the outcome depends on initialisation and noise; sampling and entropy act like exploration and threaten (C,C) more.
>
> **Gradient.** With logit difference $z$, $\pi_C = \sigma(z)$ and $J = \pi_C f_C + (1-\pi_C) f_D$, where $f_C = qR + (1-q)S$ and $f_D = qT + (1-q)P$ for a partner who plays C with probability $q$. Since $d\sigma/dz = \sigma(1-\sigma)$, $\partial J/\partial z = \pi_C(1-\pi_C)(f_C - f_D)$, and for $R=4, S=0, T=3, P=2$, $f_C - f_D = q - 2(1-q) = 3q - 2$. Like the replicator equation $\dot x = x(1-x)(f_C - f_D)$, it pushes towards C exactly when $q > 2/3$.
>
> **Outcome.** A fresh network gives $\pi_C \approx 1/2$ to both agents, so each sees $q \approx 1/2 < 2/3$ and moves towards D; the pair goes to (D,D). Initial policies above $2/3$ would go to (C,C). Both pure equilibria are stable ($f_C - f_D = +1$ near $q = 1$, $-2$ near $q = 0$), so which one is reached depends on where the policies start and on noise, exactly the problem Q-learners had.
>
> **Why the network does not help.** The Stag Hunt has a single state, so generalisation across states, PPO's advantage in the Coin Game, has nothing to work on; clipping limits the size of steps and leaves their direction unchanged. Sampled actions and the entropy bonus keep the policy stochastic, which, as with $\varepsilon$-greedy, makes (C,C) more fragile than (D,D). Equilibrium selection belongs to the game and to independent learners each following their own payoff.

## Flashcards

> [!card]- What is the stateless Q-learning update?
> $Q(a) \leftarrow Q(a) + \alpha(r - Q(a))$, only for the action just played.

> [!card]- Why can the $\gamma \max_{a'} Q(s',a')$ term be dropped in a repeated stage game?
> With one state it adds the same amount to both actions' targets, so it never changes which is greedy.

> [!card]- With a constant learning rate $\alpha$, what does $Q(a)$ estimate?
> An exponentially weighted average of recent payoffs of $a$, memory about $1/\alpha$ plays.

> [!card]- What does "independent learning" mean for two agents?
> Each learns from its own actions and rewards only, treating the other as part of the environment.

> [!card]- Why is the environment of an independent learner non-stationary?
> The other agent is learning too, so the payoff each action earns keeps changing.

> [!card]- Against a coin-flipping partner in the Stag Hunt ($R=4, S=0, T=3, P=2$), what do C and D earn on average?
> C earns 2, D earns 2.5.

> [!card]- Independent Q-learners start at $Q=(0,0)$ in the Stag Hunt. When do they reach (C,C)?
> Essentially only if both play C in round 1, probability about 1/4.

> [!card]- About how often do independent Q-learners from $Q=(0,0)$, $\varepsilon=0.1$, end at (C,C) in the Stag Hunt after 3000 rounds?
> Under 10% of runs; the rest end at (D,D).

> [!card]- Why do Stag Hunt Q-learners lock into (D,D) so fast?
> Once the partner plays D, C pays 0, so Q(C) never sees evidence that it could pay 4.

> [!card]- Why is (D,D) reached by Prisoner's Dilemma learners not a coordination failure?
> D is dominant and (D,D) the unique Nash equilibrium; the problem is incentives, a social dilemma.

> [!card]- With belief-based initial values in the Stag Hunt, when does an agent start with C?
> When $Q(C) - Q(D) = 3b - 2 > 0$, i.e. belief $b > p^* = 2/3$.

> [!card]- An optimist (belief 0.9) meets a sceptic (belief 0.5) as Q-learners in the Stag Hunt. Who adapts?
> The optimist: its C earns 0 and Q(C) drops 10% per round; it switches in round 4.

> [!card]- Why does the sceptic never adapt to the optimist in the Stag Hunt?
> It never plays C, so Q(C) is never updated, and its D is rewarded with 3 or 2.

> [!card]- Why does the optimist relapse to C a few times after switching to D?
> At (D,D) its Q(D) falls towards 2 while its stale Q(C) is still above 2.

> [!card]- Against a partner exploring 10% of the time at (C,C), what do C and D pay in the Stag Hunt?
> C pays 3.8, D pays 2.95.

> [!card]- How many exploratory zeros in a row push $Q(C)$ from 3.8 below $Q(D) = 2.95$ at $\alpha = 0.1$?
> Three: $3.8 \to 3.42 \to 3.08 \to 2.77$.

> [!card]- Why do Q-learners leave (C,C) in the Stag Hunt even though C is the best response?
> The partner's exploration gives occasional zeros, and recency-weighted Q reads a short bad streak as a change.

> [!card]- How does (C,C) survival in the Stag Hunt change as $\varepsilon$ rises from 0.01 to 0.5?
> It falls: survives at 0.01, mostly at 0.05, mostly lost at 0.1, gone within rounds at 0.5.

> [!card]- Why is escaping from (D,D) to (C,C) far rarer than the reverse for Q-learners?
> C pays 4 only on joint exploration, Q(C) must climb from about 0.2 to 2, and both must switch together.

> [!card]- In basin terms, why is (C,C) more fragile than (D,D) in the Stag Hunt with $p^* = 2/3$?
> Leaving (C,C) needs the partner to defect over 1/3 of the time; leaving (D,D) needs cooperation over 2/3.

> [!card]- In the Coin Game, what does a pickup of the other player's coin do to the rewards?
> The picker gets +1 and the coin's owner gets −2.

> [!card]- What is the welfare formula of the Coin Game?
> Welfare = pickups × (2 × own-colour share − 1).

> [!card]- What is the expected total reward per pickup when both Coin Game agents are colour-blind?
> 0: half the pickups add +1, half add −1.

> [!card]- Why does a self-interested Coin Game agent take every reachable coin?
> Taking pays +1 whatever the colour; the −2 lands in the other's reward, never in its own.

> [!card]- Why can independent Q-learners not learn restraint in the Coin Game?
> Restraint only helps the other, and their state holds no record of the other's behaviour, so no reciprocity.

> [!card]- Why is an own-colour share of about 0.6 in the Coin Game not evidence of cooperation?
> A selfish agent gains 3 by taking its own coin before the other does, only 1 for the other's coin.

> [!card]- How did tabular Q-learners' Coin Game welfare change with more training?
> It peaked early (about +2.5) and fell to about +0.6 as pickups kept rising.

> [!card]- How many rows does the Coin Game Q-table have with $k$ irrelevant lamps?
> $1458 \cdot 2^k$.

> [!card]- Why can only 1,152 of the 1,458 Coin Game position states occur at decision time?
> The coin is never on a cell occupied by an agent.

> [!card]- Under the uniform-visit assumption, what is the probability a state is never visited in $B$ decisions?
> $(1 - 1/N)^B \approx e^{-B/N}$.

> [!card]- What is the expected number of visited states among $N$ after $B$ uniform decisions?
> $N(1 - e^{-B/N})$.

> [!card]- What training budget do the Coin Game runs use with the code's default of 1000 updates?
> $B = 1000 \times 32 \times 16 = 512{,}000$ decisions per agent.

> [!card]- Why do irrelevant lamps ruin a tabular Q-learner with a fixed budget?
> Each lamp pattern is a separate row, so experience per row falls by $2^k$ and nothing transfers.

> [!card]- What does a learner need to be unaffected by irrelevant inputs?
> Generalisation: parameters shared across observations, able to give irrelevant inputs no weight.

> [!card]- What are the actor and the critic in PPO?
> Actor: softmax policy network. Critic: network estimating the state value $V(s)$.

> [!card]- What is the one-step advantage estimate in actor-critic methods?
> $\text{Adv} = r_t + \gamma V(s_{t+1}) - V(s_t)$.

> [!card]- What does the GAE parameter $\lambda$ control in PPO?
> The horizon of the advantage estimate: 0 one-step, 1 Monte Carlo.

> [!card]- What does PPO's clipping of the importance ratio $\rho$ do?
> Stops rewarding policy change beyond $1 \pm \epsilon$, keeping each update in a trust region.

> [!card]- What does the critic add compared with REINFORCE?
> A baseline and bootstrapping, which lower the variance of the gradient and sharpen credit assignment.

> [!card]- Where does a PPO agent's exploration come from?
> Sampling actions from its stochastic softmax policy.

> [!card]- Which term of PPO's loss keeps exploration from disappearing too early?
> The entropy regularisation term.

> [!card]- Why is PPO hardly affected by 8 irrelevant lamps in the Coin Game?
> Its network shares weights across observations and learns to give the random lamps no influence.

> [!card]- Does independent PPO solve the Coin Game's social dilemma?
> No: own-colour share stays about 0.5 and welfare about 0, as with Q-learners.

> [!card]- Complete: "Deep RL solved ..., but not ..." for the Coin Game.
> Generalisation over irrelevant inputs, but not the social dilemma.

> [!card]- When is a Q-table still a better choice than a neural network?
> Small discrete state spaces visited often, or when exact, inspectable values are needed.

> [!card]- Why would PPO learners in the Stag Hunt still face equilibrium selection?
> Their payoff gradient $\propto 3q - 2$ points to D from a uniform start; both equilibria stay stable.

> [!card]- True or false: in the Stag Hunt, more exploration makes independent Q-learners coordinate on (C,C) more often.
> **False.** Exploration destroys (C,C) and leaves (D,D) intact.

## Links

- **Course:** [[AGAI - Overview|Agentic AI overview]]
- **Lectures:** [[AGAI-L02 - Game Theory and Population Dynamics]] · [[AGAI-L03 - Intro to Multi-Agent Reinforcement Learning]]
- **Previous lab:** [[AGAI-Lab01 - Simple Agents, Complex Dynamics]] (threshold $p^*$, uniform beliefs, replicator dynamics, the Prisoner's Dilemma tournament)
- **Concepts:** [[Q-Learning]] · [[Independent Learning]] · [[Stochastic Game]] · [[Normal-Form Game]] · [[Best Response]] · [[Nash Equilibrium]] · [[Stag Hunt]] · [[Prisoner's Dilemma]] · [[Social Dilemma]] · [[Multi-Agent Credit Assignment]] · [[Replicator Dynamics]] · [[Epsilon-Greedy Policy]] · [[Exploration vs Exploitation]] · [[Discount Factor]] · [[Tabular RL]] · [[Function Approximation]] · [[Neural Network Function Approximation]] · [[Deep Reinforcement Learning]] · [[Policy Gradient Methods]] · [[PPO]] · [[Actor-Critic]] · [[Advantage Function]] · [[Generalized Advantage Estimation]] · [[Baseline]] · [[REINFORCE]] · [[Importance Sampling]] · [[Softmax Policy]] · [[Entropy]]
- **Source:** `AGAI-Lab02.ipynb`, Agentic AI 2026/2027 (UvA MSc AI), Lab 2, "Learning Agents", werkcollege 4 Nov 2026. The notebook ships without outputs; all simulation numbers above come from my own re-implementation with the lab's parameters (own random numbers, own PPO code).
- **Reference named in the lab:** Lerer and Peysakhovich (2017), the Coin Game.
