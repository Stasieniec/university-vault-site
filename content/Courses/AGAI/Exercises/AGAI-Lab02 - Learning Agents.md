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
> Do independent learners reach a good equilibrium? In the [[Stag Hunt]], two independent [[Q-Learning|Q-learners]] mostly lock into the safe (D,D): initial values decide the equilibrium, and one agent's exploration knocks the other out of (C,C) but almost never out of (D,D). In the Coin Game, tabular Q-learners collect coins but take the other's too, because that cost falls on the other agent. Irrelevant random lamps blow up the Q-table; a neural network ([[PPO]]) ignores them but still takes the other's coins.
>
> Part I is mock question 12 (open) exactly; Part II feeds mock question 10 (why only about 56% of coins are the agent's own) and the Deep RL open question 21 (tables against networks). Part III belongs to [[AGAI-L04 - Deep RL and Intro to Deep MARL|Lecture 4]] and is kept short here.

## Part I: independent Q-learners in the Stag Hunt

### The game and the learners

The Stag Hunt with $R = 4$ (both hunt stag), $S = 0$ (I hunt stag alone), $T = 3$ (I hunt hare, the other stag), $P = 2$ (both hunt hare). C is stag, D is hare. It has two pure [[Nash Equilibrium|Nash equilibria]], (C,C) and (D,D). Stag is the best response only if the partner hunts stag with probability above $p^* = 2/3$ (from [[AGAI-Lab01 - Simple Agents, Complex Dynamics|Lab 1]]). (C,C) pays more, but (D,D) is **risk-dominant** ($P - S = 2 > R - T = 1$): misjudging the partner costs more when hunting stag than when hunting hare.

The game repeats and there is a single state, so each agent keeps just two numbers, $Q(C)$ and $Q(D)$, and uses **stateless Q-learning**:

$$Q(a) \leftarrow Q(a) + \alpha\,\big(r - Q(a)\big)$$

where $a$ is the action just played (only its value is updated), $r$ the payoff received, and $\alpha = 0.1$ the learning rate. The $\gamma \max_{a'} Q(s', a')$ term of regular Q-learning is dropped: with one state it adds the same amount to both actions' targets, so it never changes which action is greedy. Actions are [[Epsilon-Greedy Policy|ε-greedy]] (random with probability ε, otherwise the larger Q, ties broken at random). Each agent treats the learning partner as part of the environment ([[Independent Learning]]), so the payoff of each action keeps changing.

### Learners starting at zero lock into (D,D)

Both start at $Q = (0, 0)$ with $\varepsilon = 0.1$. At first both break ties at random, so each faces a 50/50 partner:

$$\mathbb{E}[r \mid C] = \tfrac12 \cdot 4 + \tfrac12 \cdot 0 = 2, \qquad \mathbb{E}[r \mid D] = \tfrac12 \cdot 3 + \tfrac12 \cdot 2 = 2.5.$$

D earns more on average and **never less than 2**, while C earns 0 whenever the partner plays D. A run reaches (C,C) essentially only if both play C in round 1 (probability 1/4). Otherwise it locks into (D,D) through a feedback loop: once my partner plays D, C pays me 0, so I never see evidence that C could pay 4, and my D in turn makes C pay 0 for my partner. Once both play D, $Q(D)$ goes to 2, and $Q(C)$ is updated only during exploration, where it mostly earns 0. Exploration then knocks most remaining (C,C) runs over to (D,D) (see below), so learning does worse than Lab 1's rational agents and replicator dynamics.

**The Prisoner's Dilemma check.** The same learners in the Prisoner's Dilemma always end at (D,D). That is **not a coordination failure**: D is dominant, so (D,D) is the only Nash equilibrium and the learners found it. The bad outcome there is a [[Social Dilemma|social dilemma]] (an incentive problem). In the Stag Hunt (C,C) is itself an equilibrium that the learners could have reached and missed: that is a coordination failure.

### Initial values as beliefs

An agent that believes its partner hunts stag with probability $b$ can start with the expected payoffs as its values:

$$Q(C) = 4b + 0(1-b) = 4b, \qquad Q(D) = 3b + 2(1-b) = b + 2.$$

It starts with C when $4b > b + 2$, so **$b > 2/3$**, exactly Lab 1's threshold: its first greedy action is the best response to its belief. With little exploration ($\varepsilon = 0.01$), two optimists stay at (C,C) and two pessimists at (D,D): play confirms the beliefs, so equilibrium selection moves into the initialisation.

**An optimist ($b = 0.9$) meets a sceptic ($b = 0.5$).** The optimist starts with $Q = (3.6, 2.9)$ and plays C; the sceptic starts with $Q = (2.0, 2.5)$ and plays D. The optimist earns 0, so each round its $Q(C)$ falls by 10%: $3.6 \to 3.24 \to 2.92 \to 2.62$, below its $Q(D) = 2.9$, and it switches to D. The sceptic earns 3 and keeps D; it never plays C, so its $Q(C)$ is never updated and it never finds out that stag could pay 4. They end at (D,D): **the agent whose action is punished does the adapting.**

### Exploration knocks learners out of (C,C), almost never out of (D,D)

Start both agents at (C,C) with $Q = (3.6, 2.9)$. With $\varepsilon = 0.1$ the partner plays D in 5% of rounds, so C still earns $0.95 \cdot 4 = 3.8$ against $0.95 \cdot 3 + 0.05 \cdot 2 = 2.95$ for D: C is the best response, and a rational agent would stay. Yet most learner pairs leave (C,C) within a few thousand rounds, and the higher ε, the faster.

The reason is what $Q(C)$ estimates. With a constant learning rate, $Q(C)$ is an **exponentially weighted average of the recent payoffs** of C (it remembers roughly the last $1/\alpha = 10$ payoffs): it tracks "what C paid lately", which is noisy. Each exploratory D by the partner pays me 0 and cuts $Q(C)$ by 10%: three such zeros in a short stretch take it $3.8 \to 3.42 \to 3.08 \to 2.77$, below $Q(D) \approx 2.95$. Then I play D, my partner earns 0 every round, and it follows within a few rounds. **One agent's exploration is noise in the other's rewards**, and the learner reads it as a change in the partner.

The opposite escape is far less likely. At (D,D), $Q(C)$ must climb above $Q(D) \approx 2$. But $Q(C)$ is updated only when I explore C, and pays 4 only if my partner explores C in the same round, a chance of $(\varepsilon/2)^2 = 0.25\%$ per round; all other C plays pay 0 and pull it back down. Even then both must switch together, since an agent that switches alone earns 0 and switches back. In basin terms: leaving (C,C) needs the partner to look like a hare hunter more than $1/3$ of the time, leaving (D,D) needs it to look like a stag hunter more than $2/3$ of the time. So more exploration does not help coordination: it destroys (C,C) and leaves (D,D) intact.

## Part II: the Coin Game with tabular Q-learning

### The game

A grid-world social dilemma like the repeated Prisoner's Dilemma. Two agents, red and blue, move on a $3 \times 3$ grid that wraps around at the edges and always holds one coin, red or blue. Both move at the same time (up, down, left, right); an agent picks up the coin by moving onto its cell, and a new coin of random colour appears. An episode lasts 16 steps.

| pickup | picker | coin's owner | total |
|---|---|---|---|
| own coin | +1 | (the picker) | +1 |
| other's coin | +1 | −2 | −1 |

A pickup pays the picker **+1 whatever the colour**, and **the coin's owner gets −2 when the other agent takes it**. Going for your own coins and leaving the other's alone is the cooperative pattern; taking every reachable coin is defection. The lab measures pickups per episode (random agents: about 3.7), restraint (the share of own-colour pickups: 0.5 for colour-blind agents) and welfare (total reward). From the table, welfare = pickups × (2 × restraint − 1), so colour-blind pickups give welfare near 0.

Each agent runs regular [[Q-Learning]] with a table, $Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma\max_{a'}Q(s',a') - Q(s,a))$, with $\gamma = 0.95$ (the look-ahead now credits moves towards a coin). The state is own cell, other's cell, coin cell and whether the coin is mine: $9 \cdot 9 \cdot 9 \cdot 2 = 1{,}458$ rows. Each agent treats the other as part of the environment.

### Why the learners take the other's coins

The learners get good at the individual task (several times more pickups than random agents), but their restraint stays near one half (about 56% own colour), so their welfare stays low, not far above that of random agents (about 0).

**Why:** an agent's own reward ignores the cost it causes. Taking a coin pays +1 whatever its colour. Leaving the other's coin pays 0, and if the other then takes it, still 0. The −2 from taking the other's coin falls on the other agent and never enters the taker's Q-values. So for its own reward taking every reachable coin is always at least as good as leaving it, the role D plays in the Prisoner's Dilemma. When both agents do this, half the pickups add +1 to the total and half add $+1 - 2 = -1$:

$$\mathbb{E}[\text{total reward per pickup}] = \tfrac12(+1) + \tfrac12(-1) = 0.$$

The share sits a little above 0.5 without being cooperation: grabbing my own coin before the other does also saves me the −2, so a selfish agent races harder for its own coins.

**Why independent Q-learning cannot find restraint by itself:** each agent maximises its own return and treats the other as part of the environment. Restraint gives up +1 and helps only the other, and the state holds positions and the coin with no record of whether the other has been restrained, so neither agent can reward restraint or punish taking (no reciprocity, as Tit-for-Tat had in Lab 1). Against any fixed behaviour of the other, taking every coin is the best response, so the result is mutual defection, as in the Prisoner's Dilemma.

### Irrelevant lamps break the table

Each agent's observation gets $k$ **lamps**, switched on or off at random every step, with no effect on the game. The best policy ignores them. For a table, two observations that differ in one lamp are different rows:

$$N = 1{,}458 \cdot 2^k: \quad 1{,}458 \text{ for } k = 0, \quad 23{,}328 \text{ for } k = 4, \quad 373{,}248 \text{ for } k = 8.$$

The **training budget** is fixed, counted in decisions per agent (each decision updates one row):

$$B = \text{updates} \times \text{parallel games} \times \text{steps per episode} = 300 \times 32 \times 16 = 153{,}600.$$

That is about 105 decisions per row without lamps and about 0.4 with 8 lamps.

With 8 lamps performance drops to near random. The number of rows grows by $2^k$ while the budget stays, so most rows are never visited and the rest only a few times. An unvisited row still holds zeros, so the agent acts at random there. And a table cannot discover that the lamps do not matter, because **rows share nothing**: what is learned under one lamp pattern is never used under another, so the same lesson has to be learned 256 times over, from 1/256 of the data each time.

A learner needs **generalisation** to be unaffected: a [[Function Approximation|function]] of the observation with weights shared across all inputs, such as a neural network. Then what it learns about the coin's position applies under every lamp pattern, and it can learn to give the lamps little weight. The price (from [[AGAI-L04 - Deep RL and Intro to Deep MARL|Lecture 4]]): instability from moving targets and correlated data, possible divergence, no general convergence guarantee, and more computation.

## Part III: PPO in the Coin Game

Each agent runs its own [[PPO]] learner (independent PPO): an actor network that outputs action probabilities and is sampled (which is where its exploration comes from, kept alive by an entropy bonus), and a critic network that estimates the state value. [[AGAI-L04 - Deep RL and Intro to Deep MARL|Lecture 4]] covers the method.

- **With 8 lamps PPO is hardly affected**, while the table collapses: the shared weights generalise across lamp patterns.
- **Without lamps the table does fine**, and each update costs it less computation than a network update: a small state space visited often is a table's best case.
- **PPO does not solve the social dilemma**: each agent still optimises its own reward with the other as environment, so taking every coin stays its best response. **Deep RL solved generalisation, but not the social dilemma**, which comes from the incentives.
- PPO learners in the Stag Hunt would still face equilibrium selection: with one state there is nothing to generalise, and stag still pays only when the partner hunts stag more than 2/3 of the time.

## What to be able to answer

- The stateless update, and why the $\gamma\max$ term can be dropped with one state.
- Expected payoffs against a 50/50 partner (C 2, D 2.5) and why learners from zero lock into (D,D).
- Why the Prisoner's Dilemma result is a social dilemma and the Stag Hunt result a coordination failure.
- $Q(C) = 4b$, $Q(D) = b + 2$, so C when $b > 2/3$; the optimist against the sceptic.
- Why exploration knocks learners out of (C,C) and almost never out of (D,D).
- The Coin Game rewards, why taking every coin is best for one's own reward, expected total 0 per pickup.
- Rows $1{,}458 \cdot 2^k$, budget $B$, why lamps ruin a table and how a network avoids it, and what a network costs.
- "Deep RL solved generalisation, but not the social dilemma."

## Exam questions

> [!exam]- Mock Q12 (open, 10 points): Two independent Q-learners play a Stag Hunt with R = 4, S = 0, T = 3, P = 2. They use the stateless update Q(a) ← Q(a) + α(r − Q(a)) and ε-greedy exploration. (a) Early in learning the partner plays almost at random. Compute the expected payoff of C and of D against a 50/50 partner. (2 points) (b) Both agents start at Q = (0, 0). Explain why they usually lock into (D, D). (2 points) (c) Explain why escaping from (C, C) to (D, D) is much more likely than escaping in the other direction. (3 points) (d) An agent with belief b about its partner starts with Q(C) = 4b and Q(D) = 3b + 2(1 − b). For which b does it start by playing C? What happens when an agent with b = 0.9 meets an agent with b = 0.5, with little exploration? (3 points)
> **Key points:** C earns 2, D earns 2.5; D always earns at least 2 while C earns 0 against D, and once both play D, Q(C) is updated only in exploration, where it mostly earns 0; from (C, C) a few exploratory D moves push Q(C) below Q(D), while leaving (D, D) needs both agents to explore C in the same rounds; b > 2/3; the b = 0.9 agent earns 0 and its Q(C) falls, the b = 0.5 agent earns 3 and keeps D, so they end at (D, D).
>
> **(a)** C: $0.5 \cdot 4 + 0.5 \cdot 0 = 2$. D: $0.5 \cdot 3 + 0.5 \cdot 2 = 2.5$.
>
> **(b)** D always earns at least 2. C only pays off if the partner also plays C at that moment, and it earns 0 otherwise, so it is riskier than receiving 2 by playing D. Once both play D, $Q(D)$ goes to 2, and $Q(C)$ is updated only during exploration, where it mostly earns 0 (the partner plays D). So $Q(C)$ never gets evidence that C can pay 4. Bonus: (D, D) is the risk-dominant equilibrium.
>
> **(c)** From (C, C), a few exploratory D moves by one agent lower the other's $Q(C)$ below $Q(D)$: playing C against D pays 0, and with a constant learning rate each 0 cuts $Q(C)$ by 10% ($3.8 \to 3.42 \to 3.08 \to 2.77 < 2.95$). That agent switches to D, and its partner then earns 0 and follows. From (D, D), $Q(C)$ must rise above $Q(D)$, which needs both agents to explore C in the same rounds (chance $(\varepsilon/2)^2$ per round), several times, and both must switch together. That is far less likely.
>
> **(d)** $4b > b + 2$ gives $b > 2/3$. With $b = 0.9$ the agent plays C ($Q = (3.6, 2.9)$), and with $b = 0.5$ the agent plays D ($Q = (2, 2.5)$). The first agent earns 0, and its $Q(C)$ falls quickly ($3.6 \to 3.24 \to 2.92 \to 2.62 < 2.9$, so it switches after three rounds). The second earns 3 and keeps D; it never plays C, so it never learns that stag can pay 4. They end at (D, D).

> [!exam]- Mock Q21 (open, 10 points): Tabular Q-learners play the Coin Game, whose table has 1,458 rows. Each of k random lamps doubles the number of rows (as lamps can be on or off). Training plays 300 updates, each with 32 parallel games of 16 steps, and every step gives one decision per agent. Coins per episode are 14.5 with no lamps and 4.9 with 8 lamps, while random agents collect 3.7. (a) Compute the number of rows N for k = 0 and k = 8, and the budget B of decisions per agent. (3 points) (b) Use these numbers to explain the drop in performance. Why can't agents with tabular learning discover that the lamps do not matter? (3 points) (c) Explain how a neural network avoids this problem. Name one new challenge it introduces. (4 points)
> **Key points:** $N = 1{,}458$ for $k = 0$ and $1{,}458 \cdot 2^8$ for $k = 8$, $B = 300 \cdot 32 \cdot 16$; rows grow while the budget stays, so most rows are never visited and unvisited rows hold zeros (random play); rows that differ in one lamp share nothing, so each lesson is relearned for every lamp pattern; a network computes values or a policy from the observation with shared weights, so what it learns about the coin's position applies to all lamp patterns and the lamps get little weight; new challenges: instability from moving targets or correlated data, possible divergence, no general convergence guarantee, more computation.
>
> **(a)** $N = 1{,}458$ for $k = 0$, and $N = 1{,}458 \cdot 2^8 = 373{,}248$ for $k = 8$. $B = 300 \cdot 32 \cdot 16 = 153{,}600$ decisions per agent.
>
> **(b)** With 8 lamps, most rows are never visited, and the others only a few times (about 0.4 decisions per row, against about 105 without lamps). An unvisited row still holds zeros, so the agent acts at random there. The number of rows grows but the budget remains, so the probability of visiting each state becomes too low, and performance falls from 14.5 coins to 4.9, close to random (3.7). Two rows that differ in one lamp share nothing, so each lesson must be learned again for each of the other lamp patterns; that is why the table cannot discover that the lamps do not matter.
>
> **(c)** A network computes values or a policy as a function of the observation, with shared weights. What it learns about the coin's position applies to all lamp patterns, and it can give the lamps little weight. New challenges: instability from moving targets or correlated data, possible divergence, no general convergence guarantee, and more computation needed.

> [!exam]- Two independent Q-learners play a Prisoner's Dilemma with R = 3, S = 0, T = 5, P = 1, using the stateless update and ε-greedy exploration. (a) Compute the expected payoff of C and of D against a 50/50 partner. (2 points) (b) The learners always end at (D, D). Is this a coordination failure, as in the Stag Hunt? Explain. (3 points) (c) An agent with belief b that its partner cooperates starts with Q(C) = 3b and Q(D) = 5b + (1 − b). For which b does it start with C? (2 points) (d) Both agents start with Q = (3, 1) and play (C, C). Explain why exploration makes them leave. (3 points)
> **Key points:** C: 1.5, D: 3; no: D is dominant (T > R, P > S), so (D, D) is the only Nash equilibrium and the learners found it, the bad outcome is a social dilemma; 3b > 4b + 1 has no solution in [0, 1], so D for every belief; exploratory D earns T = 5 > R = 3 and raises Q(D), the partner's exploratory D gives 0 and lowers Q(C), so Q(D) overtakes Q(C), and then the partner earns 0 and follows.
>
> **(a)** C: $0.5 \cdot 3 + 0.5 \cdot 0 = 1.5$. D: $0.5 \cdot 5 + 0.5 \cdot 1 = 3$.
>
> **(b)** No. D is dominant: against C it earns $T = 5 > R = 3$, against D it earns $P = 1 > S = 0$. So (D, D) is the only Nash equilibrium, and the learners found it. (C, C) gives both more, but it is not an equilibrium, so there was nothing better to coordinate on. The bad outcome is a social dilemma, an incentive problem. In the Stag Hunt (C, C) is an equilibrium that the learners miss, which is a coordination failure.
>
> **(c)** $Q(D) = 4b + 1$, and $3b > 4b + 1$ would need $b < -1$. No belief makes C the first choice: D is the best response to anything.
>
> **(d)** At (C, C) each agent's $Q(C)$ sits near 3. When an agent explores D against a cooperating partner it earns 5, so its $Q(D)$ rises towards 5, and when the partner explores D, the agent's C earns 0, which lowers $Q(C)$. After a few explorations $Q(D) > Q(C)$ and the agent defects. Its partner's C now earns 0, so the partner follows, and they end at (D, D).

> [!exam]- In the Coin Game a pickup pays +1 to the picker whatever the coin's colour, and the coin's owner gets −2 when the other agent takes it. Two independent Q-learners pick up many coins, but only about half are their own colour. (a) Show that welfare (total reward) = pickups × (2 × own share − 1), and compute the expected total reward per pickup for colour-blind agents. (3 points) (b) Explain, from one agent's own reward, why taking the other's coin is never worse than leaving it. (3 points) (c) Why can independent Q-learning not learn restraint by itself? (2 points) (d) Would independent PPO agents behave differently? (2 points)
> **Key points:** own pickup adds +1 to the total, other's pickup +1 − 2 = −1, so welfare = pickups × (share − (1 − share)); colour-blind: 0 per pickup; taking pays +1, leaving pays 0 now and 0 if the other takes it, the −2 falls on the other agent; the other is part of the environment and the state has no record of its behaviour, so no reciprocity, and taking is the best response to any fixed behaviour; PPO generalises better but optimises the same selfish reward, so it also takes the other's coins.
>
> **(a)** A pickup of one's own coin adds +1 to the total; a pickup of the other's coin adds $+1 - 2 = -1$. With $n$ pickups and own share $p$, welfare $= np - n(1-p) = n(2p - 1)$. Colour-blind agents have $p = 1/2$, so the expected total per pickup is $\frac12 - \frac12 = 0$.
>
> **(b)** Taking the coin pays the agent +1. Leaving it pays 0 now, and if the other then takes it, the agent still gets 0 (it is the other's coin). The −2 that taking causes falls on the other agent, so it never enters the taker's reward or Q-values. Taking is at least as good in every state, like D in the Prisoner's Dilemma.
>
> **(c)** Each agent maximises its own return and treats the other as part of the environment. Restraint costs +1 and helps only the other, and the state holds positions and the coin with no record of the other's past behaviour, so neither agent can reward restraint or punish taking. Against any fixed behaviour of the other, taking every coin is the best response.
>
> **(d)** No. PPO handles irrelevant inputs better, but each PPO agent still optimises its own reward with the other treated as environment, so taking every coin is still its best response: restraint stays near 0.5 and welfare near 0.

> [!exam]- Tabular Q-learners play a Coin Game on a 4 × 4 grid. The state is the agent's own cell, the other agent's cell, the coin's cell and the coin's colour. Training plays 500 updates of 32 parallel games of 20 steps each, one decision per agent per step. (a) Compute the number of rows N without lamps and with 6 random lamps, and the budget B. (3 points) (b) Use B/N to explain what happens to performance with 6 lamps. (3 points) (c) Does adding a random lamp change the optimal policy? What does it do to the table? (2 points) (d) Why does a neural network not need to visit every row? (2 points)
> **Key points:** $N_0 = 16 \cdot 16 \cdot 16 \cdot 2 = 8{,}192$, $N_6 = 8{,}192 \cdot 2^6 = 524{,}288$, $B = 500 \cdot 32 \cdot 20 = 320{,}000$; about 39 decisions per row without lamps, about 0.6 with 6, so most rows unvisited, zeros, random play; the optimal policy is unchanged, but each lamp doubles the rows and values are not shared between rows; shared weights generalise from similar observations.
>
> **(a)** $N_0 = 16 \cdot 16 \cdot 16 \cdot 2 = 8{,}192$. $N_6 = 8{,}192 \cdot 2^6 = 524{,}288$. $B = 500 \cdot 32 \cdot 20 = 320{,}000$ decisions per agent.
>
> **(b)** Without lamps there are about $320{,}000 / 8{,}192 \approx 39$ decisions per row, enough to learn something in most rows. With 6 lamps there are about 0.6 per row, so most rows are never visited and the rest only once or twice. Unvisited rows hold their initial zeros, so the agent acts at random there, and performance falls towards that of random agents.
>
> **(c)** No: the lamp has no effect on the game, so the best policy ignores it. But each lamp doubles the number of rows, and values are not shared between rows that differ only in the lamp, so the table must learn everything again for each lamp pattern.
>
> **(d)** A network computes its output from the observation with weights shared across all inputs. Learning from one observation changes the output for similar observations too, so it can generalise to observations it has never seen and learn to give the lamps little weight.

## Flashcards

> [!card]- What is the stateless Q-learning update?
> $Q(a) \leftarrow Q(a) + \alpha(r - Q(a))$, only for the action just played.

> [!card]- Why can the $\gamma \max_{a'} Q(s',a')$ term be dropped in a repeated game with one state?
> It adds the same amount to both actions' targets, so it never changes which is greedy.

> [!card]- In the Stag Hunt with $R=4, S=0, T=3, P=2$, what do C and D earn against a 50/50 partner?
> C earns 2, D earns 2.5.

> [!card]- Why do Stag Hunt Q-learners starting at $Q = (0,0)$ lock into (D,D)?
> Once the partner plays D, C pays 0, so Q(C) never sees that C can pay 4.

> [!card]- With $Q(C) = 4b$ and $Q(D) = 3b + 2(1-b)$ in the Stag Hunt, when does an agent start with C?
> When $b > 2/3$.

> [!card]- An optimist (belief 0.9) meets a sceptic (belief 0.5) as Q-learners in the Stag Hunt. Who adapts?
> The optimist: its C earns 0, so its Q(C) falls until D is greedy.

> [!card]- With a constant learning rate, what does a stateless Q-learner's $Q(a)$ estimate?
> An exponentially weighted average of the recent payoffs of $a$.

> [!card]- Why do Q-learners leave (C,C) in the Stag Hunt although C is the best response?
> The partner's exploratory D moves pay 0, and a few zeros push Q(C) below Q(D).

> [!card]- Why is escaping from (D,D) to (C,C) far rarer than the reverse?
> C pays 4 only when both explore C in the same round, and both must switch together.

> [!card]- Why is the Prisoner's Dilemma learners' (D,D) not a coordination failure?
> D is dominant, so (D,D) is the only Nash equilibrium: a social dilemma.

> [!card]- In the Coin Game, what happens when an agent takes the other player's coin?
> The picker gets +1 and the coin's owner gets −2.

> [!card]- Why does a self-interested Coin Game agent take every reachable coin?
> The −2 falls on the other agent, so its own reward ignores that cost.

> [!card]- What is the expected total reward per pickup when both Coin Game agents are colour-blind?
> 0: half the pickups add +1, half add −1.

> [!card]- Why can independent Q-learners not learn restraint in the Coin Game?
> The state holds no record of the other's behaviour, so restraint cannot be reciprocated.

> [!card]- How many rows does the Coin Game Q-table have with $k$ irrelevant lamps?
> $1{,}458 \cdot 2^k$.

> [!card]- How is the training budget $B$ of the Coin Game counted?
> Decisions per agent: updates × parallel games × steps per episode.

> [!card]- Why can a Q-table not discover that the lamps do not matter?
> Rows share nothing: each lamp pattern must be learned separately.

> [!card]- Complete for the Coin Game: "Deep RL solved ..., but not ...".
> Generalisation over irrelevant inputs, but not the social dilemma.

> [!card]- True or false: more exploration helps Stag Hunt Q-learners reach (C,C).
> **False.** Exploration knocks them out of (C,C) and leaves (D,D) intact.

> [!card]- Multiple selection: why do Stag Hunt Q-learners starting at zero mostly end at (D,D)? (A) D earns more against a 50/50 partner. (B) D is dominant. (C) C pays 0 whenever the partner plays D. (D) (D,D) is risk-dominant.
> **A, C and D.** B: against C, stag (4) beats hare (3), so D is not dominant.

## Links

- **Course:** [[AGAI - Overview|Agentic AI overview]]
- **Lectures:** [[AGAI-L03 - Intro to Multi-Agent Reinforcement Learning]] (independent learning, equilibrium selection) · [[AGAI-L04 - Deep RL and Intro to Deep MARL]] (why tables stop working, PPO)
- **Previous lab:** [[AGAI-Lab01 - Simple Agents, Complex Dynamics]] (threshold $p^* = 2/3$, replicator dynamics, Tit-for-Tat)
- **Concepts:** [[Q-Learning]] · [[Independent Learning]] · [[Stag Hunt]] · [[Prisoner's Dilemma]] · [[Social Dilemma]] · [[Nash Equilibrium]] · [[Epsilon-Greedy Policy]] · [[Exploration vs Exploitation]] · [[Tabular RL]] · [[Function Approximation]] · [[Neural Network Function Approximation]] · [[Deep Reinforcement Learning]] · [[PPO]] · [[Actor-Critic]]
- **Source:** `AGAI-Lab02.ipynb`, Agentic AI 2026/2027 (UvA MSc AI), Lab 2, "Learning Agents", werkcollege 4 Nov 2026. The Coin Game is from Lerer and Peysakhovich (2017).
