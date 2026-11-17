---
type: lecture
course: 5204AGAI6Y
week: 1
lecture: 2
date: 2026-10-29
book_sections: ["Albrecht, Christianos and Schäfer Ch 3", "Albrecht, Christianos and Schäfer Ch 4"]
topics:
  - "[[Normal-Form Game]]"
  - "[[Repeated Game]]"
  - "[[Stochastic Game]]"
  - "[[Best Response]]"
  - "[[Nash Equilibrium]]"
  - "[[Pareto Optimality]]"
  - "[[Social Dilemma]]"
  - "[[Prisoner's Dilemma]]"
  - "[[Stag Hunt]]"
  - "[[Replicator Dynamics]]"
status: complete
---

# AGAI-L02: Game Theory and Population Dynamics

> [!abstract] In short
> Four game models describe interacting agents, from a one-shot payoff matrix to a partially observable stochastic game. A Nash equilibrium (everyone best-responds to the others) predicts what self-interested agents do; Pareto optimality and welfare say which outcomes are good. In the Prisoner's Dilemma and the Stag Hunt the two disagree: social dilemmas.
>
> Equilibria do not say how agents get there. Under the replicator equation of evolutionary game theory, a strategy spreads when it earns more than the population average: the Prisoner's Dilemma ends in all-D, the Stag Hunt tips to all-C or all-D depending on the start, and Hawk-Dove settles in a stable mix.
>
> Feeds mock Q1 (parts of a stochastic game), Q2 (picking a model), Q3 (Nash limitations), Q4 and Q5 (replicator equation) and the open Q6 (Prisoner's Dilemma, also Lab 1 Q2).

## Simple rules, surprising group behaviour

Simple reflex agents (they act on the current percept alone, see [[Agent Program]]) already show **emergence**, group behaviour nobody programmed. **Boids** flock from three local rules: separation (move away from nearby agents), alignment (steer towards their average direction) and cohesion (move towards their average position). Conway's **Game of Life** gets moving shapes from birth and death rules on a grid.

## Games as models of multi-agent interaction

A game formally defines a multi-agent interaction. Four models, each a special case of the next:

| model | use it when |
|---|---|
| one-shot normal-form game | agents act once, at the same time; payoffs depend only on the actions |
| repeated normal-form game | the same game is played again and again; no state |
| stochastic game | a state changes with the joint action, and everyone sees it |
| partially observable stochastic game (POSG) | agents see only part of the state or of others' actions |

### Normal-form games: actions and rewards only

> [!definition] Normal-form game
> Defines the interaction between 2 or more agents: no time, no state, only actions and rewards. A 3-tuple $(I, \{A_i\}_{i \in I}, \{R_i\}_{i \in I})$:
> - $I$: a finite set of agents
> - $A_i$: a finite set of actions for each agent $i$
> - $R_i : A \to \mathbb{R}$: the reward function of agent $i$, where $A = A_1 \times \dots \times A_n$ is the set of **joint actions** (one action per agent)

Each reward depends on what everyone did. With two agents the game is a matrix; each cell gives (row reward, column reward).

**[[Prisoner's Dilemma]]** (C = cooperate, D = defect): D pays more whatever the other does, yet both prefer (C, C) to the (D, D) they end up in.

| | C | D |
|---|---|---|
| **C** | 3, 3 | 0, 5 |
| **D** | 5, 0 | 1, 1 |

**[[Stag Hunt]]**: the best action is to copy the other agent. (C, C) is best for both, but cooperating alone earns 0, so C is risky.

| | C | D |
|---|---|---|
| **C** | 4, 4 | 0, 3 |
| **D** | 3, 0 | 2, 2 |

**Zero-sum** games: rewards in each cell sum to zero (Matching Pennies). **Common-reward**: everyone gets the same reward. **General-sum**: anything else, such as these two games.

Each agent follows a [[Policy|policy]] $\pi_i$, a probability distribution over its actions. A policy that always picks the same action is **pure**; one that randomises is **mixed**.

### Repeating the game: policies with memory

In a [[Repeated Game|repeated normal-form game]] a policy can condition on the history of joint actions, $\pi_i(a_i^t \mid h^t)$ with $h^t = (a^0, \dots, a^{t-1})$. **Tit-for-Tat (TFT)** cooperates first, then copies the opponent's last action. **Win-Stay Lose-Shift (WSLS)** cooperates first, keeps its action after a top payoff (3 or 5 in the Prisoner's Dilemma) and switches otherwise. The number of steps is crucial: always-defect gains once by exploiting TFT in round 1, which matters in a short game; over many rounds, reciprocators earning 3 per round with each other pull ahead ([[AGAI-Lab01 - Simple Agents, Complex Dynamics|Lab 1]]).

### Adding states: stochastic games

A [[Stochastic Game|stochastic game]] adds **states** that change with the joint action; each state can have its own payoff matrix.

> [!definition] Stochastic game
> A 6-tuple $(I, S, \{A_i\}, \{R_i\}, T, \mu)$:
> - $I$: a finite set of agents
> - $S$: a finite set of states
> - $A_i$: a finite set of actions for each agent $i$
> - $R_i(s, a, s')$: the reward function of each agent $i$ ($a$ is the joint action)
> - $T(s' \mid s, a)$: the transition function, the probability of next state $s'$ given state $s$ and joint action $a$
> - $\mu$: the initial state distribution

An action matters twice: for its reward and for which state comes next. Example: two robots collecting apples on a grid they both see. With one state it is a repeated normal-form game; with one agent, a [[Markov Decision Process]]. **Not** part of the definition: an observation function (a POSG adds that) and a policy (agents choose policies; the game does not contain them).

### Hiding the state: partially observable stochastic games

A **POSG** models limited sensing: agents receive observations that give incomplete information about the state and others' actions. Examples: **autonomous driving** (a car cannot see behind a truck) and **card games** with hidden hands.

> [!definition] Partially observable stochastic game (POSG)
> An 8-tuple: a stochastic game plus, for each agent $i$,
> - $O_i$: a finite set of observations
> - $\mathcal{O}_i(o_i \mid a, s)$: the observation function, the probability that agent $i$ observes $o_i$ when joint action $a$ led to state $s$

Since the agent does not see the full state, its policy conditions on its own observation history, $h_i^t = (o_i^0, \dots, o_i^{t-1})$.

## Solution concepts: what will agents do, and what is good?

Solution concepts predict which actions agents play and judge how desirable a **joint policy** $\pi = (\pi_1, \dots, \pi_n)$ is. They use $U_i(\pi)$, the **expected return** of agent $i$. Against a partner who plays C with probability $q$, C earns $qR + (1-q)S$ in expectation ($R$: C against C, $S$: C against D).

### Best response

Which action is best depends on what the others do (in the Stag Hunt, C against C and D against D), so optimality is defined against their fixed policies.

> [!definition] Best response
> A best-response policy maximises the expected return for agent $i$ against a given set of policies for all other agents $\pi_{-i}$:
> $$BR_i(\pi_{-i}) = \arg\max_{\pi_i} U_i(\langle \pi_i, \pi_{-i} \rangle)$$
> where $\langle \pi_i, \pi_{-i} \rangle$ is the complete joint policy.

$BR_i$ is a set: several policies can tie. An action that is a best response to every action of the others is **dominant** (D in the Prisoner's Dilemma).

### Nash equilibrium: everyone best-responds at once

> [!definition] Nash equilibrium (NE)
> Applies the idea of a **mutual best response** to general-sum games with two or more agents. No agent can improve its expected return by changing its policy, assuming the other agents' policies remain fixed:
> $$\forall i, \; \forall \pi_i' : \quad U_i(\langle \pi_i', \pi_{-i} \rangle) \le U_i(\pi)$$

John Nash (1950) proved that every finite general-sum normal-form game has at least one Nash equilibrium, once mixed policies are allowed.

**Pure equilibria in a matrix:** mark the row agent's best reward in each column and the column agent's best reward in each row; a cell marked twice is a pure NE. **Mixed equilibria:** an agent only mixes over actions that earn the same against the other's policy (if one earned more, it would drop the others).

> [!example] Three games
> **Prisoner's Dilemma** (4, 4; $-2$, 6; 6, $-2$; 0, 0): D is best against C and against D, so only (D, D).
>
> **Stag Hunt** (4, 4; 0, 3; 3, 0; 2, 2): (C, C) and (D, D), plus a mixed one. Against a partner playing C with probability $q$, C earns $4q$ and D earns $2 + q$, equal at $q = 2/3$.
>
> **Rock-Paper-Scissors**: from every pure cell someone gains by switching, so the only equilibrium is mixed, each action with probability 1/3.

### What a Nash equilibrium does not tell you

- **Sub-optimality:** equilibria do not always maximise expected returns. An equilibrium can be worse for all players than another outcome: (D, D) against (C, C) in the Prisoner's Dilemma.
- **Non-uniqueness:** there can be multiple, even infinitely many, equilibria with different expected returns, and the concept does not say which one will be played (the Stag Hunt has three).
- **Incompleteness:** an equilibrium does not specify actions off the equilibrium path (after someone deviates), and does not say how agents reach it.
- **Computation** (from the mock exam's answer key; the course lists only the first three): computing an equilibrium can be hard in general games.

Not limitations: every finite game has an equilibrium (possibly mixed), and the concept applies to any number of players. The first two lead to refinements such as Pareto optimality; the third leads to evolutionary game theory.

### Desirable outcomes: Pareto optimality, welfare and fairness

> [!definition] Pareto optimality
> A joint policy $\pi$ is **Pareto-dominated** by $\pi'$ if $U_i(\pi') \ge U_i(\pi)$ for all agents $i$ and $U_i(\pi') > U_i(\pi)$ for at least one. A joint policy is **Pareto-optimal** if no other joint policy Pareto-dominates it: nothing improves the expected return of one agent without reducing it for another.

In the Stag Hunt only (C, C) = (4, 4) is Pareto optimal. In the Prisoner's Dilemma, (C, C), (C, D) and (D, C) all are: moving away from (C, D) = (0, 5) hurts the defector, even though the cooperator has its worst payoff. Pareto optimality rules out waste and says nothing about fairness.

> [!formula] Welfare and fairness
> $$W(\pi) = \sum_{i} U_i(\pi), \qquad F(\pi) = \prod_{i} U_i(\pi)$$
> $\pi$ is **welfare-optimal** if it maximises $W$ and **fairness-optimal** if it maximises $F$.

The welfare-optimal outcome is the **social optimum**. Every social optimum is Pareto optimal, but not the reverse. The product rewards equal splits: (5, 5) and (9, 1) both have welfare 10, but fairness 25 against 9.

### Social dilemmas

A [[Social Dilemma|social dilemma]]: self-interested agents reach an outcome worse for the group than one they could have reached together. The **Prisoner's Dilemma** is a dilemma of cooperation: its only NE, (D, D), is the only outcome that is not Pareto optimal. The **Stag Hunt** is a dilemma of coordination (trust): the best outcome (C, C) is an equilibrium, but so is the worse (D, D).

## From rational players to adapting populations

### Evolutionary game theory

Real agents are often not so rational: they use trial and error, have incomplete information, and imitate others. So the question becomes how a population's behaviour changes over time, depending on what others do. In a Stag Hunt population with 50% C and 50% D, does C grow or shrink? Both pure outcomes are equilibria, so the static theory cannot say.

**Evolutionary game theory** is evolutionary biology plus game theory. Fit types spread, but fitness cannot be measured in isolation: **it depends on the frequency of other types**, as a payoff depends on others' strategies. No rationality is needed.

### The replicator equation

A large population, a fraction $x$ playing C and $1 - x$ playing D, is paired at random. The [[Replicator Dynamics|replicator equation]]:

> [!formula] Replicator equation
> $$\dot x = x \big( f_C(x) - \bar f(x) \big)$$
> - $x$: the fraction of individuals using strategy C
> - $f_C(x)$: the average payoff (fitness) of C when a fraction $x$ plays C
> - $\bar f(x) = x f_C(x) + (1-x) f_D(x)$: the average fitness in the population

**The assumption: C becomes more common when it earns more than the population average.** The factor $x$ means a strategy nobody uses cannot spread. With two strategies, substituting $\bar f$ gives $f_C - \bar f = (1-x)(f_C - f_D)$, so:

> [!formula] Replicator equation, two strategies
> $$\dot x = x(1 - x)\big(f_C(x) - f_D(x)\big)$$

$x(1-x)$ is positive whenever both strategies are present, so the sign of $f_C - f_D$ alone decides whether C grows. A partner plays C with probability $x$, so $f_C(x) = xR + (1-x)S$ and $f_D(x) = xT + (1-x)P$.

It does not assume that agents compute best responses or switch all at once, that a strategy grows because most agents already play it, or that Pareto-optimal outcomes win. It describes **biology** (replication and natural selection), **economics** (social learning, imitating those who do better) and **computer science** (reinforcement learning dynamics).

A **rest point** has $\dot x = 0$: $x = 0$, $x = 1$, and any $x$ with $f_C = f_D$. It is **stable** if the population returns after a small push, **unstable** otherwise. All-D is stable when $P > S$ (a lone cooperator among defectors earns less than they do); all-C when $R > T$.

### The Stag Hunt population: a tipping point

With $R = 4$, $S = 0$, $T = 3$, $P = 2$: $f_C = 4x$, $f_D = 2 + x$, so $\dot x = x(1-x)(3x - 2)$. Rest points 0 and 1 are stable, 2/3 is unstable: below 2/3, D earns more and C shrinks to 0; above it, C grows to 1.

**The 50/50 question:** at $x = 1/2$ a C agent earns 2 and a D agent 2.5, so C decreases, all the way to all-D. The tipping point 2/3 is the mixed Nash equilibrium, but no adapting population ends there. Which equilibrium is reached depends on where the population starts.

### The Prisoner's Dilemma in a population

In general $f_C - f_D = x(R - T) + (1-x)(S - P)$. In a Prisoner's Dilemma both terms are negative ($T > R$, $P > S$), so C shrinks from any start. A standard version is the **donation game**: cooperating costs $c$ and gives the partner $b$, with $b > c > 0$.

| | C | D |
|---|---|---|
| **C** | $b-c$, $b-c$ | $-c$, $b$ |
| **D** | $b$, $-c$ | 0, 0 |

Nash equilibrium (D, D); social optimum (C, C). In a population $f_C - f_D = -c$ always, so the population ends **fully D**. All-C is a rest point but unstable: one defector among cooperators earns $b > b - c$ and spreads.

### Hawk-Dove: coexistence

C = conventional, D = dangerous:

| | C | D |
|---|---|---|
| **C** | 3, 3 | 2, 8 |
| **D** | 8, 2 | $-2$, $-2$ |

The best response is the opposite action: D against C (8 > 3), C against D (2 > $-2$). The symmetric Nash equilibrium is **mixed**: C earns $2 + q$ and D earns $10q - 2$, equal at $q = 4/9$. (Outside the course material, which names only the mixed equilibrium: the asymmetric pure outcomes (C, D) and (D, C), where one agent yields, are also equilibria.) In a population, $\dot x = x(1-x)(4 - 9x)$, and it converges to a stable **coexistence of C and D** at 4/9, because each strategy earns more when it is rare.

### Equilibria against population outcomes

| game | Nash equilibria | social optimum | replicator outcome |
|---|---|---|---|
| Prisoner's Dilemma | (D, D) | (C, C) | all D, from any start |
| Stag Hunt | (C, C), (D, D), mixed at 2/3 | (C, C) | all C above 2/3, all D below |
| Hawk-Dove | mixed at 4/9 (also (C, D), (D, C), outside the course material) | (C, D) or (D, C) | stable mix at 4/9 C |

## What to be able to answer

- The four game models, the components of a stochastic game and a POSG, and which model fits a scenario.
- Best response, Nash equilibrium, Nash's existence result and the NE's limitations.
- Pure equilibria, Pareto optima and the social optimum of a 2x2 game (mock Q6 below).
- The replicator equation in both forms, what it assumes, and its outcome in the Prisoner's Dilemma, the Stag Hunt and Hawk-Dove.

## Exam questions

> [!exam]- (Mock Q6) A Prisoner's Dilemma has row payoffs R (C against C), S (C against D), T (D against C) and P (D against D), with T > R > P > S. The game is symmetric. (a) Show that D is the best response to both C and D. (b) Show that (D, D) is the only pure Nash equilibrium and that it is not Pareto optimal. (c) Show that (C, D) is Pareto optimal. Is it a good outcome? What does this say about Pareto optimality as a criterion? (d) Give the condition on the payoffs under which (C, C) is the social optimum.
> **Key points:** against C, T > R; against D, P > S, so D is dominant; any other cell has a player who gains by switching to D, and (C, C) gives both R > P; (C, D) = (S, T) gives the column player its maximum, so it is Pareto optimal but unfair; Pareto optimality ignores fairness; (C, C) is the social optimum if 2R ≥ T + S.
>
> **(a)** Against C, D earns T > R. Against D, D earns P > S. So D is a best response in both cases (a dominant action).
>
> **(b)** In any other outcome, a player who plays C can gain by switching to D, so only (D, D) is an equilibrium. It is not Pareto optimal, because (C, C) gives both players more payoff (R > P).
>
> **(c)** (C, D) leads to payoff (S, T). The column player gets its highest payoff T, so any improvement for the row player would cost the column player. So it is Pareto optimal, although the row player gets its lowest payoff. Pareto optimality says nothing about fairness; this is a limitation.
>
> **(d)** (C, C) is the social optimum if 2R ≥ T + S. ((D, D) never competes, since 2P < 2R.)

> [!exam]- A symmetric Stag Hunt has R = 4, S = 0, T = 3, P = 2. (a) Give the best response to C and to D. (b) Find the pure Nash equilibria and the Pareto-optimal outcomes. (c) Compute the expected payoff of C and of D against a partner who plays C with probability 1/2. (d) A large population plays this game under the replicator dynamics. What happens from 50% C, and from 80% C?
> **Key points:** best response to C is C (4 > 3), to D is D (2 > 0); pure NE (C, C) and (D, D), only (C, C) Pareto optimal; C earns 2, D earns 2.5; tipping point at 2/3; 50% goes to all D, 80% to all C.
>
> **(a)** Against C, C earns 4 and D 3, so C. Against D, C earns 0 and D 2, so D. The best response copies the partner.
>
> **(b)** (C, C) and (D, D) are both equilibria: in each, switching alone lowers the deviator's payoff. (C, D) and (D, C) are not: the agent playing C gains by switching. (C, C) = (4, 4) gives both agents more than any other cell, so it is the only Pareto-optimal outcome (and the social optimum). (D, D) is an equilibrium that is worse for both, the coordination dilemma.
>
> **(c)** C: 0.5 · 4 + 0.5 · 0 = 2. D: 0.5 · 3 + 0.5 · 2 = 2.5. D is the safer choice when unsure.
>
> **(d)** $f_C - f_D = 4x - (2 + x) = 3x - 2$, negative below 2/3 and positive above. From 50%, D earns more, so C shrinks to 0 (all D). From 80%, C earns more ($3.2$ against $2.8$), so C grows to 1 (all C). The interior point 2/3, the mixed equilibrium, is unstable.

> [!exam]- Hawk-Dove has (C, C) = 3, 3; (C, D) = 2, 8; (D, C) = 8, 2; (D, D) = $-2$, $-2$. (a) Give the best responses and the pure Nash equilibria. (b) Which outcomes are Pareto optimal, and which is the social optimum? (c) Find the symmetric mixed equilibrium. (d) What does the replicator dynamics predict for a large population, and why?
> **Key points:** best response is the opposite action; pure NE (C, D) and (D, C) (outside the course material); Pareto optimal (C, C), (C, D), (D, C), social optimum (C, D) and (D, C) with total 10; mixed NE C with probability 4/9; stable coexistence at 4/9 C because each strategy earns more when rare.
>
> **(a)** Against C, D is better (8 > 3); against D, C is better (2 > $-2$). So the pure equilibria are (C, D) and (D, C), where one agent yields. (The course names only the mixed equilibrium of part (c); the pure ones follow from the best responses but are outside the course material.)
>
> **(b)** (D, D) = ($-2$, $-2$) is dominated by every other cell. (C, C), (C, D) and (D, C) cannot be improved for one agent without hurting the other, so they are Pareto optimal. Totals: 6, 10, 10, $-4$, so the social optima are (C, D) and (D, C).
>
> **(c)** Partner plays C with probability $q$: C earns $3q + 2(1-q) = 2 + q$, D earns $8q - 2(1-q) = 10q - 2$. Equal at $q = 4/9$, so each agent plays C with probability 4/9.
>
> **(d)** $\dot x = x(1-x)(4 - 9x)$: C grows below 4/9 and shrinks above it, so the population converges to a stable mix of about 44% C. Each strategy earns more when it is rare: among conventional agents a dangerous one earns 8, among dangerous agents a conventional one earns 2 against their $-2$. Unlike the Stag Hunt, here the mixed equilibrium is where the population ends.

## Flashcards

> [!card]- List the four game models from simplest to most general.
> One-shot normal-form, repeated normal-form, stochastic game, partially observable stochastic game.

> [!card]- What are the three components of a normal-form game?
> Agents, an action set per agent, a reward function per agent over joint actions.

> [!card]- What are the components of a stochastic game?
> Agents, states, action sets, reward functions $R_i(s, a, s')$, transition function $T(s' \mid s, a)$, initial state distribution.

> [!card]- What does a POSG add to a stochastic game?
> For each agent, a set of observations and an observation function.

> [!card]- Why does a POSG policy condition on the observation history?
> The agent does not see the full state, only incomplete observations of it.

> [!card]- Two agents choose once, at the same time, and payoffs depend only on the action pair. Which model fits best?
> A normal-form game.

> [!card]- Self-driving cars cannot see behind trucks. Which game model fits?
> A partially observable stochastic game.

> [!card]- What does a policy condition on in a repeated normal-form game?
> The history of joint actions so far.

> [!card]- What distinguishes a common-reward game from a zero-sum game?
> Common-reward: everyone gets the same reward. Zero-sum: rewards in each cell sum to zero.

> [!card]- What is a best response?
> A policy that maximises an agent's expected return against fixed policies of the others.

> [!card]- What is a Nash equilibrium?
> A joint policy where no agent gains by changing its policy while the others keep theirs.

> [!card]- What did Nash prove?
> Every finite normal-form game has at least one equilibrium, possibly mixed.

> [!card]- Why must every action in a mixed equilibrium earn the same?
> If one earned more, the agent would drop the others.

> [!card]- What are the Nash equilibria of the Stag Hunt (4,4; 0,3; 3,0; 2,2)?
> (C, C), (D, D), and a mixed one with C at probability 2/3.

> [!card]- What does non-uniqueness of the Nash equilibrium mean?
> A game can have several equilibria and the concept does not say which is played.

> [!card]- What does incompleteness of the Nash equilibrium mean?
> It says nothing about off-equilibrium play or how agents reach an equilibrium.

> [!card]- When is a joint policy Pareto optimal?
> No other joint policy helps one agent without hurting another.

> [!card]- Why is (C, D) in the Prisoner's Dilemma Pareto optimal?
> Any change lowers the defector's payoff T, the highest in the game.

> [!card]- How do welfare and fairness differ?
> Welfare is the sum of returns; fairness is the product, which favours equal splits.

> [!card]- When is (C, C) the social optimum of a symmetric Prisoner's Dilemma?
> When $2R \ge T + S$.

> [!card]- Why is the Stag Hunt a dilemma of coordination and the Prisoner's Dilemma one of cooperation?
> Stag Hunt: the good outcome is an equilibrium, but so is a worse one. PD: the only equilibrium is inefficient.

> [!card]- Why can fitness not be measured in isolation in evolutionary game theory?
> It depends on how frequent the other types are.

> [!card]- What does the replicator equation assume?
> A strategy becomes more common when it earns more than the population average.

> [!card]- State the two-strategy replicator equation.
> $\dot x = x(1 - x)(f_C - f_D)$.

> [!card]- Give the three readings of the replicator equation.
> Natural selection (biology), imitation (economics), reinforcement learning dynamics (computer science).

> [!card]- Why does a Stag Hunt population (4,4; 0,3; 3,0; 2,2) starting at 50% C lose its cooperators?
> At 50%, C earns 2 and D earns 2.5, and C keeps earning less below 2/3.

> [!card]- Why does no adapting Stag Hunt population end at the mixed equilibrium?
> It is an unstable rest point: any push tips the population to all-C or all-D.

> [!card]- Why is all-C an unstable rest point in the Prisoner's Dilemma?
> A single defector earns T > R and spreads.

> [!card]- Why does a Hawk-Dove population settle in a stable mix?
> Each strategy earns more when it is rare.

> [!card]- True or false: some finite games have no Nash equilibrium, even with mixed strategies.
> **False.** Nash proved every finite game has one once mixed strategies are allowed.

> [!card]- True or false: the replicator equation assumes agents compute a best response to the population.
> **False.** It only assumes that strategies earning more than average become more common.

> [!card]- True or false: a stochastic game includes an observation function for each agent.
> **False.** Observation functions are what a POSG adds.

> [!card]- Multiple selection: which are limitations of the Nash equilibrium? (A) It applies only to two-player games. (B) An equilibrium can be worse for all players than another outcome. (C) Computing one can be hard in general games. (D) There can be several, and it does not say which is played.
> **B, C and D.** A: it applies to any number of players.

> [!card]- Multiple selection: which are part of a stochastic game? (A) A set of states. (B) A joint policy. (C) A transition function over joint actions. (D) An observation function per agent. (E) A reward function per agent.
> **A, C and E.** B: policies are chosen by agents. D: that is a POSG.

> [!card]- Multiple selection: which replicator expressions with two strategies are correct? (A) $x(1-x)(f_C - f_D)$ (B) $x(1-x)(f_C + f_D)$ (C) $x(f_C - \bar f)$ (D) $(1-x)(f_C - f_D)$ (E) $x(f_C - f_D)$
> **A and C.** B adds the fitnesses; D drops the factor $x$; E drops the factor $1 - x$.

## Links

- **Course:** [[AGAI - Overview|Course overview]]
- **Previous lecture:** [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
- **Lab:** [[AGAI-Lab01 - Simple Agents, Complex Dynamics]] (best responses, greed and fear, belief thresholds, repeated games, replicator dynamics of the Prisoner's Dilemma and Stag Hunt)
- **Background reading:** Albrecht, Christianos and Schäfer, *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches* (MIT Press, 2024), sections 3.1-3.4, 4.1, 4.2, 4.4, 4.7-4.9
- **Source:** Agentic Artificial Intelligence, Lecture 2, "Game Theory and Population Dynamics", Fernando P. Santos, MSc AI, UvA, 29 October 2026; mock midterm 2026 with answer key
