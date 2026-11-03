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
> How do we describe interacting agents formally, predict what they will do, and judge whether the result is any good? Games model the interaction, from a one-shot payoff matrix up to partially observable stochastic games with states and private observations. The Nash equilibrium, a mutual best response, predicts play; Pareto optimality, welfare and fairness say which outcomes are desirable. The two often disagree, which is what makes the Prisoner's Dilemma and the Stag Hunt social dilemmas.
>
> Equilibria are static: they do not say how agents get there. Evolutionary game theory adds the dynamics. In a population where better-earning strategies spread (the replicator equation), the Prisoner's Dilemma ends in full defection, the Stag Hunt tips to one of two pure states, and the Hawk-Dove game settles in a stable mix.
>
> The exam format is not announced yet; expect the game models and solution concepts, and computations: the Nash equilibria and optimal outcomes of a payoff matrix, and the rest points and stability of a replicator equation.

## Simple rules, surprising group behaviour

The course starts with **simple reflex agents** (agents that pick an action from the current percept alone, with no memory or planning; see [[Agent Program]]). Even these teach three lessons that carry over to LLM agents: **emergence** (group behaviour that no single agent was programmed to produce), **communication**, and **dilemmas**.

Two classic examples of emergence:

- **Boids** (Reynolds, 1987: "bird-oid" agents) produce a realistic flock from three local rules. **Separation:** move away from nearby agents to avoid crowding or collisions. **Alignment:** steer towards the average direction of nearby agents. **Cohesion:** move towards the average position of nearby agents to stay with the group. No rule mentions a flock.
- **Conway's Game of Life** is a grid of cells, each alive or dead, updated in steps. A live cell with fewer than 2 live neighbours dies (underpopulation); with 2 or 3 it survives; with more than 3 it dies (overpopulation). A dead cell with exactly 3 live neighbours becomes alive (birth); otherwise it stays dead. From these rules come stable shapes, shapes that cycle, and shapes that travel across the grid (the glider).

Game theory, the mathematics of interactions where each agent's outcome depends on what the others do, gives the tools to analyse such dynamics.

## Games as models of multi-agent interaction

A game formally defines a multi-agent interaction: who the agents are, what they can do, and what each gets. The course uses a hierarchy of four models, each containing the previous one as a special case:

1. **One-shot normal-form game:** $n$ agents act once, at the same time.
2. **Repeated normal-form game:** the same normal-form game played over many time steps; still one state.
3. **Stochastic game:** $n$ agents, $m$ states, fully observable.
4. **Partially observable stochastic game (POSG):** $n$ agents, $m$ states, each agent sees only part of what happens.

### Normal-form games: actions and rewards only

A [[Normal-Form Game|normal-form game]] has no time and no state, only actions and rewards.

> [!definition] Normal-form game
> A 3-tuple $(I, \{A_i\}_{i \in I}, \{R_i\}_{i \in I})$:
> - $I$: a finite set of agents, $I = \{1, \dots, n\}$
> - $A_i$: the finite set of actions of agent $i$
> - $R_i : A \to \mathbb{R}$: the reward function of agent $i$, where $A = A_1 \times \dots \times A_n$ is the set of **joint actions** (one action per agent)

Each reward depends on the joint action, so on what everyone else did too. With two agents the game is a matrix: rows are agent 1's actions, columns agent 2's, and each cell gives (row agent's reward, column agent's reward). The two games from Lecture 1:

**[[Prisoner's Dilemma]]** (C = cooperate, D = defect):

| | C | D |
|---|---|---|
| **C** | 3, 3 | 0, 5 |
| **D** | 5, 0 | 1, 1 |

What is special: D pays more whatever the other agent does (5 > 3 against C, 1 > 0 against D), yet both prefer (C, C) to the (D, D) they end up in.

**[[Stag Hunt]]**, for example two coding agents that can coordinate changes to a shared repository (C) or each do a small, safe task alone (D):

| | C | D |
|---|---|---|
| **C** | 4, 4 | 0, 3 |
| **D** | 3, 0 | 2, 2 |

What is special: the best action copies the other agent's. (C, C) is best for both, but cooperating alone earns 0, so C is risky.

Three classes of games, by how the rewards relate:

- **Zero-sum:** the rewards in every cell sum to zero, so one agent's gain is the other's loss. Matching Pennies: both show a coin; the row agent wins (1, $-1$) if the coins match, the column agent wins ($-1$, 1) if they differ.
- **Common-reward:** all agents get the same reward in every cell, so their interests coincide. Example: (A, A) gives 4, 4; (A, B) gives 0, 0; (B, A) gives 3, 3; (B, B) gives 2, 2.
- **General-sum:** any other relation, such as the Prisoner's Dilemma and the Stag Hunt, where interests partly align and partly conflict.

**How agents choose.** Each agent follows a [[Policy|policy]] $\pi_i$, a probability distribution over its actions, and samples action $a_i \in A_i$ with probability $\pi_i(a_i)$. A policy that always picks the same action is **pure**; one that randomises is **mixed**.

### Repeating the game: policies with memory

Interactions are often sequential. In a [[Repeated Game|repeated normal-form game]] the same game is played at time steps $t = 0, 1, 2, \dots$ and a policy can condition on the history:

$$\pi_i(a_i^t \mid h^t), \qquad h^t = (a^0, a^1, \dots, a^{t-1})$$

where $a_i^t$ is agent $i$'s action at step $t$ and $h^t$ is the sequence of joint actions played so far. Two strategies from Lab 1:

- **Tit-for-Tat (TFT):** start by cooperating; then repeat the opponent's last action.
- **Win-Stay Lose-Shift (WSLS):** start by cooperating; keep your action if it earned one of the top payoffs (3 or 5 in the Prisoner's Dilemma), switch otherwise.

The **number of steps is crucial**. In the Prisoner's Dilemma, always-defect gains once by exploiting TFT in round 1, which counts for a lot in a short game; in a long one, reciprocators earning 3 per round with each other pull ahead. [[AGAI-Lab01 - Simple Agents, Complex Dynamics|Lab 1]] finds that against TFT and WSLS, always-defect wins for 1 to 3 rounds and TFT from 4.

### Adding states: stochastic games

A [[Stochastic Game|stochastic game]] adds **states**: the environment changes in response to the joint action, and each state can have its own game.

> [!definition] Stochastic game
> A 6-tuple $(I, S, \{A_i\}_{i \in I}, \{R_i\}_{i \in I}, T, \mu)$:
> - $I$: a finite set of agents
> - $S$: a finite set of states
> - $A_i$: the finite set of actions of agent $i$
> - $R_i$: the reward function of agent $i$
> - $T$: the transition function
> - $\mu$: the initial state distribution

Outside the course material, the usual signatures are $R_i : S \times A \times S \to \mathbb{R}$ (reward for moving from one state to the next under a joint action), $T : S \times A \times S \to [0, 1]$ (the probability $T(s, a, s')$ of reaching $s'$ from $s$ under joint action $a$) and $\mu : S \to [0, 1]$ (the probability of starting in each state).

> [!example] A three-state game
> Two agents, each with actions 1 and 2, and three states $s_1, s_2, s_3$, each with its own payoff matrix:
>
> | | $s_1$ | $s_2$ | $s_3$ |
> |---|---|---|---|
> | (1, 1) | 1, 1 | 1, 3 | 1, 1 |
> | (1, 2) | 0, 1 | 0, 4 | 0, 0 |
> | (2, 1) | 2, 2 | 4, 0 | 4, 1 |
> | (2, 2) | 1, 1 | 3, 1 | 3, 2 |
>
> Some transitions: in $s_1$, joint action (2, 2) pays (1, 1) and leads back to $s_1$ with probability 0.5 or to $s_2$ with 0.5. In $s_2$, (1, 2) pays (0, 4) and leads to $s_1$ with probability 0.8 or $s_3$ with 0.2. In $s_3$, (2, 1) pays (4, 1) and returns to $s_1$ for sure. So this game has 2 agents, 2 actions each and 3 states. An action now matters twice: for today's reward and for which game is played tomorrow.

Two more examples. The two-square vacuum world of Lecture 1 is a stochastic game with **one** agent: 3 actions (Left, Right, Suck) and 8 states (2 robot positions times dirty or clean for each square; counts worked out here, since the course material leaves them open). A grid world where two agents walk around collecting apples (level-based foraging) has 2 agents and a state for every arrangement of agents and remaining apples. Outside the course material: a stochastic game with one agent is a [[Markov Decision Process]], and a repeated normal-form game is a stochastic game with one state.

### Hiding the state: partially observable stochastic games

A **POSG** models decision processes with incomplete information: agents receive **observations** that give incomplete information about the state and about other agents' actions. It applies wherever sensing is limited: **autonomous driving** (a car cannot see behind a truck), or **strategic games with private, hidden information** such as card games.

> [!definition] Partially observable stochastic game (POSG)
> An 8-tuple $(I, S, \{A_i\}, \{R_i\}, T, \mu, \{O_i\}, \{\mathcal{O}_i\})$: a stochastic game plus, for each agent $i$,
> - $O_i$: a finite set of observations
> - $\mathcal{O}_i : A \times S \times O_i \to [0, 1]$: the observation function, giving the probability that agent $i$ observes $o_i$ when joint action $a$ led to state $s$

The agent no longer sees the state, so its policy conditions on its own **observation history**:

$$\pi_i(a_i^t \mid h_i^t), \qquad h_i^t = (o_i^0, o_i^1, \dots, o_i^{t-1})$$

In the foraging grid, each agent sees only a small square of cells around itself. Outside the course material: a POSG with one agent is a [[POMDP]], and a stochastic game is a POSG in which every agent's observation reveals the full state.

## Solution concepts: what will agents do, and what is good?

The models describe interactions without saying what happens. **Solution concepts** answer four questions: Which actions will agents play? Which combinations of actions are desirable? What does it mean for an agent to act optimally? How desirable is the **joint policy** $\pi = (\pi_1, \pi_2, \dots, \pi_n)$ of all agents?

All the definitions use $U_i(\pi)$, the **expected return** of agent $i$ under joint policy $\pi$. In a normal-form game, with agents choosing independently, it is the reward averaged over joint actions (the formula is standard notation from outside the course material):

$$U_i(\pi) = \sum_{a \in A} \Big( \prod_{j \in I} \pi_j(a_j) \Big) R_i(a)$$

where $\prod_j \pi_j(a_j)$ is the probability of joint action $a = (a_1, \dots, a_n)$.

### Best response

"Optimal" only makes sense relative to what the others do. In the Stag Hunt, C is the best action against a partner who plays C, and D against a partner who plays D. So optimality is defined against fixed policies of the others.

> [!definition] Best response
> A **best-response policy** $\pi_i$ maximises the expected return for agent $i$ against a given set of policies for all other agents, $\pi_{-i} = (\pi_1, \dots, \pi_{i-1}, \pi_{i+1}, \dots, \pi_n)$:
> $$BR_i(\pi_{-i}) = \arg\max_{\pi_i} U_i(\langle \pi_i, \pi_{-i} \rangle)$$
> where $\langle \pi_i, \pi_{-i} \rangle$ is the complete joint policy.

$BR_i$ is a set, since several policies can tie. See [[Best Response]].

### Nash equilibrium: everyone best-responds at once

> [!definition] Nash equilibrium (NE)
> The Nash equilibrium applies the idea of a **mutual best response** to general-sum games with two or more agents. A joint policy $\pi$ is a Nash equilibrium if no agent $i$ can improve its expected return by changing its policy, assuming the other agents' policies remain fixed:
> $$\forall i, \; \forall \pi_i' : \quad U_i(\langle \pi_i', \pi_{-i} \rangle) \le U_i(\pi)$$
> In a NE, each agent's policy is a best response to all other agents' policies.

John Nash (1950) proved that **every** finite general-sum normal-form game has at least one such equilibrium once mixed policies are allowed, using a fixed-point theorem: some joint strategy is a best response to itself.

**Finding pure equilibria in a matrix.** For each column, mark the row agent's best reward; for each row, mark the column agent's best reward. A cell marked twice is a pure NE.

**Finding a mixed equilibrium.** An agent only mixes between actions that earn the same against the other's policy; if one earned more, it would drop the other. So in a symmetric two-action game, find the probability $q$ of the partner playing C at which C and D earn the same.

> [!example] Prisoner's Dilemma, Stag Hunt, Rock-Paper-Scissors
> **Prisoner's Dilemma** with (C, C) = 4, 4; (C, D) = $-2$, 6; (D, C) = 6, $-2$; (D, D) = 0, 0. Against C the row agent prefers D (6 > 4), against D it prefers D (0 > $-2$); the column agent likewise. Only (D, D) is marked twice: the unique NE.
>
> **Stag Hunt** (4, 4; 0, 3; 3, 0; 2, 2). Against C the best response is C (4 > 3), against D it is D (2 > 0). (C, C) and (D, D) are both NE, plus a mixed one: against a partner playing C with probability $q$, C earns $4q$ and D earns $3q + 2(1 - q) = 2 + q$, equal at $q = 2/3$, so each plays C with probability 2/3.
>
> **Rock-Paper-Scissors**, zero-sum: win 1, lose $-1$, tie 0.
>
> | | R | P | S |
> |---|---|---|---|
> | **R** | 0, 0 | $-1$, 1 | 1, $-1$ |
> | **P** | 1, $-1$ | 0, 0 | $-1$, 1 |
> | **S** | $-1$, 1 | 1, $-1$ | 0, 0 |
>
> No cell is marked twice: whatever pure pair is played, someone could switch to the action that beats the other's. The equilibrium is mixed: each agent plays each action with probability 1/3. Against that policy every action earns $\frac13(0) + \frac13(-1) + \frac13(1) = 0$, so no change of policy can earn more than 0, and neither agent can gain by deviating.

### What a Nash equilibrium does not tell you

- **Sub-optimality.** Nash equilibria do not always maximise expected returns. In the Prisoner's Dilemma (D, D) is the NE, but (C, C) gives both agents more.
- **Non-uniqueness.** There can be multiple, even infinitely many, equilibria, each with different expected returns. The Stag Hunt has three; the concept does not say which one agents will play.
- **Incompleteness.** An equilibrium says where no one wants to move. In sequential games it does not specify what agents do off the equilibrium path (after someone deviates), and it does not say how agents reach the equilibrium in the first place.

The first two lead to refinements that pick out desirable outcomes (next section); the third leads to the population dynamics of evolutionary game theory.

### Desirable outcomes: Pareto optimality, welfare and fairness

To narrow down the solutions, we can add requirements that describe what is **desirable** for the group.

> [!definition] Pareto optimality
> A joint policy $\pi$ is **Pareto-dominated** by another joint policy $\pi'$ if
> $$\forall i : U_i(\pi') \ge U_i(\pi) \quad \text{and} \quad \exists i : U_i(\pi') > U_i(\pi)$$
> A joint policy $\pi$ is **Pareto-optimal** if it is not Pareto-dominated by any other joint policy. Intuition: no other joint policy improves the expected return for at least one agent without reducing it for any other agent.

In the Stag Hunt, (C, C) = (4, 4) dominates every other cell, so it is the only Pareto-optimal outcome. In the Prisoner's Dilemma, (C, C), (C, D) and (D, C) are all Pareto optimal: any move away from (D, C) hurts the defector. Pareto optimality rules out waste and says nothing about fairness. See [[Pareto Optimality]]. Welfare and fairness constrain the space further:

> [!formula] Welfare and fairness
> $$W(\pi) = \sum_{i \in I} U_i(\pi), \qquad F(\pi) = \prod_{i \in I} U_i(\pi)$$
> A joint policy is **welfare-optimal** if $\pi \in \arg\max_{\pi'} W(\pi')$ and **fairness-optimal** if $\pi \in \arg\max_{\pi'} F(\pi')$.

The sum ignores how the total is split; the product rewards equal splits. Two outcomes with payoffs (5, 5) and (9, 1) have the same welfare, 10, but fairness 25 against 9. Outside the course material: a welfare-optimal outcome is always Pareto optimal (a Pareto improvement would raise the sum), and the product only works as a fairness measure for positive payoffs.

### Social dilemmas

Side by side, the two kinds of solution concept show why the games of Lecture 1 are **[[Social Dilemma|social dilemmas]]**: situations where agents who each pursue their own interest reach an outcome that is worse for the group than one they could have reached together.

- **Prisoner's Dilemma, a dilemma of cooperation:** the only NE, (D, D), is the only outcome that is not Pareto optimal.
- **Stag Hunt, a dilemma of coordination or trust:** the welfare-optimal (C, C) is an equilibrium, but so is the worse (D, D), and nothing in the NE concept favours the good one.

## From rational players to adapting populations

The solution concepts so far assume rational agents who compute best responses, and leave open how agents would ever reach an equilibrium. John von Neumann and Oskar Morgenstern said it of their own theory in *Theory of Games and Economic Behavior*:

> "We repeat most emphatically that our theory is thoroughly static. A dynamic theory would unquestionably be more complete and therefore preferable."

They added that building a dynamic theory is futile as long as the static side is not thoroughly understood.

Real agents are often **not so rational**: they resort to **trial and error**, have **incomplete information**, and **imitate** others. So the question becomes how agents in a population change their behaviour over time, depending on what others are doing. Concretely: in a large Stag Hunt population, half C and half D, meeting at random, does the number of C agents increase, decrease or stay the same? Both pure outcomes are Nash equilibria, so the static theory cannot answer.

### Evolutionary game theory

**Evolutionary game theory** combines evolutionary biology with game theory. In **natural selection** the genotype of fit individuals spreads in a population. But **fitness cannot be measured in isolation: it depends on the frequency of other types**, just as in game theory a payoff depends on the strategies of others. A cooperator does well among cooperators and badly among defectors. In the 1970s the biologists John Maynard Smith and George R. Price proposed applying game theory to understand evolution. The agents need no rationality: what earns more becomes more common.

### The replicator equation

Take a large population where a fraction $x$ uses strategy C and $1 - x$ uses D, and agents are paired at random. The [[Replicator Dynamics|replicator equation]] (Taylor and Jonker, 1978) describes how $x$ evolves:

> [!formula] Replicator equation
> $$\frac{dx}{dt} = \dot{x} = x \big( f_C(x) - \bar{f}(x) \big)$$
> - $x$: the fraction of individuals using strategy C
> - $f_C(x)$: the average payoff (fitness) of C when a fraction $x$ adopts C
> - $\bar{f}(x) = x f_C(x) + (1 - x) f_D(x)$: the average fitness in the population, with $f_D(x)$ the fitness of D

**C replicates if the fitness of C is higher than the average.** The factor $x$ makes growth proportional to how common C already is: a strategy nobody uses cannot spread.

The same equation describes the dynamics of strategy evolution in three fields:

- **Biology:** replication and natural selection; fitter types have more offspring.
- **Economics:** social learning and imitation; agents copy those who do better.
- **Computer science:** reinforcement learning dynamics. The link is Börgers and Sarin, "Learning through reinforcement and replicator dynamics" (their result, from outside the course material: a simple reinforcement learner follows the replicator equation on average).

**The two-strategy form.** Substitute $\bar f$:

$$f_C - \bar f = f_C - x f_C - (1 - x) f_D = (1 - x)(f_C - f_D)$$

so

> [!formula] Replicator equation, two strategies
> $$\dot{x} = x(1 - x)\big(f_C(x) - f_D(x)\big)$$

$x(1-x)$ is zero when either strategy is absent and positive in between, so the **sign of $f_C - f_D$ alone** decides whether C grows. With random pairing a partner plays C with probability $x$, so for payoffs $R$ (C meets C), $S$ (C meets D), $T$ (D meets C), $P$ (D meets D):

$$f_C(x) = xR + (1 - x)S, \qquad f_D(x) = xT + (1 - x)P$$

### Reading the dynamics: rest points and stability

A **rest point** is a state where $\dot x = 0$: $x = 0$, $x = 1$, and any $x$ where $f_C = f_D$. A rest point is **stable** if the population returns to it after a small push in either direction, and **unstable** if a push sends it away. You read this off the sign of $\dot x$ on either side: if $\dot x > 0$ just below and $\dot x < 0$ just above, the population is pushed back from both sides.

> [!example] The Stag Hunt population
> With $R = 4$, $S = 0$, $T = 3$, $P = 2$:
> $$f_C(x) = 4x, \qquad f_D(x) = 3x + 2(1 - x) = 2 + x$$
> $$\dot x = x(1 - x)(4x - 2 - x) = x(1 - x)(3x - 2)$$
> Rest points: $x = 0$, $x = 2/3$, $x = 1$. For $0 < x < 2/3$, $3x - 2 < 0$ so $\dot x < 0$; for $2/3 < x < 1$, $\dot x > 0$.
>
> ```
>  x = 0 ●  <<<<<<<<<<<<<<<<<<  ○  >>>>>>>  ● x = 1
>        stable    x falls    2/3  x rises   stable
>                           unstable
> ```
>
> $x = 0$ and $x = 1$ are stable; $x = 2/3$ is unstable, a tipping point. **Answer to the 50/50 question:** at $x = 1/2$, $\dot x = \frac12 \cdot \frac12 \cdot (-\frac12) = -\frac18 < 0$, so the number of C agents **decreases**, all the way to zero. A C agent earns $4 \cdot \frac12 = 2$ on average, a D agent $2.5$: cooperation pays only once more than two thirds of the population cooperates.

The interior rest point $x = 2/3$ is the Stag Hunt's mixed Nash equilibrium, but no adapting population ends there: the slightest push tips it. The two coincide because, with random pairing, the share $x$ plays the role of a rational agent's belief that its partner cooperates ([[AGAI-Lab01 - Simple Agents, Complex Dynamics|Lab 1]] works this out).

**A quick stability check** (derived in Lab 1): write $\dot x = g(x)$. Then $g'(0) = S - P$ and $g'(1) = T - R$, and a rest point is stable when $g' < 0$ there. So all-D is stable when $P > S$ (a lone cooperator among defectors earns less than they do), and all-C is stable when $R > T$ (a lone defector among cooperators earns less than they do).

### The Prisoner's Dilemma in a population

Write the Prisoner's Dilemma as a **donation game**: cooperating means paying a cost $c$ to give the partner a benefit $b$, with $b > c > 0$ (pulling someone out of the water costs you effort and saves them).

| | C | D |
|---|---|---|
| **C** | $b - c$, $b - c$ | $-c$, $b$ |
| **D** | $b$, $-c$ | 0, 0 |

This is a Prisoner's Dilemma since $T = b > R = b - c > P = 0 > S = -c$. The **Nash equilibrium is (D, D)**: D earns $c$ more than C whatever the partner does. The **social optimum** (the outcome with the largest sum of payoffs) is **(C, C)**, with total $2(b - c) > 0$.

What can we expect from evolution?

$$f_C(x) = x(b - c) + (1 - x)(-c) = xb - c, \qquad f_D(x) = xb$$

$$\dot x = x(1 - x)(-c)$$

Cooperating always costs exactly $c$ relative to defecting, whoever you meet, so $\dot x < 0$ for every $0 < x < 1$. The population ends **fully composed of D**, from any start. $x = 1$ is a rest point, but unstable: one defector among cooperators earns $b$ against their $b - c$ and spreads. Evolution here reproduces the Nash prediction and its sub-optimality.

### Hawk-Dove: coexistence

The **Hawk-Dove** or **Chicken** game, with C = conventional and D = dangerous (outside the course material, the story: two drivers race towards each other; the conventional one swerves, the dangerous one keeps going):

| | C | D |
|---|---|---|
| **C** | 3, 3 | 2, 8 |
| **D** | 8, 2 | $-2$, $-2$ |

Here the best response is the opposite of the partner's action: against C, D earns more (8 > 3); against D, C earns more (2 > $-2$). The symmetric Nash equilibrium is **mixed**. If the partner plays C with probability $q$, C earns $3q + 2(1 - q) = 2 + q$ and D earns $8q - 2(1 - q) = 10q - 2$; these are equal at $q = 4/9$, so each agent plays C with probability 4/9. Outside the course material, but it follows from the best responses: the game also has two asymmetric pure equilibria, (C, D) and (D, C), where one agent yields and the other does not.

In a population:

$$f_C(x) = 2 + x, \qquad f_D(x) = 10x - 2, \qquad \dot x = x(1 - x)(4 - 9x)$$

Rest points: $x = 0$, $x = 4/9 \approx 0.44$, $x = 1$. Below $4/9$, $\dot x > 0$; above, $\dot x < 0$. So $x = 0$ and $x = 1$ are unstable ($g'(0) = S - P = 4 > 0$, $g'(1) = T - R = 5 > 0$) and **$x = 4/9$ is stable**: the population converges to a **coexistence of C and D**. Each strategy earns more when it is rare: among mostly conventional drivers the dangerous one gets 8, among mostly dangerous ones the conventional one gets 2 against their $-2$.

The interior rest point is again the mixed equilibrium, but here the population ends there. Whether adaptation reaches a mixed equilibrium depends on the game.

### Equilibria against population outcomes

| Game | Nash equilibria | Welfare optimum | Replicator outcome |
|---|---|---|---|
| Prisoner's Dilemma | (D, D) | (C, C) | all D, from any start |
| Stag Hunt | (C, C), (D, D), mixed at 2/3 | (C, C) | all C above 2/3, all D below |
| Hawk-Dove | mixed at 4/9 (plus (C, D), (D, C)) | (C, D) or (D, C), total 10 | stable mix, 4/9 C |

## What to be able to write

- The four game models, their tuples and what each adds, with examples; zero-sum, common-reward and general-sum games.
- Best response and Nash equilibrium in words and formulas, Nash's existence result, and the three limitations.
- Pure and mixed equilibria, Pareto optima, welfare and fairness optima of a given matrix.
- Why static solution concepts lead to evolutionary game theory; the replicator equation, its two-strategy form and its three readings.
- Rest points and stability for the Stag Hunt (and the 50/50 question), the donation game and Hawk-Dove.

Model answers are in the exam questions below.

## Exam questions

> [!exam]- Describe the hierarchy of game models from one-shot normal-form games to partially observable stochastic games. Give each model's formal definition, say what it adds to the previous one, and how agents' policies change.
> **Key points:** normal-form $(I, \{A_i\}, \{R_i\})$, no time or state, $R_i$ on joint actions, policies are distributions over actions; repeated: same game over time, policies condition on action history $h^t = (a^0, \dots, a^{t-1})$; stochastic game $(I, S, \{A_i\}, \{R_i\}, T, \mu)$ adds states, transitions, initial distribution; POSG adds observation sets $O_i$ and functions $\mathcal{O}_i : A \times S \times O_i \to [0,1]$, policies condition on observation histories; each model contains the previous one, with an example of each.
>
> A **normal-form game** is a tuple $(I, \{A_i\}, \{R_i\})$: a finite set of agents $I$, a finite action set $A_i$ for each agent, and reward functions $R_i : A \to \mathbb{R}$ on the joint actions $A = A_1 \times \dots \times A_n$. There is no time and no state, only actions and rewards. Each agent follows a policy $\pi_i$, a distribution over its actions. Examples: the Prisoner's Dilemma and Stag Hunt (general-sum), Matching Pennies (zero-sum), a common-reward coordination game.
>
> A **repeated normal-form game** plays the same game over time steps. Policies become conditioned on the history of joint actions, $\pi_i(a_i^t \mid h^t)$ with $h^t = (a^0, \dots, a^{t-1})$, which allows strategies such as Tit-for-Tat and Win-Stay Lose-Shift. The number of steps matters for which strategies do well.
>
> A **stochastic game** $(I, S, \{A_i\}, \{R_i\}, T, \mu)$ adds a finite set of states $S$, a transition function $T$ (the probability of the next state given the current state and joint action) and an initial state distribution $\mu$; rewards can depend on the state. Each state can carry its own game, and actions affect both the immediate reward and the future state. Examples: a three-state game with a matrix per state, the vacuum world (one agent), a foraging grid world with two agents.
>
> A **partially observable stochastic game** adds, for each agent, a finite observation set $O_i$ and an observation function $\mathcal{O}_i : A \times S \times O_i \to [0, 1]$, the probability of each observation given the joint action and resulting state. Agents get incomplete information about the state and others' actions, so policies condition on their own observation history $h_i^t = (o_i^0, \dots, o_i^{t-1})$. Examples: autonomous driving, card games with hidden hands, agents in a grid world that see only nearby cells. Each model is a special case of the next: a repeated normal-form game is a stochastic game with one state, and a stochastic game is a POSG where observations reveal the state.

> [!exam]- Define a best response and a Nash equilibrium formally. What did Nash prove, and what are the three limitations of the Nash equilibrium as a solution concept?
> **Key points:** $BR_i(\pi_{-i}) = \arg\max_{\pi_i} U_i(\langle \pi_i, \pi_{-i}\rangle)$; NE: $\forall i, \pi_i'$: $U_i(\langle\pi_i', \pi_{-i}\rangle) \le U_i(\pi)$, mutual best response; Nash 1950: every finite general-sum normal-form game has an equilibrium, possibly mixed; sub-optimality (PD) and non-uniqueness (SH, possibly infinitely many, different returns); incompleteness (off-equilibrium paths, how equilibria are reached).
>
> A **best response** of agent $i$ to the other agents' policies $\pi_{-i}$ is a policy that maximises $i$'s expected return given them: $BR_i(\pi_{-i}) = \arg\max_{\pi_i} U_i(\langle \pi_i, \pi_{-i} \rangle)$, where $\langle \pi_i, \pi_{-i} \rangle$ is the complete joint policy and $U_i$ the expected return. A **Nash equilibrium** is a joint policy $\pi$ in which every agent's policy is a best response to the others': for every agent $i$ and every alternative policy $\pi_i'$, $U_i(\langle \pi_i', \pi_{-i} \rangle) \le U_i(\pi)$. No agent can improve its expected return by changing its own policy while the others keep theirs.
>
> **Nash (1950)** proved that every finite general-sum normal-form game has at least one Nash equilibrium if mixed policies are allowed (by a fixed-point argument). Rock-Paper-Scissors has no pure equilibrium but a mixed one, each action with probability 1/3.
>
> **Limitations.** (1) **Sub-optimality:** equilibria need not maximise returns; in the Prisoner's Dilemma (D, D) is the equilibrium while (C, C) gives both more. (2) **Non-uniqueness:** there can be several, even infinitely many, equilibria with different returns, as in the Stag Hunt, and the concept does not say which is played. (3) **Incompleteness:** in sequential games equilibria do not specify actions off the equilibrium path, and the concept does not say how agents reach an equilibrium. The first two motivate refinements such as Pareto optimality, welfare and fairness; the third motivates dynamic approaches such as evolutionary game theory.

> [!exam]- In the game below (row player's payoff first), find all pure Nash equilibria, the Pareto-optimal outcomes, and the welfare-optimal and fairness-optimal outcomes. Rows U, D; columns L, R; (U, L) = 5, 1; (U, R) = 0, 0; (D, L) = 1, 1; (D, R) = 3, 3.
> **Key points:** best responses: row U against L, D against R, column L against U, R against D; pure NE (U, L) and (D, R); Pareto optimal (U, L) and (D, R); welfare 6, 0, 2, 6, so both NE are welfare-optimal; fairness 5, 0, 1, 9, so (D, R) is the only fairness-optimal outcome.
>
> **Nash.** Against L the row player prefers U (5 > 1); against R it prefers D (3 > 0). Against U the column player prefers L (1 > 0); against D it prefers R (3 > 1). Cells where both are best-responding: (U, L) and (D, R). (U, R) and (D, L) are not equilibria: at (U, R) the row player gains by switching to D, at (D, L) the column player gains by switching to R.
>
> **Pareto.** (D, R) = (3, 3) dominates (D, L) = (1, 1) and (U, R) = (0, 0). (U, L) = (5, 1) and (D, R) = (3, 3) do not dominate each other (the row player prefers the first, the column player the second), so both are Pareto optimal.
>
> **Welfare** $W = U_1 + U_2$: 6, 0, 2, 6. Both (U, L) and (D, R) are welfare-optimal; the sum cannot separate them. **Fairness** $F = U_1 \cdot U_2$: 5, 0, 1, 9. Only (D, R) is fairness-optimal, because the product prefers the equal split.

> [!exam]- Show that Rock-Paper-Scissors (win 1, lose $-1$, tie 0) has no pure Nash equilibrium, and that both players choosing each action with probability 1/3 is a Nash equilibrium. What kind of game is it?
> **Key points:** zero-sum (payoffs in each cell sum to 0); in every pure cell the loser, or either player at a tie, can switch to the beating action and gain; against uniform play each pure action earns $\frac13(0 - 1 + 1) = 0$, so every mixed policy is also worth 0; no profitable deviation, so uniform play is a NE; consistent with Nash's existence result.
>
> It is a **zero-sum** game: in every cell one player's gain is the other's loss. **No pure equilibrium:** at a tie, either player can switch to the action that beats the other's and get 1 instead of 0; at a win for one player, the loser can switch to the action that beats the winner's and get 1 instead of $-1$. So in every pure cell some player has a profitable deviation.
>
> **Mixed equilibrium.** Suppose the column player plays R, P, S with probability 1/3 each. The row player's expected payoff from R is $\frac13 \cdot 0 + \frac13 \cdot (-1) + \frac13 \cdot 1 = 0$, and by symmetry also 0 from P and from S. Any mixed policy averages these, so it also earns 0. No deviation earns more than the uniform policy's 0, so uniform play is a best response to uniform play, and by symmetry the same holds for the column player. Both best-responding means a Nash equilibrium, which illustrates Nash's theorem: equilibria always exist once mixed policies are allowed.

> [!exam]- Why does evolutionary game theory study populations of agents instead of rational players? Explain the idea of frequency-dependent fitness, state the replicator equation with its terms, and give its three interpretations.
> **Key points:** static solution concepts do not say how equilibria are reached (von Neumann and Morgenstern: theory "thoroughly static"), and real agents use trial and error, incomplete information, imitation; biology plus game theory, Maynard Smith and Price in the 1970s; fitness depends on the frequency of other types, like payoffs on others' strategies; $\dot x = x(f_C(x) - \bar f(x))$, C replicates when fitter than average; biology (natural selection), economics (imitation), computer science (RL dynamics, Börgers and Sarin).
>
> Solution concepts such as the Nash equilibrium are static: they say which joint policies are stable but are silent on how agents get there, and they assume rational agents computing best responses. Von Neumann and Morgenstern called their own theory "thoroughly static" and a dynamic theory more complete. Real agents often learn by trial and error, have incomplete information and imitate others, so we ask how the strategies in a population change over time depending on what others do.
>
> Evolutionary game theory combines evolutionary biology and game theory, following Maynard Smith and Price in the 1970s. In natural selection fit types spread, but fitness cannot be measured in isolation: it depends on how frequent the other types are, exactly as a payoff in game theory depends on the others' strategies.
>
> The replicator equation: $\dot x = x\big(f_C(x) - \bar f(x)\big)$, where $x$ is the fraction using strategy C, $f_C(x)$ the average payoff (fitness) of C when a fraction $x$ uses C, and $\bar f(x) = x f_C + (1-x) f_D$ the population's average fitness. C replicates when it is fitter than average, at a rate proportional to its current share. The same equation describes replication and natural selection in biology, social learning by imitation in economics, and reinforcement learning dynamics in computer science (Börgers and Sarin showed that simple reinforcement learning follows it on average).

> [!exam]- Derive the two-strategy replicator equation $\dot x = x(1-x)(f_C - f_D)$. Apply it to the Stag Hunt ($R=4, S=0, T=3, P=2$): find the rest points and their stability, and say what happens to a population that starts with 50% C.
> **Key points:** $\bar f = x f_C + (1-x) f_D$ gives $f_C - \bar f = (1-x)(f_C - f_D)$; $f_C = 4x$, $f_D = 3x + 2(1-x)$, so $\dot x = x(1-x)(3x-2)$; rest points 0 (stable), 2/3 (unstable), 1 (stable); at $x = 1/2$, $\dot x = -1/8$, C decreases to 0; the interior point is the mixed NE and a tipping point.
>
> **Derivation.** $\bar f = x f_C + (1-x) f_D$, so $f_C - \bar f = (1-x) f_C - (1-x) f_D = (1-x)(f_C - f_D)$, and multiplying by $x$ gives $\dot x = x(1-x)(f_C - f_D)$. Since $x(1-x) > 0$ for $0 < x < 1$, the sign of $f_C - f_D$ decides the direction.
>
> **Fitness.** With random pairing a partner plays C with probability $x$: $f_C = 4x + 0 \cdot (1-x) = 4x$ and $f_D = 3x + 2(1-x) = 2 + x$. So $f_C - f_D = 3x - 2$ and $\dot x = x(1-x)(3x - 2)$.
>
> **Rest points.** $x = 0$, $x = 1$, and $x = 2/3$ where $3x - 2 = 0$. For $x < 2/3$, $\dot x < 0$; for $x > 2/3$, $\dot x > 0$. So the population moves away from $2/3$ on both sides: $2/3$ is unstable, and $0$ and $1$ are stable (also $g'(0) = S - P = -2 < 0$, $g'(1) = T - R = -1 < 0$).
>
> **50% C.** $\dot x(1/2) = \frac12 \cdot \frac12 \cdot (-\frac12) = -\frac18 < 0$: C agents earn 2 on average against D agents' 2.5, so C decreases, and keeps decreasing to $x = 0$, all D. The interior rest point is the mixed Nash equilibrium, but it is a tipping point: populations above two thirds C go to all C, below to all D. Equilibrium selection in the Stag Hunt depends on where the population starts.

> [!exam]- Analyse the donation game ($b > c > 0$) and the Hawk-Dove game ((C,C) = 3,3; (C,D) = 2,8; (D,C) = 8,2; (D,D) = $-2$,$-2$) under the replicator dynamics. For each, give the Nash equilibrium, the long-run population, and compare with the Stag Hunt.
> **Key points:** donation: NE (D,D), social optimum (C,C); $f_C = xb - c$, $f_D = xb$, $\dot x = -c\,x(1-x) < 0$, all D from any start; Hawk-Dove: symmetric NE mixed with C at 4/9, $\dot x = x(1-x)(4 - 9x)$; 0 and 1 unstable, 4/9 stable, coexistence; mixed equilibrium unstable in the Stag Hunt, stable in Hawk-Dove.
>
> **Donation game.** Payoffs (C,C) = $b-c$ each, (C,D) = $-c$, $b$; (D,C) = $b$, $-c$; (D,D) = 0. D earns $c$ more than C against either action, so (D, D) is the unique Nash equilibrium, while (C, C) maximises the sum, $2(b - c) > 0$. In a population, $f_C = x(b-c) + (1-x)(-c) = xb - c$ and $f_D = xb$, so $\dot x = x(1-x)(-c) < 0$ for all interior $x$: cooperators decline from any start and the population ends fully D. $x = 1$ is an unstable rest point, since a single defector earns $b > b - c$.
>
> **Hawk-Dove.** Against C, D is better (8 > 3); against D, C is better (2 > $-2$). The symmetric equilibrium is mixed: with the partner playing C with probability $q$, C earns $2 + q$ and D earns $10q - 2$, equal at $q = 4/9$. (There are also asymmetric pure equilibria (C, D) and (D, C).) In a population, $f_C = 3x + 2(1-x) = 2 + x$ and $f_D = 8x - 2(1-x) = 10x - 2$, so $\dot x = x(1-x)(4 - 9x)$. Rest points $0$, $4/9$, $1$; $\dot x > 0$ below $4/9$ and $< 0$ above, so $4/9$ is stable and the pure states are unstable ($g'(0) = S - P = 4 > 0$, $g'(1) = T - R = 5 > 0$). The population converges to coexistence, about 44% C, because each strategy earns more when it is rare.
>
> **Comparison.** In the Stag Hunt the interior rest point (the mixed equilibrium at 2/3) is unstable and populations go to all C or all D depending on the start. In Hawk-Dove the mixed equilibrium is the stable outcome. In the donation game there is no interior rest point and the Nash prediction (all D) is reached from everywhere.

## Flashcards

> [!card]- Which three lessons for modern AI systems do simple reflex agents already teach?
> Emergence, communication, dilemmas.

> [!card]- What are the three rules of the Boids model?
> Separation, alignment, cohesion.

> [!card]- In Conway's Game of Life, when does a dead cell become alive?
> When it has exactly 3 live neighbours.

> [!card]- List the hierarchy of game models from simplest to most general.
> One-shot normal-form, repeated normal-form, stochastic game, partially observable stochastic game.

> [!card]- What is a normal-form game formally?
> A tuple $(I, \{A_i\}, \{R_i\})$: agents, action sets, reward functions on joint actions.

> [!card]- What does a normal-form game lack that a stochastic game has?
> Time and states (with transitions and an initial state distribution).

> [!card]- What distinguishes a common-reward game from a zero-sum game?
> Common-reward: all agents get the same reward. Zero-sum: rewards in each cell sum to zero.

> [!card]- In a repeated normal-form game, what does a policy condition on?
> The history of joint actions, $h^t = (a^0, \dots, a^{t-1})$.

> [!card]- What are the six components of a stochastic game?
> Agents, states, action sets, reward functions, transition function, initial state distribution.

> [!card]- What does a POSG add to a stochastic game?
> Observation sets $O_i$ and observation functions $\mathcal{O}_i$ for each agent.

> [!card]- What is the signature of a POSG observation function?
> $\mathcal{O}_i : A \times S \times O_i \to [0, 1]$.

> [!card]- How does an action in a stochastic game matter twice?
> It earns a reward now and decides which state's game is played next.

> [!card]- Why do POSG policies condition on observation histories instead of states?
> Agents only receive incomplete observations of the state and others' actions.

> [!card]- Give two application areas of POSGs.
> Autonomous driving; card games with private, hidden information.

> [!card]- What four questions do solution concepts answer?
> Which actions agents play, which are desirable, what acting optimally means, how desirable the joint policy is.

> [!card]- What is a best-response policy?
> A policy maximising an agent's expected return against fixed policies of all other agents.

> [!card]- State the Nash equilibrium condition formally.
> $\forall i, \pi_i'$: $U_i(\langle \pi_i', \pi_{-i} \rangle) \le U_i(\pi)$.

> [!card]- What is a joint policy in which every agent's policy is a best response to the others' called?
> A Nash equilibrium.

> [!card]- What did John Nash prove about finite normal-form games?
> Every finite general-sum normal-form game has at least one equilibrium, possibly mixed.

> [!card]- Why does Nash's existence result need mixed policies?
> Some games, such as Rock-Paper-Scissors, have no pure equilibrium.

> [!card]- What is the Nash equilibrium of Rock-Paper-Scissors?
> Each player plays each action with probability 1/3.

> [!card]- How do you find the pure Nash equilibria of a two-player payoff matrix?
> Mark each agent's best reward against each opponent action; cells marked twice are pure equilibria.

> [!card]- Why must every action in a mixed equilibrium earn the same?
> If one action earned more, the agent would drop the others and stop mixing.

> [!card]- What are the three limitations of the Nash equilibrium?
> Sub-optimality, non-uniqueness, incompleteness.

> [!card]- What does the incompleteness of the Nash equilibrium refer to?
> It specifies no actions off the equilibrium path and not how equilibria are reached.

> [!card]- When is a joint policy Pareto-dominated by another?
> The other gives every agent at least as much and some agent strictly more.

> [!card]- Which outcomes of the Stag Hunt (4,4; 0,3; 3,0; 2,2) are Pareto optimal?
> Only (C, C).

> [!card]- How do welfare optimality and fairness optimality differ?
> Welfare maximises the sum of returns; fairness maximises the product, favouring equal splits.

> [!card]- What makes a game a social dilemma?
> Self-interested play leads to an outcome worse for the group than one reachable together.

> [!card]- Why is the Prisoner's Dilemma a dilemma of cooperation and the Stag Hunt one of coordination?
> In the PD the only equilibrium is inefficient; in the SH the efficient outcome is an equilibrium but so is a worse one.

> [!card]- What did von Neumann and Morgenstern say about their own game theory?
> That it is thoroughly static, and a dynamic theory would be more complete.

> [!card]- In evolutionary game theory, what does fitness depend on?
> The frequency of other types in the population.

> [!card]- Which biologists proposed applying game theory to evolution?
> John Maynard Smith and George R. Price, in the 1970s.

> [!card]- Why does evolutionary game theory not need rational agents?
> Strategies that earn more simply become more common.

> [!card]- State the replicator equation for strategy C.
> $\dot x = x\big(f_C(x) - \bar f(x)\big)$.

> [!card]- When does strategy C grow under the replicator dynamics?
> When its fitness is higher than the population average.

> [!card]- Why can a strategy nobody uses never spread under the replicator equation?
> Growth is proportional to its current share $x$, which is zero.

> [!card]- Give the three interpretations of the replicator equation.
> Natural selection (biology), imitation (economics), reinforcement learning dynamics (computer science).

> [!card]- State the two-strategy replicator equation.
> $\dot x = x(1 - x)\big(f_C(x) - f_D(x)\big)$.

> [!card]- Why does the sign of $f_C - f_D$ alone decide the direction in the two-strategy replicator equation?
> $x(1 - x)$ is positive for every $x$ strictly between 0 and 1.

> [!card]- What is the replicator equation of the Stag Hunt with $R = 4$, $S = 0$, $T = 3$, $P = 2$?
> $\dot x = x(1 - x)(3x - 2)$.

> [!card]- What are the rest points of the Stag Hunt replicator dynamics, and which are stable?
> 0 and 1 stable, 2/3 unstable.

> [!card]- Why does a Stag Hunt population ($R = 4$, $S = 0$, $T = 3$, $P = 2$) starting at 50% C lose its cooperators?
> At $x = 1/2$, C earns 2 on average against D's 2.5; below 2/3, C always earns less.

> [!card]- How can you tell from the payoffs whether all-D is a stable rest point?
> It is stable when $P > S$: a lone cooperator earns less than the defectors.

> [!card]- Why do cooperators decline from any start in a donation-game population?
> Cooperating always earns exactly $c$ less than defecting, so $\dot x = x(1 - x)(-c) < 0$.

> [!card]- What is the long-run outcome of the Hawk-Dove game (3,3; 2,8; 8,2; $-2$,$-2$) under the replicator dynamics?
> Stable coexistence at 4/9 C.

> [!card]- Why do both pure states of Hawk-Dove fail to be stable?
> Each strategy earns more when it is rare, so it spreads back from near zero.

> [!card]- How does the mixed equilibrium of the Stag Hunt differ from that of Hawk-Dove under the replicator dynamics?
> Stag Hunt: unstable tipping point. Hawk-Dove: stable population outcome.

> [!card]- True or false: in the donation game, a population of all cooperators is a stable rest point.
> **False.** It is a rest point, but a single defector earns more and spreads.

## Links

- **Course:** [[AGAI - Overview|Course overview]]
- **Previous lecture:** [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]] (agent types, the dilemmas introduced)
- **Lab:** [[AGAI-Lab01 - Simple Agents, Complex Dynamics]] for worked analyses of best responses, belief thresholds, repeated-game tournaments, the replicator dynamics of the Prisoner's Dilemma and Stag Hunt, and the stability check
- **Background reading:** Albrecht, Christianos and Schäfer, *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches* (MIT Press, 2024), Chapter 3 (Games: models of multi-agent interaction) and Chapter 4 (Solution concepts for games)
- **Source:** Agentic Artificial Intelligence, Lecture 2, "Game Theory and Population Dynamics", Fernando P. Santos, MSc AI, UvA, 29 October 2026
- **Papers and books cited:** C. W. Reynolds, "Flocks, herds and schools: A distributed behavioral model", ACM SIGGRAPH Computer Graphics (1987). J. F. Nash, "Equilibrium points in $n$-person games", PNAS 36(1), 48-49 (1950). J. von Neumann and O. Morgenstern, *Theory of Games and Economic Behavior* (1944), section 4.8.2. P. D. Taylor and L. B. Jonker, "Evolutionary stable strategies and game dynamics", Mathematical Biosciences (1978). T. Börgers and R. Sarin, "Learning through reinforcement and replicator dynamics", Journal of Economic Theory (1997).
