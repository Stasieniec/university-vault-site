---
type: lecture
course: 5204AGAI6Y
week: 2
lecture: 3
date: 2026-11-03
book_sections: ["Albrecht, Christianos and Schäfer, MARL, Ch 5-9 (optional background)"]
topics:
  - "[[Reinforcement Learning]]"
  - "[[Markov Decision Process]]"
  - "[[Value Function]]"
  - "[[Bellman Equation]]"
  - "[[Value Iteration]]"
  - "[[Q-Learning]]"
  - "[[Discount Factor]]"
  - "[[Epsilon-Greedy Policy]]"
  - "[[Exploration vs Exploitation]]"
  - "[[Stochastic Game]]"
  - "[[Nash Equilibrium]]"
  - "[[Independent Learning]]"
  - "[[Multi-Agent Credit Assignment]]"
status: complete
---

# AGAI-L03: Intro to Multi-Agent Reinforcement Learning

> [!abstract] In short
> How does one agent learn from reward alone, and what breaks when several agents learn in the same environment? A single agent faces a Markov decision process: values obey the Bellman equations, which value iteration solves when the model is known and Q-learning approximates from samples when it is not. With several agents the problem is a stochastic game, where transitions and each agent's reward depend on the joint action, so "maximise your return" no longer defines a solution. Learning becomes non-stationary, credit must be split between agents, learners can settle in a bad equilibrium, and joint spaces grow exponentially with the number of agents. Independent Q-learning (each agent treats the others as environment) has no guarantees and miscoordinates in the climbing game, but it scales and is the standard baseline.
>
> No mock exam exists yet. Likely questions: why multi-agent RL is harder than single-agent RL, independent learners, and small computations (a Q-learning update, effective horizon, ε-greedy probabilities, classifying a payoff structure).

## Learning from your own actions

[[Reinforcement Learning]] is **sequential decision making**: your actions shape what you see next. Chess and robot control qualify; time-series forecasting and speech recognition do not, because a forecast does not change future demand.

Supervised learning is **open-loop**: it gets a dataset $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}$ of inputs with correct labels, assumed i.i.d. (independent and identically distributed), and learns $f_\theta(\mathbf{x}_i) \approx \mathbf{y}_i$. RL is **closed-loop**: the agent sees $s_t$, picks its own $a_t$, and gets back a reward and the next state. Its data $(s_1, a_1, r_1, \dots, s_T, a_T, r_T)$ is **not i.i.d.** (consecutive samples are correlated and depend on how it acts) and has **no labels, only reward**. The goal is a policy $\pi_\theta: s_t \to a_t$ maximising $\sum_t r_t$.

## The pieces of an RL problem

An **agent** interacts with an **environment** to achieve a **goal**:

- **Agent:** a [[Policy|policy]] $\pi_\theta(a \mid s)$, the probability of action $a$ in state $s$, with parameters $\theta$.
- **Environment:** an MDP $\mathcal{M} = \{\mathcal{S}, \mathcal{A}, \mathcal{T}, r\}$ ([[Markov Decision Process]]): states, actions, transitions $P(s_{t+1} \mid s_t, a_t)$ and rewards $r_t = R(s_t, a_t, s_{t+1})$.
- **Goal:** maximise return, $J(\theta) = \mathbb{E}_\pi[\sum_t r_t] \approx \frac{1}{N}\sum_{i=1}^{N}\sum_t r_t^i$, estimated by averaging $N$ sampled runs ($r_t^i$ is the reward at step $t$ of run $i$).

Algorithms loop: run the policy to generate samples, estimate the return (or fit a model), improve the policy.

The running example: **Spot**, a robot dog in a 4×4 grid, starts top-left; home (bottom-right) is worth +1, four holes −1; actions up, right, down, left.

```
 S  .  .  .        S = start
 .  X  .  X        X = hole (-1)
 .  .  .  X        H = home (+1)
 X  .  .  H
```

> [!definition] Markov property
> The **state is a complete summary of the past, as far as the future is concerned**: knowing earlier states and actions does not change $P(s_{t+1} \mid s_t, a_t)$.

If Spot has a battery the state leaves out, the same cell can lead to different futures depending on how long Spot has walked, so the cell is no longer Markov (partial observation is the topic of [[POMDP|POMDPs]]).

Solving an MDP means finding a policy that maximises the **expected discounted return**

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k=0}^{\infty}\gamma^k r_{t+k}, \qquad \gamma \in [0, 1)$$

where the [[Discount Factor]] $\gamma$ makes a reward $k$ steps ahead count $\gamma^k$ as much as one now. (The RL course writes this step's reward as $R_{t+1}$.) This is the single-agent **solution concept**, the criterion for what counts as a solution. An **[[Optimal Policy|optimal policy]] always exists**: best from every starting state, expressible as a deterministic map from state to action, possibly several of them, all with the same maximum return.

## Values and the Bellman equations

Spot has $4^{16} \approx 4.3 \times 10^9$ deterministic policies ($|\mathcal{A}|^{|\mathcal{S}|}$), too many to search. The Markov property helps: the future depends only on the current state, so "go right" next to home is optimal whatever happens elsewhere, and decisions can be built state by state from values.

> [!definition] State and action values ([[Value Function]])
> $$V^\pi(s) = \mathbb{E}_\pi\Big[\sum_{t=0}^{\infty}\gamma^t r_t \,\Big|\, s_0 = s\Big], \qquad Q^\pi(s,a) = \mathbb{E}_\pi\Big[\sum_{t=0}^{\infty}\gamma^t r_t \,\Big|\, s_0 = s, a_0 = a\Big]$$
> $V^\pi(s)$ is the expected discounted return from $s$ under $\pi$; $Q^\pi(s,a)$ the same after first taking $a$. Both are **specific to a policy**. They relate by $V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s,a)$ and, for the optimal policy, $V^*(s) = \max_a Q^*(s,a)$.

"A state is only as good as where it leads": values satisfy the self-consistency conditions called the [[Bellman Equation|Bellman equations]], and they are correct when these hold in every state.

> [!formula] Bellman equations for a policy $\pi$
> $$V^\pi(s) = \sum_a \pi(a \mid s)\Big[r(s,a) + \gamma\sum_{s'}P(s' \mid s,a)\,V^\pi(s')\Big]$$
> $$Q^\pi(s,a) = r(s,a) + \gamma\sum_{s'}P(s' \mid s,a)\sum_{a'}\pi(a' \mid s')\,Q^\pi(s',a')$$
> - $\pi(a \mid s)$: probability the policy picks $a$; $r(s,a)$: step reward; $\gamma$: discount
> - $P(s' \mid s,a)$: probability of landing in $s'$; $V^\pi(s')$, $Q^\pi(s',a')$: value of where you land

Turn $=$ into $\leftarrow$, apply to every state from any initial guess, and **repeat until the values stop changing**: that evaluates $\pi$. Now make the policy **greedy**, $\pi(s) = \arg\max_a[r(s,a) + \gamma\sum_{s'}P(s' \mid s,a)V(s')]$, and the average over actions becomes a max:

> [!formula] Value iteration
> $$V(s) \leftarrow \max_a\Big[r(s,a) + \gamma\sum_{s'}P(s' \mid s,a)\,V(s')\Big]$$

This **chases a moving target**, since the evaluated policy improves after every update; **when the values stop changing, the policy is optimal** ([[Value Iteration]], a [[Dynamic Programming]] method for the [[Bellman Optimality Equation]]).

## Learning without a model: Q-learning

Dynamic programming **needs $P(s' \mid s,a)$ and $R(s,a,s')$**. That model is often unavailable or intractable (Breakout has a bazillion states); usually there is only a simulator to **sample**. The RL problem: no prior information on $P$ and $R$, only experience $(s,a,r,s')$, learned one transition at a time. So **replace the model with samples**: the expectation $\sum_{s'}P(s' \mid s,a)[\dots]$ becomes one observed transition, and with a greedy target this gives [[Q-Learning]].

> [!formula] Q-learning update
> $$Q(s,a) \leftarrow Q(s,a) + \alpha\Big[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\Big]$$
> - $(s,a,r,s')$: one observed transition; $\alpha \in (0,1]$: learning rate
> - $r + \gamma\max_{a'}Q(s',a')$: the **target**, a one-sample estimate of $Q(s,a)$ assuming the best action next
> - the bracket is the [[TD Error|TD error]]; if $s'$ is terminal there is no future value and the target is just $r$

Dynamic programming makes a **full pass** (every state, action and outcome, weighted by probability); Q-learning uses a **single sample**, so its target is **noisy** and $\alpha$ averages the noise out over visits. The max makes it learn the greedy policy's values while acting exploratorily ([[Temporal Difference Learning]]). The full loop (choose ε-greedily, act, update, decay ε) is the independent Q-learning pseudocode further down, run with one agent.

> [!example] One update by hand
> $Q(s,\text{down}) = 0.5$; Spot moves down, gets $r = 0$, and $\max_{a'}Q(s',a') = 0.8$. With $\alpha = 0.1$, $\gamma = 0.9$: target $= 0.9 \times 0.8 = 0.72$, TD error $= 0.22$, new $Q = 0.5 + 0.1 \times 0.22 = 0.522$.

Sampling brings new challenges: **exploration** (you only learn about states you visit), **noise and credit assignment** (biased, high-variance updates; late rewards must be traced to their causes) and **data efficiency** (samples can be expensive). Value iteration is simple and fast but needs a model; Q-learning needs no prior knowledge but must explore and is slow.

## Discount, learning rate and exploration

**Discount factor.** $\gamma$ sets the present value of future rewards: eat the cupcake now, or invest for a bigger payoff in five years.

> [!formula] Effective horizon
> $$H_{\text{eff}} = \frac{1}{1-\gamma}$$
> roughly how many steps ahead the agent looks: $\gamma = 0.9$ gives 10, $\gamma = 0.99$ gives 100. The weights sum to $\sum_k \gamma^k = \frac{1}{1-\gamma}$, so a reward of 1 per step is worth $\frac{1}{1-\gamma}$ undiscounted steps.

It is usually set high (0.99), and it is needed mainly for **infinite-horizon tasks**, where it keeps the sum finite. It also shapes the solution: in an **apple orchard** whose regrowth depends on how many apples are left, a low $\gamma$ picks everything now and nothing regrows, while a high $\gamma$ leaves apples to keep the orchard producing. (Outside the course material: a low $\gamma$ suits tasks where only the near future matters, and short horizons are quicker to learn.)

**Learning rate.** $\alpha$ sets how much a new sample is trusted against current belief; the update is a **running average**:

$$Q(s,a) \leftarrow (1-\alpha)\,Q(s,a) + \alpha\Big[r + \gamma\max_{a'}Q(s',a')\Big]$$

In a **deterministic** environment a high $\alpha$ is fast and fine, a low one slow and wasteful. In a **stochastic** one (a slippery ice grid where moves sometimes slide) a high $\alpha$ is noisy and never settles, a low one smooth and accurate but slow. The fix is to **decay $\alpha$**.

**Exploration.** If you never try things, you never discover they are better. An agent that keeps refining its first path to a +2 goal never finds the +5 goal further away. Too little exploration sticks with the suboptimal, too much wastes samples ([[Exploration vs Exploitation]]), so exploration also follows a **decay schedule**.

> [!formula] ε-greedy and Boltzmann exploration
> $$\pi_{\varepsilon}(a \mid s) = \begin{cases}1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'}Q(s,a')\\[2mm] \dfrac{\varepsilon}{|\mathcal{A}|} & \text{otherwise}\end{cases} \qquad\qquad \pi_{\tau}(a \mid s) = \frac{\exp\big(Q(s,a)/\tau\big)}{\sum_{a'}\exp\big(Q(s,a')/\tau\big)}$$
> - $\varepsilon$: probability of a uniformly random action, which can also be the greedy one (hence its extra $\varepsilon/|\mathcal{A}|$)
> - $|\mathcal{A}|$: number of actions; $\tau$: temperature, low is nearly greedy, high nearly uniform

[[Epsilon-Greedy Policy|ε-greedy]] is simple and easy to decay, but its **exploration is blind**: a disastrous action is tried as often as a nearly-best one. Boltzmann ([[Softmax Policy|softmax]]) exploration is **value-aware** but **hard to tweak**, since a good $\tau$ depends on the scale of the Q-values.

> [!example] Exploration probabilities
> Four actions, $\varepsilon = 0.2$: greedy action $1 - 0.2 + 0.05 = 0.85$, each other $0.05$. Boltzmann with $Q = (2, 1)$: the better action gets $1/(1+e^{-1}) \approx 0.73$ at $\tau = 1$ and $1/(1+e^{-2}) \approx 0.88$ at $\tau = 0.5$.

Tuning is "part science, part wizardry", since good values depend on the environment, rewards and noise: **always run multiple seeds** (results vary a lot between them), look at reward curves, change one parameter at a time, watch the agent act.

## Many learners in one environment

In multi-agent RL (MARL), **agents interact in a shared environment to achieve their own goals**. Each agent $i$ has its own policy $\pi_i(a \mid s)$; transitions and rewards depend on the **joint action** (everyone's actions together); goals need not be aligned. Examples: chefs sharing a kitchen (Overcooked), hide and seek, rock-paper-scissors. It is a [[Multi-Agent System]] whose agents learn.

MARL is a **family of problems** that differ in:

- **who the agents are:** how many (2, 10, thousands); shared, opposed or mixed rewards;
- **who I will face:** my training opponents or new ones; control of all agents or one among many; shared learning rules or not;
- **what they can do together:** communicate freely, with limited bandwidth or not at all; observe everything or part of the world;
- **the structure of the game:** simultaneous or turn-based; one shot or repeated; symmetric or heterogeneous roles.

### Payoff structure decides what a good solution is

The relation between the agents' rewards shapes what a good solution means, what can go wrong and which algorithms make sense.

| | Fully cooperative | General-sum | Zero-sum |
|---|---|---|---|
| Rewards | $R_1 = R_2$ | anything in between | $R_1 = -R_2$ |
| In words | we all want the same thing | we each have our own preferences | your win is my loss |
| Examples | warehouse robots, Overcooked | traffic at an intersection | hide and seek, rock-paper-scissors |
| Learning must | coordinate how to split tasks | also balance individual and group welfare | stay robust to many opponents |

General-sum learners struggle with **Pareto inefficiency**, outcomes where everyone could be better off, like mutual defection in the [[Prisoner's Dilemma]] ([[Pareto Optimality]], [[Social Dilemma]]). In zero-sum games my changes prompt my opponent's counter-changes. To **classify a game**, check every cell: equal payoffs everywhere is fully cooperative, payoffs summing to zero everywhere is zero-sum, anything else general-sum. The Prisoner's Dilemma (3,3 / 0,5 / 5,0 / 1,1) is general-sum: the sums 6, 5, 5, 2 are not zero.

## Stochastic games: the MDP with many agents

> [!definition] Stochastic game
> The **multi-agent generalisation of an MDP**: $N$ agents act **simultaneously** and **each has its own reward**. A [[Stochastic Game]] is the tuple $(\mathcal{N}, \mathcal{S}, \{\mathcal{A}^i\}, P, \{R^i\}, \gamma)$:
> - $\mathcal{N}$: the $N$ agents; $\mathcal{S}$: the shared state space; $\mathcal{A}^i$: agent $i$'s actions
> - $P(s' \mid s, a^1, \dots, a^N)$: transitions, depending on the **joint action**
> - $R^i(s, a^1, \dots, a^N, s')$: agent $i$'s own reward function; $\gamma$: discount
>
> [[AGAI-L02 - Game Theory and Population Dynamics|Lecture 2]] writes the same object as $(I, S, \{A_i\}, \{R_i\}, T, \mu)$, with transitions $T$ and an initial-state distribution $\mu$; here the transitions are $P$ and the discount $\gamma$ is part of the tuple.

> [!example] The iterated Prisoner's Dilemma
> $N = 2$; the state is **last round's actions of both players**, $\mathcal{S} = \{CC, CD, DC, DD\}$; $\mathcal{A}^i = \{C, D\}$; the joint action fixes the next state (play C and D, next state CD); rewards come from the PD matrix (3 each for mutual cooperation, 5 and 0 when one defects on a cooperator, 1 each for mutual defection). A policy can react to the state: "cooperate if the other cooperated last time". A one-shot [[Normal-Form Game|normal-form game]] is the single-state case.

> [!formula] Agent $i$'s objective
> $$J^i(\pi^1, \dots, \pi^N) = \mathbb{E}\Big[\sum_t \gamma^t R_t^i \,\Big|\, \pi^1, \dots, \pi^N\Big]$$
> Agent $i$'s expected discounted return ($R_t^i$ is its reward at step $t$) **depends on everyone else's policy too**, but agent $i$ controls only its own $\pi^i(a^i \mid s)$.

In an MDP the solution concept was "maximise your return". That is **no longer expressive enough**: my best policy depends on yours and yours on mine. MARL optimises instead for **stability** (a [[Nash Equilibrium]]: no agent gains by changing only its own policy, each policy a [[Best Response]] to the others), **no regret** (outside the course material: in hindsight no single fixed action would have done better), **social welfare** (total reward) or **robustness to opponents**.

### What convergence means when everyone learns

In an MDP, learning has converged when the policy stops changing. In **matching pennies**, a penalty kick, each player picks heads (H) or tails (T); the goalkeeper (row) wins if they match, the kicker (column) if they differ.

| | H | T |
|---|---|---|
| **H** | 1, −1 | −1, 1 |
| **T** | −1, 1 | 1, −1 |

If the goalkeeper leans to H, the kicker should lean to T, so the goalkeeper should switch to T, and so on. **Any change I make gives my partner a reason for the opposite change**, so the learning dynamics **orbit** around the equilibrium (each plays H with probability 0.5) without settling. Yet on average the cycling policies play each action half the time and earn the equilibrium's reward (0 each). Whether that counts as converged depends on the notion: **exact policy** (no), **average policy** over many episodes (yes), **average reward** of the optimal policy (yes). Stochastic games challenge both MDP notions, optimality and convergence.

## Why multi-agent learning is hard

**Non-stationarity** and **credit assignment** limit **how well we can learn**, since both make the signal very noisy. **Equilibrium selection** decides **where we end up**: there are many equilibria, and the desirable ones are often harder to reach. **Scalability** makes all of them worse.

### Non-stationarity: experience has a shelf life

To agent $i$ the other agents are part of the environment, which therefore changes when they do:

> [!formula] The environment as agent $i$ experiences it
> $$P^i(s' \mid s, a^i) = \sum_{a^{-i}}\Big[\prod_{j \neq i}\pi^j(a^j \mid s)\Big]\,P(s' \mid s, a^i, a^{-i})$$
> - $a^{-i}$: the actions of all agents except $i$, summed over every combination
> - $\prod_{j \neq i}\pi^j(a^j \mid s)$: probability the others play that combination
> - $P(s' \mid s, a^i, a^{-i})$: the game's true, fixed transition function
>
> $P^i$ **changes whenever $\pi^{-i}$ (the others' policies) changes**, and so does agent $i$'s expected reward.

In single-agent RL, targets move only because my estimates improve; in MARL they also move because the **opponents' policies change**. Experience has a shelf life: targets we bootstrap from (build new estimates on) go stale. Consequences:

- **Cycling:** [[Best Response|best responses]] chase each other, as in matching pennies; both keep improving, neither converges.
- **Broken guarantees:** MDP convergence proofs, such as Q-learning's, assume a fixed environment.
- **Noisy targets:** bootstrapping from stale estimates gives a low signal-to-noise ratio.

In practice (robot football, say) it is a moving signal that is hard to track, the non-stationarity [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI|Lecture 1]] flagged.

### Credit assignment: whose action earned the reward

Single-agent RL has **temporal** credit assignment: which earlier action led to this reward? MARL adds a **new axis, agent credit**: whose action led to it? See [[Multi-Agent Credit Assignment]].

> [!example] The lazy agent in level-based foraging
> Agents on a grid collect apples (outside the course material: an apple can be loaded only if the agents loading it have enough combined level). Agents A, B and C share a **team reward** of +1 per apple. A and B do the work while C wanders in a corner, yet C also receives +1, so whatever it was doing is reinforced: **a lazy agent gets its behaviour reinforced because a teammate succeeded.**

Disentangling contributions needs **counterfactuals** (what if C had acted differently?), so many samples and brutal exploration: a new layer on an already hard causal inference problem, at very low signal-to-noise. It is **worst in fully cooperative games**, where there is no individual signal at all, but **also present in general-sum games**, where individual rewards still depend on the others.

> [!example] Fishing from a shared lake
> Agents compete for fish, but the lake can be depleted (a [[Social Dilemma]]). **Wrong lesson from a good outcome:** I fish aggressively while the others hold back and catch lots, so aggression looks good, though it paid only because they held back. **Wrong lesson from a bad outcome:** I fish sustainably, the others hold back too, I catch little, so sustainability looks bad, though that shared restraint is what keeps the lake stocked. (The course has the others holding back in both cases. The second is cleaner with the others fishing aggressively: they cause my small catch, and I blame my own restraint.)

### Equilibrium selection: settling in the wrong equilibrium

This one is **native to multi-agent settings and inescapable**: even perfect learners can converge to **suboptimal equilibria**. The cause is the game itself, which has several equilibria, so better learning does not remove it. Humans do it all the time: corrupt government persists where playing along pays as long as everyone else plays along.

- **Path dependence:** history and initial conditions shape the solution.
- **Reachability:** some equilibria exist but are not reachable. Driving on the left and on the right are both equilibria; a country keeps the one its history gave it, since the first driver to switch crashes. Each European country likewise settles on whichever messaging app its people's contacts use (WhatsApp, Facebook Messenger or Viber).

Learners **tend to converge to "safe" equilibria**, not necessarily optimal ones, because **costly exploration mistakes dissuade agents**. In a gridworld [[Stag Hunt]], a hunter collects a plant alone for +1 or the stag together for +5; one who tries for the stag while the other takes plants gets nothing, so both learn plants are safe. It is a very tough exploration problem, outcomes are hard to predict, and complexity grows with the number of agents.

### Scalability: the curse of dimensionality

Joint spaces grow exponentially. In an 8×8 grid with 5 actions per agent, $N$ agents have

$$|\mathcal{S}_{\text{joint}}| = 64^N, \qquad |\mathcal{A}_{\text{joint}}| = 5^N$$

Four agents give $64^4 \approx 1.7 \times 10^7$ joint states and $625$ joint actions; ten give about $10^{18}$ and $10^7$. And **every other issue gets worse**: more policies chasing each other (more factors in the product over $j \neq i$), harder credit assignment, equilibria harder to escape.

## Independent learning: treat everyone else as environment

**Naive** MARL methods reuse single-agent RL, either **centralized** (one learner chooses the joint action, as if one agent controlled all) or **decentralized** (each agent learns alone). **Multi-agent-aware** methods (joint action learning, agent modelling, extensions of IGA (infinitesimal gradient ascent), no-regret learning) are in the Albrecht book. The course covers only the decentralized **independent Q-learning (IQL)**.

> [!definition] Independent learning
> Each agent runs regular Q-learning on its own actions and rewards and **treats the other agents as part of the environment**. Their changing behaviour is absorbed into the rewards and transitions it experiences, so its MDP is not fixed: $\text{MDP}^i_t \neq \text{MDP}^i_{t+k}$. See [[Independent Learning]].

```pseudo
Algorithm: Independent Q-Learning (N agents, tabular)
─────────────────────────────────────────────────────
Input: learning rate α ∈ (0,1], discount γ ∈ [0,1), exploration rate ε, N agents
Initialize for each agent i:
  Q^i(s, a^i) ← 0 for all s ∈ S, a^i ∈ A^i     // own actions only, no joint actions

Loop for each episode:
  s ← initial state
  Loop until s is terminal:
    For each agent i (all choose at the same time):
      With probability ε: a^i ← random action in A^i     // explore
      Otherwise:          a^i ← argmax_b Q^i(s, b)        // exploit
    Execute the joint action (a^1, ..., a^N)
    Observe next state s' and rewards r^1, ..., r^N  // fully cooperative: all r^i equal
    For each agent i:
      If s' is terminal: target^i ← r^i               // no future value
      Else:              target^i ← r^i + γ · max_b Q^i(s', b)
      Q^i(s, a^i) ← Q^i(s, a^i) + α · (target^i − Q^i(s, a^i))
                      // the others' actions never appear: they act like noise
    s ← s'
  Decay ε (and α) according to the schedule
Output: greedy policy π^i(s) = argmax_b Q^i(s, b) for each agent
```

With $N = 1$ this is ordinary tabular Q-learning.

The update is plain Q-learning; what changes is what it estimates. In a one-state game with shared reward $R(a,b)$, agent A's value averages over its partner B's actions:

$$Q_A(a) \approx \sum_b \pi_B(b)\,R(a,b)$$

with $\pi_B(b)$ how often B currently plays $b$. A good joint action looks bad to A if B rarely plays its half.

### IQL in the climbing game

The **climbing game** is fully cooperative (shared reward):

| | B: a | B: b | B: c |
|---|---|---|---|
| **A: a** | 11 | −30 | 0 |
| **A: b** | −30 | 7 | 6 |
| **A: c** | 0 | 0 | 5 |

(a, a) = 11 is the Pareto-optimal Nash equilibrium, (b, b) = 7 a suboptimal one. A **centralized Q-learner** with one value per joint action finds 11 quickly. **Independent Q-learners get stuck at (c, c) = 5**; (a, a) is unreachable for them.

> [!example] Why independent learners end at (c, c)
> Early on B explores roughly uniformly, so $Q_A(a) \approx (11 - 30 + 0)/3 \approx -6.3$, $Q_A(b) \approx (-30 + 7 + 6)/3 \approx -5.7$, $Q_A(c) \approx (0 + 0 + 5)/3 \approx 1.7$. c looks best because it never risks −30, and by symmetry both agents drift to it. The −30 dissuades exploring: a pays 11 only if the partner also plays a. They are **stuck at minimax**: the action whose worst case is best (0 for c, −30 for the others; game theory calls this the maximin action).
>
> Outside the course material: (c, c) is not even an equilibrium, since with B on c, A would get 6 from b. Exploration hides this: B's occasional random a costs b −30. With B greedy on c and $\varepsilon = 0.2$, B plays c with probability $0.87$ and a, b with $0.07$ each, so $Q_A(b) \approx 0.07(-30) + 0.07(7) + 0.87(6) \approx 3.7$, below $Q_A(c) \approx 0.87 \times 5 \approx 4.3$.

With very low, decaying exploration the independent learners average about 5.5 and the spread across runs widens to roughly 4 to 7, which reads as some runs reaching (b, b) = 7 and others staying at (c, c) = 5: a **non-stationary outcome** that changes from run to run. The centralized learner sits at (b, b) for a while, then still reaches 11.

### IQL in the stag hunt and in level-based foraging

In the gridworld stag hunt, centralized Q-learning approaches the stag's value; the independent learners stay at the plant level for a long time, then improve slowly and only partly, with a large spread across seeds.

In level-based foraging with two agents starting in opposite corners, the result reverses: IQL reaches the optimal return sooner than centralized Q-learning (CQL), if more noisily. **Centralized** learning tracks and explores $|A|^N$ joint actions; **decentralized** learning $|A| \times N$. With 6 actions and 2 agents that is "one choice among 36 versus two choices among 6". So the averaging in $Q_A(a)$ is a **problem for coordination** (it hides good joint actions, as in the climbing game) and a **benefit for scalability** (each agent explores a small space, which pays once the joint space blows up).

### What to take from independent learning

IQL is hard to work with but not fundamentally "broken": messy and without guarantees, it **scales to large, complex environments where central learning becomes intractable**. The recipe also works with IPPO (independent proximal policy optimisation), DDPG (deep deterministic policy gradient), SAC (soft actor-critic) and others, and is a widely used baseline across MARL. Tables still limit it; deep policies, which help generalisation, come next lecture ([[Deep Reinforcement Learning]]).

## What to be able to write

- RL against supervised learning; agent, MDP, return; the Markov property.
- $V^\pi$, $Q^\pi$, Bellman equations, value iteration and why it needs a model; the Q-learning update; challenges of sampling.
- The effects of $\gamma$, $\alpha$ and exploration, with the small computations.
- MARL's dimensions; the three payoff structures and classifying a game.
- Stochastic games, the iterated PD, agent $i$'s objective, the replacement solution concepts, convergence in matching pennies.
- Non-stationarity, credit assignment, equilibrium selection, scalability.
- Independent Q-learning: pseudocode, $Q_A(a) \approx \sum_b \pi_B(b) R(a,b)$, the climbing game, coordination against scalability.

Model answers are in the exam questions below.

## Exam questions

> [!exam]- How does reinforcement learning differ from supervised learning? Describe the components of an RL problem and explain what makes a state Markov.
> **Key points:** RL is closed-loop and sequential, supervised learning open-loop; RL data is not i.i.d. and has rewards instead of labels; agent = policy $\pi_\theta(a \mid s)$, environment = MDP $\{\mathcal{S}, \mathcal{A}, \mathcal{T}, r\}$, goal = maximise expected (discounted) return; Markov: the state summarises the past as far as the future is concerned; hidden information such as a battery level breaks it.
>
> In supervised learning someone provides a fixed dataset of inputs with correct labels, assumed i.i.d., and the model learns $f_\theta(\mathbf{x}) \approx \mathbf{y}$; the model's outputs do not change the data it sees, so it is open-loop. RL is sequential decision making: the agent's actions shape what it sees next. It observes a state, chooses an action, and receives a reward and the next state, so the data is a trajectory produced by its own policy. That data is not i.i.d. (consecutive samples are correlated and depend on the policy), and there are no labels, only a reward saying how good the outcome was.
>
> The agent is its policy $\pi_\theta(a \mid s)$, a probability over actions in each state with parameters $\theta$. The environment is a Markov decision process: states $\mathcal{S}$, actions $\mathcal{A}$, transition probabilities $P(s' \mid s,a)$ and a reward function $R(s,a,s')$. The goal is to maximise the expected discounted return $\mathbb{E}[\sum_k \gamma^k r_{t+k}]$. Algorithms repeat: run the policy to generate samples, estimate returns or fit a model, improve the policy.
>
> A state is Markov when it is a complete summary of the past as far as the future is concerned: the distribution of the next state given the current state and action does not change when earlier history is added. A robot's grid position is Markov only if nothing else matters. If the robot has a battery the state does not include, the same cell can lead to different futures depending on how long it has been running, so the position alone is no longer Markov.

> [!exam]- Define $V^\pi$ and $Q^\pi$, write the Bellman equations for a policy, and explain how iterating them leads to value iteration. Why does this approach need a model?
> **Key points:** $V^\pi$ and $Q^\pi$ as expected discounted returns, specific to a policy; $V^\pi = \sum_a \pi Q^\pi$, $V^* = \max_a Q^*$; Bellman equations as self-consistency conditions, iterated until values stop changing; a greedy policy turns the average into a max (value iteration, a moving target, optimal when it stops changing); needs $P$ and $R$, often unavailable or intractable.
>
> $V^\pi(s) = \mathbb{E}_\pi[\sum_t \gamma^t r_t \mid s_0 = s]$ is the expected discounted return from $s$ following $\pi$; $Q^\pi(s,a)$ is the same but with action $a$ taken first. Both depend on the policy. They are related by $V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s,a)$, and for the optimal policy $V^*(s) = \max_a Q^*(s,a)$.
>
> The Bellman equations express self-consistency: $V^\pi(s) = \sum_a \pi(a \mid s)[r(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V^\pi(s')]$, and similarly $Q^\pi(s,a) = r(s,a) + \gamma \sum_{s'} P(s' \mid s,a) \sum_{a'} \pi(a' \mid s') Q^\pi(s',a')$. A state's value is the step reward plus the discounted expected value of where you land. The values are right when these hold in every state. Starting from any guess and applying them as updates to all states until the values stop changing evaluates the policy.
>
> If the policy is made greedy with respect to the current values, $\pi(s) = \arg\max_a[r(s,a) + \gamma \sum_{s'} P V(s')]$, the update becomes $V(s) \leftarrow \max_a[r(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V(s')]$: value iteration. It chases a moving target, since the policy being evaluated improves after every update, and when the values stop changing the greedy policy is optimal. This works by exploiting the Markov property (decisions can be made state by state) instead of searching $|\mathcal{A}|^{|\mathcal{S}|}$ policies.
>
> The catch is that every update uses $P(s' \mid s,a)$ and $R$. In most problems that model is unknown, or so large that it is intractable (Atari games have astronomically many states); usually there is only a simulator to sample from.

> [!exam]- Derive the Q-learning update from the Bellman equation for $Q$, give its pseudocode, and explain the challenges that learning from samples brings.
> **Key points:** without a model, replace the expectation over next states by one observed transition $(s,a,r,s')$; greedy target $r + \gamma \max_{a'} Q(s',a')$; move by learning rate $\alpha$ towards it; full pass against single sample, noisy target; pseudocode with ε-greedy action choice; challenges: exploration, noisy or high-variance updates and credit assignment, data efficiency.
>
> The Bellman update for $Q$ is $Q(s,a) \leftarrow r(s,a) + \gamma \sum_{s'} P(s' \mid s,a) \sum_{a'} \pi(a' \mid s') Q(s',a')$. Without $P$ and $R$, the agent can only act and observe a transition $(s,a,r,s')$. Q-learning replaces the expectation over $s'$ with that single sample and uses the greedy policy in the target, giving the target $r + \gamma \max_{a'} Q(s',a')$. Because one sample is noisy, the estimate moves only a fraction $\alpha$ of the way: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$.
>
> Pseudocode: initialise $Q$ to 0. For each episode, start in $s$; until terminal, choose $a$ ε-greedily from $Q$, take it, observe $r$ and $s'$, set target $= r$ if $s'$ is terminal and $r + \gamma \max_{a'} Q(s',a')$ otherwise, update $Q(s,a)$ by $\alpha$ times the TD error, set $s \leftarrow s'$. Decay ε and α over episodes; return the greedy policy.
>
> Dynamic programming does a full pass over every state, action and outcome; Q-learning uses one state, one action, one outcome. That brings challenges. Exploration: the agent learns only about states it visits, so it needs an exploration schedule. Noise and credit assignment: single-sample targets give biased, high-variance updates, and delayed rewards have to be traced back to their causes. Data efficiency: samples may be expensive. Value iteration is simple and fast but needs the model; Q-learning needs no prior knowledge but is slower.

> [!exam]- Computation: an agent has $Q(s,a) = 2$, takes $a$, receives $r = 1$ and lands in $s'$ with $\max_{a'} Q(s',a') = 4$. With $\alpha = 0.25$ and $\gamma = 0.5$, what is the new $Q(s,a)$? What would it be if $s'$ were terminal?
> **Key points:** target $= r + \gamma \max_{a'} Q(s',a') = 1 + 0.5 \times 4 = 3$; TD error $= 3 - 2 = 1$; new $Q = 2 + 0.25 \times 1 = 2.25$; terminal: target $= 1$, error $= -1$, new $Q = 1.75$.
>
> The Q-learning update is $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$. The target is $1 + 0.5 \times 4 = 3$. The TD error is $3 - 2 = 1$. The new value is $2 + 0.25 \times 1 = 2.25$. Equivalently, as a running average: $(1 - 0.25) \times 2 + 0.25 \times 3 = 1.5 + 0.75 = 2.25$.
>
> If $s'$ is terminal there is no future value, so the target is just $r = 1$. The TD error is $1 - 2 = -1$ and the new value is $2 + 0.25 \times (-1) = 1.75$.

> [!exam]- What does the discount factor do? Compute the effective horizon for $\gamma = 0.9$ and $\gamma = 0.99$, find the $\gamma$ with an effective horizon of 50 steps, and use the apple orchard to explain how $\gamma$ changes the solution. Then explain the learning rate's role.
> **Key points:** $\gamma$ sets the present value of future rewards, balancing short and long term; effective horizon $1/(1-\gamma)$: 10 and 100 steps, $\gamma = 0.98$ for 50; needed for infinite-horizon tasks, usually high (0.99); orchard: low $\gamma$ over-harvests, high $\gamma$ leaves apples to regrow; $\alpha$ weighs new samples against current beliefs (running average); high $\alpha$ fine when deterministic, noisy when stochastic; low $\alpha$ slow; decay.
>
> The discount factor $\gamma \in [0,1)$ weights a reward $k$ steps ahead by $\gamma^k$, so it sets the present value of future rewards and balances short-term against long-term gains. The effective horizon is $1/(1-\gamma)$, because the weights sum to $\sum_k \gamma^k = 1/(1-\gamma)$: for $\gamma = 0.9$ it is $1/0.1 = 10$ steps, for $\gamma = 0.99$ it is $1/0.01 = 100$ steps. For a horizon of 50, $1/(1-\gamma) = 50$ gives $\gamma = 1 - 1/50 = 0.98$. $\gamma$ is needed mainly for infinite-horizon tasks, where it keeps the sum of rewards finite, and is usually set high (0.99).
>
> $\gamma$ also changes which solution is found. In an orchard where apples regrow at a rate depending on how many are left, an agent with low $\gamma$ cares only about rewards soon, so it picks every apple now and the orchard stops producing. With high $\gamma$, future harvests count, so it leaves enough apples to regrow and harvests sustainably.
>
> The learning rate $\alpha$ controls how much a new sample is trusted against the current estimate: the update is a running average $Q \leftarrow (1-\alpha)Q + \alpha \cdot \text{target}$. In a deterministic environment a high $\alpha$ is fast and fine, and a low one only slows learning. In a stochastic environment a high $\alpha$ makes the estimate jump with every lucky or unlucky outcome so it never settles, while a low $\alpha$ is smooth and accurate but slow. The usual compromise is to decay $\alpha$: large early, small later.

> [!exam]- Computation: with 5 actions and $\varepsilon = 0.1$, give the ε-greedy probability of each action. Compare ε-greedy with Boltzmann exploration, and explain why exploration needs a schedule.
> **Key points:** greedy action $1 - 0.1 + 0.1/5 = 0.92$; each other action $0.1/5 = 0.02$ (sum $0.92 + 4 \times 0.02 = 1$); ε-greedy simple and easy to decay, exploration blind; Boltzmann $\exp(Q/\tau)/\sum\exp(Q/\tau)$, value-aware, hard to tune; too little exploration sticks to the suboptimal, too much wastes samples, so decay.
>
> With probability $1 - \varepsilon = 0.9$ the agent takes the greedy action; with probability $0.1$ it picks uniformly among all 5 actions, which can also be the greedy one. So the greedy action has probability $0.9 + 0.1/5 = 0.92$ and each of the other four $0.1/5 = 0.02$; the total is $0.92 + 0.08 = 1$.
>
> ε-greedy is simple and easy to decay, but its exploration is blind: a catastrophic action is tried as often as a nearly optimal one. Boltzmann exploration chooses $a$ with probability $\exp(Q(s,a)/\tau) / \sum_{a'} \exp(Q(s,a')/\tau)$, so it is value-aware: better actions are tried more. Its temperature $\tau$ is hard to tune, because the right value depends on the scale of the Q-values; low $\tau$ is nearly greedy, high $\tau$ nearly uniform.
>
> Exploration is needed because an agent that never tries an action never finds out it is better: one that keeps refining the first path it found to a +2 goal never discovers a +5 goal. Too much exploration wastes samples on actions already known to be bad. A decay schedule explores a lot early, when little is known, and exploits more later.

> [!exam]- Computation: classify each game as fully cooperative, zero-sum or general-sum, and say what learning has to achieve in each kind: (i) (4,4) (0,0) / (0,0) (2,2); (ii) (1,−1) (−1,1) / (−1,1) (1,−1); (iii) (3,3) (0,5) / (5,0) (1,1).
> **Key points:** rule: equal payoffs in every cell = fully cooperative, payoffs summing to zero in every cell = zero-sum, otherwise general-sum; (i) fully cooperative, (ii) zero-sum (matching pennies), (iii) general-sum (Prisoner's Dilemma); cooperative needs coordination of tasks, general-sum balancing individual and group welfare (Pareto inefficiency), zero-sum robustness to many opponents.
>
> Compare the two payoffs in each cell. In (i) they are always identical, $R_1 = R_2$, so the game is fully cooperative: both agents want the same thing, and the challenge is coordinating on the better outcome (4,4) over (2,2). In (ii) every cell sums to zero, $R_1 = -R_2$, so it is zero-sum (matching pennies): one agent's win is the other's loss, my changes prompt my opponent's counter-changes, and the aim is a policy robust to many kinds of opponent. In (iii) the payoffs differ and the sums are 6, 5, 5 and 2, so it is neither: general-sum, the Prisoner's Dilemma. Agents have their own preferences, so besides coordinating they must balance individual and group welfare, and learners struggle with Pareto inefficiency, here mutual defection (1,1), which both agents rank below (3,3).
>
> The payoff structure matters because it defines what a good solution is, what can go wrong and which algorithms make sense.

> [!exam]- Define a stochastic game and formulate the iterated Prisoner's Dilemma as one. Why is "maximise your return" no longer an adequate solution concept, and what does MARL optimise for instead?
> **Key points:** multi-agent generalisation of an MDP, $N$ agents acting simultaneously, each with its own reward; tuple $(\mathcal{N}, \mathcal{S}, \{\mathcal{A}^i\}, P, \{R^i\}, \gamma)$ with transitions and rewards depending on the joint action; IPD: $N = 2$, states CC, CD, DC, DD (last joint action), actions C, D, rewards from the PD matrix; $J^i$ depends on all policies; alternatives: stability (Nash equilibrium), no regret, social welfare, robustness to opponents.
>
> A stochastic game generalises an MDP to $N$ agents that act simultaneously, each with its own reward. It is the tuple $(\mathcal{N}, \mathcal{S}, \{\mathcal{A}^i\}, P, \{R^i\}, \gamma)$: the set of agents, a shared state space, an action set per agent, a transition function $P(s' \mid s, a^1, \dots, a^N)$ that depends on the joint action, a reward function $R^i(s, a^1, \dots, a^N, s')$ for each agent, and a discount factor.
>
> The iterated Prisoner's Dilemma has $N = 2$. The state is the previous round's joint action, $\mathcal{S} = \{CC, CD, DC, DD\}$, and each agent's actions are $\{C, D\}$. The next state is fixed by the current joint action (play C and D and the next state is CD). Rewards follow the PD matrix: 3 each for mutual cooperation, 5 and 0 when one defects on a cooperator, 1 each for mutual defection. Policies can condition on the state, for example cooperating if the other cooperated last round.
>
> Each agent's objective is $J^i(\pi^1, \dots, \pi^N) = \mathbb{E}[\sum_t \gamma^t R^i_t \mid \pi^1, \dots, \pi^N]$: its return depends on everyone's policies, but it controls only its own. "Maximise your return" therefore does not pin down a solution, because my best policy depends on yours and yours on mine. MARL optimises instead for stability (a Nash equilibrium, where no agent gains by changing only its own policy), no regret, social welfare (total reward), or robustness to many kinds of opponent, depending on the setting.

> [!exam]- Using matching pennies, explain why convergence in multi-agent learning is not as simple as in an MDP, and give the different notions of convergence.
> **Key points:** in an MDP, converged = the policy stops changing; matching pennies: matcher against mismatcher, zero-sum; any change by one player gives the other a reason for the opposite change, so learning dynamics orbit; equilibrium 0.5/0.5; cycling policies match the equilibrium on average and earn the same average reward; notions: exact policy, average policy over many episodes, average reward.
>
> In an MDP, learning has converged when the policy stops changing, and the optimal policy is then reached. In matching pennies each player picks heads or tails; the row player wins (+1, the other −1) if they match, the column player if they differ, like a goalkeeper and a penalty taker. If the row player leans towards heads, the column player's best response is tails; then the row player's best response is tails, then the column player's is heads, and so on. Any change one makes gives the other a reason to make the opposite change, so the learning dynamics orbit around the equilibrium, where each plays heads with probability 0.5, and the policies never settle.
>
> Yet averaged over time the cycling policies play each action half the time and earn the equilibrium's average reward (0 each). Whether they have "converged" depends on the notion: the policy matching the equilibrium exactly (not satisfied), the policy matching it on average over many episodes (satisfied), or the average reward matching that of the optimal policy (satisfied). Multi-agent learning therefore has to say which property it wants from a solution.

> [!exam]- Explain non-stationarity in MARL using the transition function agent $i$ experiences, and describe its consequences.
> **Key points:** other agents are part of agent $i$'s environment; $P^i(s' \mid s, a^i) = \sum_{a^{-i}} [\prod_{j \neq i} \pi^j(a^j \mid s)] P(s' \mid s, a^i, a^{-i})$ changes whenever $\pi^{-i}$ changes; single-agent targets move only as own estimates improve, MARL targets also move as others' policies change; experience has a shelf life; cycling, broken convergence guarantees, noisy targets with low signal-to-noise ratio.
>
> From agent $i$'s point of view the other agents are part of the environment. The probability of reaching $s'$ after playing $a^i$ is $P^i(s' \mid s, a^i) = \sum_{a^{-i}} [\prod_{j \neq i} \pi^j(a^j \mid s)] P(s' \mid s, a^i, a^{-i})$: the true, fixed transition function averaged over what the others might do, weighted by their policies. When the others learn, $\pi^{-i}$ changes, so $P^i$ changes too, and so does agent $i$'s expected reward. The environment agent $i$ is learning about is not stationary.
>
> In single-agent RL the learning targets move only because the agent's own estimates improve. In MARL they also move because the other agents' policies change, so an agent's experience has a shelf life: old targets it bootstraps from become outdated. Consequences: cycling, where best responses chase each other as in matching pennies and both agents keep improving without converging; broken guarantees, since convergence proofs for MDP methods such as Q-learning assume a fixed environment; and noisy targets, since bootstrapping from stale estimates gives a low signal-to-noise ratio. In practice it is a moving signal that is hard to track, and it gets worse as more agents' policies change at once.

> [!exam]- What is multi-agent credit assignment? Explain it with the lazy agent in level-based foraging and with the fishing environment, and say why it is hardest in fully cooperative games.
> **Key points:** single-agent RL has temporal credit (which earlier action), MARL adds agent credit (whose action); lazy agent: shared team reward reinforces an idle agent when teammates succeed; disentangling needs counterfactuals and many samples, a hard causal inference problem with low signal-to-noise ratio; worst in fully cooperative games (no individual signal); also in general-sum: fishing, wrong lessons from good and bad outcomes.
>
> Single-agent RL faces temporal credit assignment: deciding which earlier actions led to a reward. With several agents there is a new axis: deciding whose actions led to it. In level-based foraging, three agents collect apples and share a team reward of +1 whenever an apple is collected. If A and B do the work while C wanders in a corner, C still receives +1, so whatever C was doing is reinforced. The lazy agent is rewarded because its teammates succeeded.
>
> Separating each agent's contribution requires counterfactual reasoning (what would have happened had C acted differently?), which takes many samples and a hard exploration effort. It adds a layer to an already hard causal inference problem, at a very low signal-to-noise ratio.
>
> It is worst in fully cooperative games, because a shared reward carries no individual signal at all. It also arises in general-sum games, where rewards are individual but depend on the others. In a fishing environment with a lake that can be depleted, an agent that fishes aggressively while others hold back catches a lot and learns that aggression is good, though it only paid because others held back. An agent that fishes sustainably while the others hold back too catches little and learns that sustainability is bad, though that shared restraint is what keeps the lake stocked. (If the others were fishing aggressively instead, the small catch would be their doing, and the agent would wrongly blame its own sustainable fishing.) In both cases it credits its own action with an outcome that depended on everyone.

> [!exam]- What is the equilibrium selection problem? Why is it called native and inescapable, why do learners tend to reach "safe" equilibria, and how does the number of agents affect this and the other challenges?
> **Key points:** games can have several equilibria, and even perfect learners can converge to a suboptimal one; native to multiple agents and not fixed by better learning; path dependence (history and initial conditions decide), some equilibria exist but are unreachable (side of the road, messaging apps, corrupt government); costly exploration mistakes push learners to safe equilibria (stag hunt); more agents: exponential joint spaces ($64^N$, $5^N$), more policies chasing each other, harder credit assignment, equilibria harder to escape.
>
> Many games have several equilibria, and which one learners end up in is not determined by the game alone. Even perfect learners, each best-responding correctly, can converge to a suboptimal equilibrium. This is native to multi-agent settings and inescapable: it comes from the interaction, not from estimation error, so better single-agent learning does not remove it. Humans show it constantly. Outcomes are path dependent, shaped by history and initial conditions. Some equilibria exist but cannot be reached: driving on the left or on the right are both equilibria, a country keeps whichever history gave it, and the first driver to switch crashes; in each country people use the messaging app their contacts use; corruption persists where playing along pays as long as others do.
>
> Learners tend to converge to safe equilibria because exploration mistakes are costly. In the stag hunt, a hunter who tries for the stag while the partner collects plants gets nothing, while a plant always pays +1, so both learn that plants are safe and the +5 stag equilibrium stays out of reach. It is a very tough exploration problem with hard-to-predict outcomes.
>
> More agents make everything worse. Joint state and action spaces grow exponentially (in an 8×8 grid with 5 actions, $64^N$ joint states and $5^N$ joint actions). Non-stationarity comes from more policies changing at once, credit has to be split among more agents, and equilibria become harder to escape.

> [!exam]- What is independent Q-learning? Give its pseudocode, explain what each agent's Q-values estimate, and use the climbing game to explain why independent learners miscoordinate.
> **Key points:** each agent runs ordinary Q-learning over its own actions and treats the others as part of the environment ($\text{MDP}^i_t \neq \text{MDP}^i_{t+k}$); pseudocode; $Q_A(a) \approx \sum_b \pi_B(b) R(a,b)$, the reward averaged over the partner's actions; climbing game: against a roughly uniform partner c averages best (1.7 against −5.7 and −6.3), the −30 dissuades exploring, stuck at (c,c) = 5 ("minimax") while the optimum (a,a) = 11 is unreachable; centralized Q over joint actions finds 11.
>
> In independent Q-learning each agent $i$ keeps its own table $Q^i(s, a^i)$ over its own actions and updates it with the ordinary rule $Q^i(s,a^i) \leftarrow Q^i(s,a^i) + \alpha[r^i + \gamma \max_b Q^i(s',b) - Q^i(s,a^i)]$. The other agents' actions never appear; their behaviour is absorbed into the transitions and rewards agent $i$ experiences, so the MDP it faces changes over time.
>
> Pseudocode: initialise every $Q^i$ to 0. Each step, every agent picks its action ε-greedily from its own table; the joint action is executed; the next state and each agent's reward are observed; each agent updates its own entry with its own target; decay ε over time.
>
> In a one-state game with shared reward $R(a,b)$, agent A's estimate becomes $Q_A(a) \approx \sum_b \pi_B(b) R(a,b)$, the reward of $a$ averaged over how often B plays each action. In the climbing game (rows a, b, c: 11, −30, 0 / −30, 7, 6 / 0, 0, 5), (a,a) = 11 is the optimal equilibrium and (b,b) = 7 a worse one. Early on the partner explores roughly uniformly, so $Q_A(a) \approx -6.3$, $Q_A(b) \approx -5.7$ and $Q_A(c) \approx 1.7$: c looks best because it never risks −30. Both agents drift to c and get stuck at (c,c) = 5, the action with the best worst case. Action a pays only if the partner also plays a, and trying it otherwise costs −30, so (a,a) is unreachable. A centralized learner with one value per joint action sees 11 directly and finds it. With low, decaying exploration the independent learners sometimes reach (b,b), but the outcome varies between runs.

> [!exam]- Why is independent learning still widely used despite its problems? Compare centralized and decentralized learning for 2 agents with 6 actions each, and for 3 agents with 5 actions each.
> **Key points:** centralized explores $|A|^N$ joint actions, decentralized $|A| \times N$; 2 agents, 6 actions: 36 against 2 × 6 = 12; 3 agents, 5 actions: 125 against 15; averaging hides coordination (problem) but keeps each agent's space small (benefit); in level-based foraging IQL reaches the optimum sooner than centralized Q; no guarantees, messy, but scales where central learning is intractable; also IPPO, DDPG, SAC; standard baseline.
>
> A centralized learner treats all agents as one, so it must keep track of and explore every joint action: $|A|^N$ of them. Independent learners each handle only their own actions: $|A| \times N$ in total. With 2 agents and 6 actions that is one choice among 36 against two choices among 6 (12 values per state). With 3 agents and 5 actions it is 125 joint actions against 15.
>
> The averaging in $Q_A(a) \approx \sum_b \pi_B(b) R(a,b)$ cuts both ways. It is a problem for coordination, because it hides which joint actions are good, as in the climbing game. It is a benefit for scalability, because each agent explores a small space. In level-based foraging with two agents and two apples, independent Q-learning reached the optimal return sooner than centralized Q-learning, though more noisily, because it explores more efficiently as the joint space grows.
>
> So independent Q-learning is messy and has no convergence guarantees (the environment is non-stationary for each agent), but it is not fundamentally broken, and it scales to large, complex environments where central learning becomes intractable. The same idea applies to other single-agent algorithms (IPPO, DDPG, SAC), and independent learning is a standard baseline across MARL.

## Flashcards

> [!card]- What makes reinforcement learning closed-loop?
> The agent's own actions determine what data it sees next.

> [!card]- Why is the data in reinforcement learning not i.i.d.?
> Consecutive samples are correlated and depend on the agent's policy.

> [!card]- What are the four parts of an MDP $\{\mathcal{S}, \mathcal{A}, \mathcal{T}, r\}$?
> States, actions, transition function $P(s' \mid s,a)$, reward function $R(s,a,s')$.

> [!card]- When is a state Markov?
> When it summarises everything about the past that matters for the future.

> [!card]- How does a hidden battery break the Markov property for a robot in a grid?
> The same cell leads to different futures depending on the unobserved battery level.

> [!card]- How many deterministic policies does an MDP with $|\mathcal{S}|$ states and $|\mathcal{A}|$ actions have?
> $|\mathcal{A}|^{|\mathcal{S}|}$.

> [!card]- How are $V^\pi(s)$ and $Q^\pi(s,a)$ related?
> $V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s,a)$.

> [!card]- What do the Bellman equations express?
> Self-consistency: a state's value equals step reward plus discounted expected next-state value.

> [!card]- How does value iteration turn policy evaluation into finding the optimal policy?
> It makes the policy greedy, so the average over actions becomes a max.

> [!card]- Why can dynamic programming not be used in most RL problems?
> It needs the transition and reward model, which is usually unavailable or intractable.

> [!card]- How does Q-learning do without a model?
> It replaces the expectation over next states with one observed transition.

> [!card]- What is the Q-learning target?
> $r + \gamma \max_{a'} Q(s',a')$.

> [!card]- Why is the Q-learning target noisy?
> It is built from a single sampled transition.

> [!card]- What is the effective horizon of a discount factor $\gamma$?
> $1/(1-\gamma)$ steps.

> [!card]- What is the main reason a discount factor is needed?
> Infinite-horizon tasks: it keeps the infinite sum of rewards finite.

> [!card]- What does a low discount factor do in an apple orchard where apples regrow?
> The agent picks everything now, so the orchard stops regrowing.

> [!card]- Why is the Q-learning update a running average?
> $Q \leftarrow (1-\alpha)Q + \alpha \cdot$ target mixes the old estimate with the new target.

> [!card]- What happens with a high learning rate in a stochastic environment?
> Estimates stay noisy and never settle.

> [!card]- With ε-greedy exploration, what is the probability of the greedy action?
> $1 - \varepsilon + \varepsilon/|\mathcal{A}|$.

> [!card]- What is the weakness of ε-greedy exploration?
> It is blind: bad actions are tried as often as nearly-best ones.

> [!card]- What is the advantage of Boltzmann exploration over ε-greedy?
> It is value-aware: better actions are explored more often.

> [!card]- Why always run several random seeds in RL experiments?
> Results vary a lot between seeds.

> [!card]- How are the rewards related in a fully cooperative game?
> $R_1 = R_2$: identical in every outcome.

> [!card]- How are the rewards related in a zero-sum game?
> $R_1 = -R_2$: one agent's win is the other's loss.

> [!card]- What do learners in general-sum games struggle with?
> Pareto inefficiency: outcomes where everyone could be better off.

> [!card]- What is a stochastic game?
> The multi-agent generalisation of an MDP: simultaneous actions and a reward per agent.

> [!card]- What is the multi-agent generalisation of an MDP called?
> A stochastic game.

> [!card]- In the iterated Prisoner's Dilemma as a stochastic game, what is the state?
> Last round's actions of both players: CC, CD, DC or DD.

> [!card]- Why is "maximise your return" not enough as a solution concept in MARL?
> Each agent's return depends on every agent's policy.

> [!card]- Name four criteria MARL can optimise for instead of own return.
> Stability (Nash equilibrium), no regret, social welfare, robustness to opponents.

> [!card]- Why do learning dynamics orbit in matching pennies?
> Any change by one player gives the other a reason for the opposite change.

> [!card]- In which senses do cycling learners in matching pennies converge?
> On average: their average policy and average reward match the equilibrium.

> [!card]- Why is the environment non-stationary for a learner in MARL?
> Its experienced transitions and rewards change whenever the other agents' policies change.

> [!card]- What does "experience has a shelf life" mean in MARL?
> Old targets an agent bootstraps from go stale as the other agents change.

> [!card]- Name the consequences of non-stationarity in MARL.
> Cycling, broken convergence guarantees, noisy targets.

> [!card]- How does agent credit assignment differ from temporal credit assignment?
> Temporal: which earlier action caused the reward. Agent: whose action did.

> [!card]- What is the lazy agent problem?
> An idle agent is reinforced because its teammates earned the shared reward.

> [!card]- What is it called when an idle agent is reinforced because its teammates earned a shared reward?
> The lazy agent problem, a case of multi-agent credit assignment.

> [!card]- Why is credit assignment worst in fully cooperative games?
> A shared team reward gives no individual signal at all.

> [!card]- What is path dependence in equilibrium selection?
> History and initial conditions decide which equilibrium is reached.

> [!card]- Why do learners tend to reach "safe" equilibria?
> Costly exploration mistakes discourage trying the risky, better joint action.

> [!card]- How many joint actions do $N$ agents with 5 actions each have?
> $5^N$.

> [!card]- How does an independent learner treat the other agents?
> As part of the environment.

> [!card]- What does an independent learner's $Q_A(a)$ estimate in a one-state game?
> $\sum_b \pi_B(b) R(a,b)$: the reward averaged over the partner's actions.

> [!card]- Where do independent Q-learners end up in the climbing game?
> At (c, c) = 5, the safe action, missing the optimum (a, a) = 11.

> [!card]- Why does IQL scale better than centralized Q-learning?
> It explores $|A| \times N$ actions instead of $|A|^N$ joint actions.

> [!card]- True or false: independent Q-learning keeps Q-learning's convergence guarantee in multi-agent settings.
> **False.** The other learners make each agent's environment non-stationary, so the guarantee no longer holds.

## Links

- **Course:** [[AGAI - Overview|Course overview]]
- **Previous lecture:** [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]] (non-stationarity as a difficulty of learning agents; the Prisoner's Dilemma and Stag Hunt)
- **RL background in depth:** [[RL-L01 - Intro, MDPs & Bandits]] (MDPs, returns, values), [[RL-L02 - Dynamic Programming]] (Bellman equations, value iteration), [[RL-L04 - Temporal Difference Learning]] (Q-learning)
- **Lab:** [[AGAI-Lab02 - Learning Agents]] (equilibrium selection and path dependence, learning dynamics in gridworlds, the limits of tabular RL for MARL)
- **Concepts introduced here:** [[Stochastic Game]], [[Independent Learning]], [[Multi-Agent Credit Assignment]]
- **Source:** Agentic Artificial Intelligence, Lecture 3, "Intro to Multi-Agent Reinforcement Learning", Roman Chiva Gil, MSc AI, UvA, 3 November 2026
- **Background reading (optional, suggested for weeks 2 and 3):** Albrecht, Christianos and Schäfer, *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches* (MIT Press, 2024), Chapters 5 to 9, for the multi-agent-aware algorithms (joint action learning, agent modelling, IGA extensions, no-regret learning) not covered in the lecture
- **Also mentioned:** Acemoglu and Robinson, *Why Nations Fail*
