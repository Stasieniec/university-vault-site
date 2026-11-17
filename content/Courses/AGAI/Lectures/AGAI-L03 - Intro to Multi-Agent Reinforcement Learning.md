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
> A single agent learns in a Markov decision process: value iteration finds the optimal policy when the model is known, Q-learning learns from samples when it is not. With several learners the problem becomes a stochastic game, where transitions and each agent's reward depend on the joint action. Four things make it hard: non-stationarity, credit assignment, equilibrium selection and scalability. The two naive fixes are central learning (one learner over joint actions) and independent learning (each agent runs Q-learning and treats the others as environment). Independent learning has no guarantees but scales, and it is the standard baseline.
>
> Feeds mock questions 7 to 11 (tabular Q-learning limits, value iteration against Q-learning, advantages of independent learning, the Coin Game, central Q-learning) and, with [[AGAI-Lab02 - Learning Agents|Lab 2]], the open question 12 (Stag Hunt with independent Q-learners).

## Reinforcement learning in one page

[[Reinforcement Learning]] is **sequential decision making**: your actions shape what you see next (chess, robot control; forecasting is not, because a forecast does not change the future). Unlike supervised learning, RL is **closed-loop**: the data is **not i.i.d.** (it depends on how the agent acts) and there are **no labels, only reward**.

An RL problem has three parts:

- **Agent:** a [[Policy|policy]] $\pi(a \mid s)$, the probability of each action in each state.
- **Environment:** a [[Markov Decision Process|Markov decision process]] (MDP): states $\mathcal{S}$, actions $\mathcal{A}$, a transition function $P(s' \mid s, a)$ and a reward function $R(s, a, s')$.
- **Goal:** maximise the expected discounted return.

> [!definition] Markov property
> The state is a **complete summary of the past, as far as the future is concerned**. A robot dog whose grid cell is its state, but which has a battery the state leaves out, breaks it: the same cell leads to different futures depending on how long it has walked.

The return weights a reward $k$ steps ahead by $\gamma^k$, with [[Discount Factor|discount factor]] $\gamma \in [0, 1)$:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k=0}^{\infty}\gamma^k r_{t+k}$$

Maximising it is the single-agent **solution concept** (the criterion for what counts as a solution), and an [[Optimal Policy|optimal policy]] always exists: a map from state to action that is best from every starting state.

## Values and value iteration

$V^\pi(s)$ is the expected discounted return from state $s$ when following $\pi$; $Q^\pi(s,a)$ is the same after first taking action $a$. Both are **specific to a policy**, and $V^*(s) = \max_a Q^*(s,a)$. "A state is only as good as where it leads": values satisfy the [[Bellman Equation|Bellman equations]], self-consistency conditions that hold in every state when the values are right.

> [!formula] Bellman equation for $Q^\pi$ (recognise it)
> $$Q^\pi(s,a) = r(s,a) + \gamma\sum_{s'}P(s' \mid s,a)\sum_{a'}\pi(a' \mid s')\,Q^\pi(s',a')$$
> Step reward, plus the discounted value of where you land, averaged over next states (with their probabilities $P$) and next actions (with the policy's probabilities).

Starting from any guess and applying the equation as an update until the values stop changing evaluates a policy. Making the policy **greedy** turns the average over actions into a max:

> [!formula] Value iteration
> $$V(s) \leftarrow \max_a\Big[r(s,a) + \gamma\sum_{s'}P(s' \mid s,a)\,V(s')\Big]$$

When the values stop changing, the greedy policy is optimal ([[Value Iteration]]). **The catch: value iteration needs the model**, the transition function $P(s' \mid s,a)$ and the reward function $R(s,a,s')$. That model is often unavailable or intractable; usually there is only a simulator to sample from.

## Q-learning: replace the model with samples

With no model, the agent only has experience $(s, a, r, s')$, one transition at a time. [[Q-Learning]] replaces the sum over next states with **one observed transition** and uses the greedy action in the target.

> [!formula] Q-learning update (recognise it)
> $$Q(s,a) \leftarrow Q(s,a) + \alpha\Big[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\Big]$$
> - $r + \gamma\max_{a'}Q(s',a')$: the **target**, built from one sample, so it is noisy
> - $\alpha$: learning rate, how far to move towards the target
> - if $s'$ is terminal there is no future value: the target is just $r$

Sampling brings new challenges: **exploration** (you only learn about states you visit), **noise and credit assignment** (biased, high-variance updates) and **data efficiency** (samples can be expensive).

| | Value iteration | Q-learning |
|---|---|---|
| Needs | the model $P$, $R$ | only samples |
| Strengths | simple, fast | no prior knowledge |
| Weaknesses | model often unavailable | exploration, credit assignment, slow |

## Discount, learning rate and exploration

**Discount factor $\gamma$**: the present value of future rewards. The **effective horizon** is $1/(1-\gamma)$: 10 steps for $\gamma = 0.9$, 100 for $0.99$. It is usually set high (0.99) and is needed mainly for **infinite-horizon tasks**, where it keeps the return finite. It also changes the solution: in an apple orchard whose regrowth depends on the apples left, a low $\gamma$ picks everything now and nothing regrows, a high $\gamma$ harvests sustainably.

**Learning rate $\alpha$**: how much a new sample is trusted against current belief. The update is a running average, $Q \leftarrow (1-\alpha)Q + \alpha \cdot \text{target}$. In a deterministic environment a high $\alpha$ is fast and fine; in a stochastic one (a slippery grid) it is noisy and never settles, while a low $\alpha$ is smooth but slow. Fix: **decay $\alpha$**.

**Exploration**: if you never try things, you never discover they are better. Too little sticks with the suboptimal, too much wastes samples ([[Exploration vs Exploitation]]); fix: a **decay schedule**.

> [!formula] Two styles of exploration
> **[[Epsilon-Greedy Policy|ε-greedy]]:** with probability $\varepsilon$ a uniformly random action, otherwise the greedy one. The greedy action gets $1 - \varepsilon + \varepsilon/|\mathcal{A}|$, every other action $\varepsilon/|\mathcal{A}|$. Simple and easy to decay, but **blind**: a disastrous action is tried as often as a nearly-best one.
>
> **Boltzmann ([[Softmax Policy|softmax]]):** $\pi(a \mid s) = \exp(Q(s,a)/\tau) / \sum_{a'}\exp(Q(s,a')/\tau)$. **Value-aware** (better actions are tried more), but hard to tweak: a good temperature $\tau$ depends on the scale of the Q-values. Low $\tau$ is nearly greedy, high $\tau$ nearly uniform.

## What a Q-table cannot do

[[Tabular RL|Tabular]] Q-learning stores one value per state-action pair. That has two limits, which motivate deep RL in [[AGAI-L04 - Deep RL and Intro to Deep MARL|Lecture 4]] and the lamp experiment in [[AGAI-Lab02 - Learning Agents|Lab 2]]:

- **No generalisation.** Each value is updated in isolation, so experience in one state does not carry over to similar states, and the agent has to visit every state-action pair to get accurate values.
- **Size.** The table grows with the number of state-action pairs, so it becomes too large for complex environments.

What is *not* a limit: Q-learning does not need the transition function (that is value iteration's problem), and it handles stochastic rewards (the learning rate averages the noise).

## Many learners: stochastic games

In multi-agent RL (MARL), **agents interact in a shared environment to achieve their own goals**: each agent $i$ has its own policy $\pi^i(a \mid s)$, transitions and rewards depend on the **joint action** (everyone's actions together), and goals need not be aligned.

The payoff structure decides what a good solution means:

| | Fully cooperative | General-sum | Zero-sum |
|---|---|---|---|
| Rewards | $R_1 = R_2$ | anything in between | $R_1 = -R_2$ |
| Learning must | coordinate how to split tasks | also balance individual and group welfare | stay robust to many opponents |

General-sum learners struggle with **Pareto inefficiency** (outcomes where everyone could be better off).

> [!definition] Stochastic game
> The **multi-agent generalisation of an MDP**: $N$ agents act **simultaneously**, and **each agent has its own reward**. A [[Stochastic Game]] $(\mathcal{N}, \mathcal{S}, \{\mathcal{A}^i\}, P, \{R^i\}, \gamma)$ has:
> - a set of agents $\mathcal{N}$ and a **shared state space** $\mathcal{S}$
> - an **action set $\mathcal{A}^i$ for each agent**
> - a **transition function** $P(s' \mid s, a^1, \dots, a^N)$ that depends on the joint action
> - a **reward function $R^i(s, a^1, \dots, a^N, s')$ for each agent**, and a discount $\gamma$
>
> Not part of the definition: observation functions (those belong to the partially observable version) and policies (those are what the agents choose). [[AGAI-L02 - Game Theory and Population Dynamics|Lecture 2]] writes the same object with transitions $T$.

**Example: the iterated Prisoner's Dilemma.** Two agents, actions C and D, the state is **both players' actions in the last round** (CC, CD, DC, DD), the joint action fixes the next state, and rewards come from the PD matrix. A stochastic game with a single state is a repeated [[Normal-Form Game|normal-form game]].

Agent $i$'s objective $J^i(\pi^1, \dots, \pi^N) = \mathbb{E}[\sum_t \gamma^t R^i_t]$ **depends on everyone's policy**, but agent $i$ controls only its own. So "maximise your return" is **no longer expressive enough**: my best policy depends on yours and yours on mine. MARL optimises instead for **stability** (a [[Nash Equilibrium]]), **no regret**, **social welfare** or **robustness to opponents**.

Convergence also changes meaning. In **matching pennies** (I win if our coins match, you win if they differ) any change I make gives you a reason for the opposite change, so learning dynamics **orbit** the equilibrium (heads with probability 0.5) without settling. On average, though, the cycling policies play each action half the time and earn the equilibrium's reward. They have converged in **average policy** and **average reward**, though not in **exact policy**.

## Why multi-agent learning is hard

**Non-stationarity** and **credit assignment** limit **how well** agents learn (both make the signal very noisy). **Equilibrium selection** decides **where they end up**. **Scalability** makes all of them worse.

**Non-stationarity: experience has a shelf life.** To agent $i$ the other agents are part of the environment:

$$P^i(s' \mid s, a^i) = \sum_{a^{-i}}\Big[\prod_{j \neq i}\pi^j(a^j \mid s)\Big]\,P(s' \mid s, a^i, a^{-i})$$

the true transitions averaged over what the others play, which **changes whenever their policies change**. In single-agent RL targets move only because my estimates improve; in MARL they also move because the others' policies change, so old targets go stale. Consequences: **cycling** (best responses chase each other), **broken guarantees** (MDP convergence proofs, such as Q-learning's, no longer hold), and **noisy targets** (low signal-to-noise ratio).

**Credit assignment: whose action earned the reward.** Single-agent RL has temporal credit (which earlier action?); MARL adds **agent credit** (whose action?). In level-based foraging with a shared team reward, agents A and B collect apples while C wanders in a corner, and C's wandering is reinforced: **the lazy agent**. Untangling contributions needs counterfactuals and many samples. It is worst in **fully cooperative games** (no individual signal at all) but also present in general-sum ones: in a shared fishing lake, I fish aggressively while the others hold back, catch lots, and wrongly learn that aggression is good. See [[Multi-Agent Credit Assignment]].

**Equilibrium selection: native and inescapable.** Even perfect learners can converge to a **suboptimal equilibrium**, because the game itself has several. Outcomes are **path dependent** (history and initial conditions decide), and some equilibria exist but are **not reachable** (a country that drives on the left stays there: the first driver to switch crashes). Learners tend to reach **"safe" equilibria**, because **costly exploration mistakes dissuade them**: in a gridworld [[Stag Hunt]], a hunter who goes for the stag while the other collects plants gets nothing, so both learn that plants are safe.

**Scalability: the curse of dimensionality.** Joint spaces grow exponentially: in an 8×8 grid with 5 actions per agent, $N$ agents have $64^N$ joint states and $5^N$ joint actions. And every other problem gets worse with more agents: more policies chasing each other, harder credit assignment, equilibria harder to escape.

## Central learning and independent learning

The naive approaches reuse single-agent RL in one of two ways (multi-agent-aware methods such as agent modelling are not covered).

> [!definition] Central Q-learning
> **One Q-function over the state and the joint action**, $Q(s, a^1, \dots, a^N)$, learned by a single learner that **chooses the joint action for all agents**, as if they were one agent.

It sees the value of every joint action, so it can find coordinated optima. Its limits: it has to track and explore $|A|^N$ joint actions, which blows up with the number of agents; it needs a central controller that sees and commands everyone; and (from the MARL book, Chapter 5) it needs the agents' rewards turned into one scalar, which is straightforward with a shared reward and unclear in general-sum games.

> [!definition] Independent learning
> **Each agent runs regular Q-learning on its own actions and its own reward and treats the other agents as part of the environment.** Agent $i$ keeps $Q^i(s, a^i)$; the others' changing behaviour is absorbed into the rewards and transitions it experiences, so its MDP changes over time ($\text{MDP}^i_t \neq \text{MDP}^i_{t+k}$). See [[Independent Learning]].

What an independent learner's value means, in a one-state game with shared reward $R(a, b)$:

$$Q_A(a) \approx \sum_b \pi_B(b)\,R(a,b)$$

the reward of $a$ **averaged over what the partner currently plays**. A good joint action looks bad if the partner rarely plays its half.

**Advantages of independent learning:**
- **Scales**: each table covers only the agent's own actions, $|A| \times N$ values instead of $|A|^N$ ("two choices among 6 against one choice among 36" for 2 agents with 6 actions).
- **Agents do not need to observe the other agents' actions.**
- Works with any single-agent algorithm (IPPO, DDPG, SAC, ...), and is a widely used baseline across MARL.

**Limitations:**
- Each agent's environment is **non-stationary**, so Q-learning's **convergence guarantees are lost**; nothing guarantees a Nash equilibrium or the social optimum.
- The averaging **hides good joint actions**, so agents miscoordinate and settle on safe actions.

### The climbing game: miscoordination

A fully cooperative game (both agents get the number in the cell):

| | B: a | B: b | B: c |
|---|---|---|---|
| **A: a** | 11 | −30 | 0 |
| **A: b** | −30 | 7 | 6 |
| **A: c** | 0 | 0 | 5 |

(a, a) = 11 is the optimum. A **central Q-learner** with one value per joint action finds it quickly. **Independent learners get stuck at (c, c) = 5.** Early on the partner explores roughly uniformly, so averaged over its actions a looks bad (about −6.3), b too (about −5.7), and c best (about 1.7), because c never risks −30. The −30 dissuades exploring a, which pays only if the partner plays a at the same moment. The course calls this being **stuck at minimax**: the action with the best worst case. With very low, decaying exploration the outcome changes from run to run.

### When independent learning wins

In level-based foraging with two agents, independent Q-learning reached the optimal return sooner than central Q-learning, because each agent explores a small space. So the averaging in $Q_A(a)$ is a **problem for coordination** and a **benefit for scalability**: messy and without guarantees, independent learning scales to large environments where central learning becomes intractable. [[AGAI-Lab02 - Learning Agents|Lab 2]] runs it in the Stag Hunt and the Coin Game.

## What to be able to answer

- RL against supervised learning; agent, MDP, return; the Markov property.
- The Bellman equation and the Q-learning update on sight; terminal target $= r$.
- Why value iteration needs the model and Q-learning does not; what sampling costs.
- What $\gamma$, $\alpha$ and ε do; ε-greedy against Boltzmann; effective horizon.
- The limits of a Q-table (no generalisation, size), and what is not a limit.
- What is in a stochastic game and what is not; the iterated PD; why "maximise your return" fails; convergence in matching pennies.
- Non-stationarity, credit assignment, equilibrium selection, scalability: what each is and an example.
- Central against independent learning: what each learns, advantages, limitations; the climbing game.

## Exam questions

The mock's open question on this part (12, the Stag Hunt with independent Q-learners) is in [[AGAI-Lab02 - Learning Agents|Lab 2]].

> [!exam]- Two independent Q-learners play the climbing game, a fully cooperative game with shared reward (rows a, b, c for agent A, columns for agent B): (11, −30, 0 / −30, 7, 6 / 0, 0, 5). (a) Early in learning the partner plays uniformly at random. Compute the expected reward of each action. (3 points) (b) Explain why the learners usually end at (c, c) and not at (a, a). (3 points) (c) What does a central Q-learner learn, and why does it find (a, a)? (2 points) (d) Give one reason why independent learning is still widely used. (2 points)
> **Key points:** a: −19/3 ≈ −6.3, b: −17/3 ≈ −5.7, c: 5/3 ≈ 1.7; Q-values average over the partner's actions, c never risks −30; a pays only if both play a together, and failed attempts cost −30, so exploring a is punished; central learner has one Q-value per joint action, sees 11 directly; independent learning scales ($|A| \times N$ against $|A|^N$) and needs no observation of others' actions.
>
> (a) Against a uniform partner: a earns $(11 - 30 + 0)/3 \approx -6.3$, b earns $(-30 + 7 + 6)/3 \approx -5.7$, c earns $(0 + 0 + 5)/3 \approx 1.7$.
>
> (b) An independent learner's value is the reward averaged over what the partner plays, $Q_A(a) \approx \sum_b \pi_B(b) R(a,b)$, so c looks best: it never risks −30. Both agents drift to c. Once the partner plays c, exploring a earns 0 (or −30 when the partner explores b), so $Q(a)$ never recovers; (a, a) pays only if both explore a in the same round. They are stuck at the safe action (minimax), and the optimum is unreachable.
>
> (c) A central Q-learner keeps one Q-value per joint action, $Q(s, a^A, a^B)$, and chooses the joint action for both agents. The value of (a, a) is 11 whatever else happens, so it finds the optimum.
>
> (d) It scales: each agent tracks only its own actions ($|A| \times N$ values against $|A|^N$ joint actions), and it does not need to observe the other agents' actions. (Also: any single-agent algorithm can be used, and in larger environments such as level-based foraging it learned faster than central learning.)

> [!exam]- Four warehouse robots each have 5 actions and share a team reward. (a) How many joint actions must a central Q-learner explore per state, and how many action values do independent Q-learners keep per state in total? (2 points) (b) Give two limitations of central Q-learning. (3 points) (c) Give two advantages and two limitations of independent Q-learning. (4 points) (d) One robot often idles while the others work, and its idling is reinforced. Which challenge of MARL is this? (1 point)
> **Key points:** $5^4 = 625$ joint actions against $4 \times 5 = 20$ values; central: exponential joint action space, needs a central controller that observes and commands all robots (and one scalar reward); independent: scales, no need to observe others' actions; non-stationary environment, no convergence guarantee, averaging hides good joint actions (miscoordination); credit assignment (the lazy agent).
>
> (a) A central learner chooses among all joint actions: $5^4 = 625$. Independent learners each keep values for their own 5 actions: $4 \times 5 = 20$ in total.
>
> (b) The joint action space grows exponentially with the number of agents, so learning and exploring it becomes intractable. It needs a central controller that sees everything and commands every robot, which may be impossible during execution. (MARL book: it also needs the rewards combined into one scalar, easy here because the reward is shared.)
>
> (c) Advantages: each table covers only the agent's own actions, so it scales with the number of agents; agents do not need to observe each other's actions. Limitations: the other robots keep learning, so each robot's environment is non-stationary and Q-learning's convergence guarantees are lost; the value of an action is averaged over what the others currently do, which can hide good joint actions and lead to miscoordination.
>
> (d) Multi-agent credit assignment: with a shared team reward, the idle robot receives +1 when the others succeed (the lazy agent problem).

> [!exam]- (a) Write the Q-learning update, say what the target is, and give the target when $s'$ is terminal. (3 points) (b) What is the main limitation of value iteration compared with Q-learning? (2 points) (c) Give two limitations of tabular Q-learning. (2 points) (d) Why are the Q-learning convergence guarantees lost when several independent Q-learners learn at once? (3 points)
> **Key points:** $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma\max_{a'}Q(s',a') - Q(s,a)]$, target $r + \gamma\max_{a'}Q(s',a')$, terminal: $r$; value iteration needs the transition and reward functions; tables do not generalise (experience does not carry over to similar states) and grow with the number of state-action pairs; the others' learning makes the transitions and rewards each agent experiences change, the environment is non-stationary, and the proofs assume a fixed MDP.
>
> (a) $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma\max_{a'}Q(s',a') - Q(s,a)]$. The target $r + \gamma\max_{a'}Q(s',a')$ is a one-sample estimate of the return after taking $a$ and acting greedily afterwards; $\alpha$ sets how far the estimate moves towards it. If $s'$ is terminal there is no future value, so the target is $r$.
>
> (b) Value iteration needs a model of the environment, the transition function $P(s' \mid s,a)$ and the reward function, which is usually unavailable or intractable. Q-learning needs only sampled transitions.
>
> (c) Each value is learned in isolation, so experience in one state does not carry over to similar states and every state-action pair must be visited; and the table grows with the number of state-action pairs, too large for complex environments.
>
> (d) Each agent treats the others as part of its environment. The transitions it experiences, $P^i(s' \mid s, a^i) = \sum_{a^{-i}} [\prod_{j \neq i} \pi^j(a^j \mid s)] P(s' \mid s, a^i, a^{-i})$, change whenever the others' policies change, and so do its rewards. The convergence proof of Q-learning assumes a fixed (stationary) MDP, so it no longer applies: best responses can cycle, and targets go stale.

## Flashcards

> [!card]- When is a state Markov?
> When it summarises everything about the past that matters for the future.

> [!card]- Why is reinforcement learning data not i.i.d.?
> It depends on how the agent acts: consecutive samples are correlated.

> [!card]- What does value iteration need that Q-learning does not?
> A model: the transition function and the reward function.

> [!card]- How does Q-learning do without a model?
> It replaces the sum over next states with one observed transition.

> [!card]- What is the Q-learning target?
> $r + \gamma \max_{a'} Q(s',a')$.

> [!card]- What is the Q-learning target when the next state is terminal?
> Just $r$: there is no future value.

> [!card]- What is the effective horizon of a discount factor $\gamma$?
> $1/(1-\gamma)$ steps.

> [!card]- What is the main reason a discount factor is needed?
> Infinite-horizon tasks: it keeps the return finite.

> [!card]- What does a high learning rate do in a stochastic environment?
> Estimates stay noisy and never settle; the fix is to decay it.

> [!card]- Why is ε-greedy exploration called blind?
> A disastrous action is tried as often as a nearly-best one.

> [!card]- What is the advantage of Boltzmann exploration over ε-greedy?
> It is value-aware: better actions are tried more often.

> [!card]- Why does experience in one state not help a Q-table in a similar state?
> Each value is updated in isolation: tables do not generalise.

> [!card]- What is a stochastic game?
> The multi-agent generalisation of an MDP: simultaneous actions, a reward function per agent.

> [!card]- In a stochastic game, what do the transitions depend on?
> The state and the joint action of all agents.

> [!card]- In the iterated Prisoner's Dilemma as a stochastic game, what is the state?
> Both players' actions in the last round: CC, CD, DC or DD.

> [!card]- Why is "maximise your return" not enough as a solution concept in MARL?
> My best policy depends on yours, and yours on mine.

> [!card]- In which senses do cycling learners in matching pennies converge?
> In average policy and average reward; their exact policies never settle.

> [!card]- Why is the environment non-stationary for a learner in MARL?
> The other agents are part of it, and their policies keep changing.

> [!card]- How does agent credit assignment differ from temporal credit assignment?
> Temporal: which earlier action caused the reward. Agent: whose action did.

> [!card]- What is it called when an idle agent is reinforced because its teammates earned the shared reward?
> The lazy agent problem, a case of multi-agent credit assignment.

> [!card]- Why is credit assignment worst in fully cooperative games?
> A shared team reward gives no individual signal at all.

> [!card]- Why do learners tend to reach "safe" equilibria?
> Costly exploration mistakes dissuade them from the risky, better joint action.

> [!card]- Which MARL challenge decides where learners end up, as opposed to how well they learn?
> Equilibrium selection.

> [!card]- What does central Q-learning learn?
> One Q-function over the state and the joint action, which controls all agents.

> [!card]- Why does central Q-learning scale badly?
> It must explore $|A|^N$ joint actions, exponential in the number of agents.

> [!card]- What does an independent learner's $Q_A(a)$ estimate in a one-state game?
> The reward of $a$ averaged over the partner's current actions: $\sum_b \pi_B(b) R(a,b)$.

> [!card]- Why do independent Q-learners end at (c, c) = 5 in the climbing game?
> Averaged over the partner's exploration, c never risks −30, so it looks best.

> [!card]- True or false: independent Q-learning keeps Q-learning's convergence guarantee.
> **False.** The other learners make each agent's environment non-stationary.

> [!card]- True or false: independent learners need to observe the other agents' actions.
> **False.** Each learns from its own actions and rewards only.

> [!card]- True or false: tabular Q-learning needs the transition function in advance.
> **False.** That is value iteration; Q-learning learns from sampled transitions.

> [!card]- True or false: the definition of a stochastic game includes an observation function for each agent.
> **False.** The state is shared; observation functions belong to the partially observable version.

> [!card]- Multiple selection: which are limitations of tabular Q-learning? (A) It works only with deterministic rewards. (B) Experience in one state does not carry over to similar states. (C) It needs the transition function in advance. (D) The table grows with the number of state-action pairs.
> **B and D.** A: the learning rate averages stochastic rewards. C: that is value iteration.

> [!card]- Multiple selection: which are advantages of independent learning? (A) Each agent's environment stays stationary. (B) Each Q-table covers only the agent's own actions. (C) Agents need not observe the others' actions. (D) Convergence to a Nash equilibrium is guaranteed.
> **B and C.** A: the others' learning makes it non-stationary. D: there are no such guarantees.

> [!card]- Multiple selection: which are part of a stochastic game? (A) A reward function for each agent. (B) A joint policy. (C) A transition function over the joint action. (D) An action set for each agent.
> **A, C and D.** B: policies are what the agents choose; the game does not include them.

## Links

- **Course:** [[AGAI - Overview|Course overview]]
- **Lab:** [[AGAI-Lab02 - Learning Agents]] (independent Q-learners in the Stag Hunt and the Coin Game; feeds mock questions 10, 12 and 21)
- **Next lecture:** [[AGAI-L04 - Deep RL and Intro to Deep MARL]] (why tables stop working, deep RL)
- **Game theory background:** [[AGAI-L02 - Game Theory and Population Dynamics]] (stochastic games, Nash equilibrium, Stag Hunt)
- **RL background in depth:** [[RL-L01 - Intro, MDPs & Bandits]], [[RL-L02 - Dynamic Programming]], [[RL-L04 - Temporal Difference Learning]]
- **Concepts:** [[Markov Decision Process]], [[Bellman Equation]], [[Value Iteration]], [[Q-Learning]], [[Epsilon-Greedy Policy]], [[Tabular RL]], [[Stochastic Game]], [[Independent Learning]], [[Multi-Agent Credit Assignment]], [[Nash Equilibrium]]
- **Source:** Agentic Artificial Intelligence, Lecture 3, "Intro to Multi-Agent Reinforcement Learning", Roman Chiva Gil, MSc AI, UvA, 3 November 2026. Midterm reading: Albrecht, Christianos and Schäfer, *Multi-Agent Reinforcement Learning* (MIT Press, 2024), Chapters 2 and 5.
