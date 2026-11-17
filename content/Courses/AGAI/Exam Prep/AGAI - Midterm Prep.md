---
type: exam-prep
course: 5204AGAI6Y
status: complete
---

# AGAI - Midterm Prep

Fri 20 Nov 2026, 13:00-15:00, USC Universum Sporthal 1. Closed book. 25 multiple choice (2 points each) and 5 open questions (10 points each). Pass at 55 points.

What the mock shows:

- **Multiple choice** asks definitions, "which are limitations / advantages of X", which model or training mode fits a described scenario, and which formula is the right one among near-copies. Round boxes: one answer. Square boxes: possibly several.
- **Open questions** are the lab exercises in miniature. The only arithmetic is tiny (expected payoff against a 50/50 partner, a threshold, counting table rows).
- One open question per topic plus one on Assignment 1. Each topic below ends with the shape of its open question and an answer that would get the points.

Full notes per topic: [[AGAI-L02 - Game Theory and Population Dynamics|Lecture 2]], [[AGAI-Lab01 - Simple Agents, Complex Dynamics|Lab 1]], [[AGAI-L03 - Intro to Multi-Agent Reinforcement Learning|Lecture 3]], [[AGAI-Lab02 - Learning Agents|Lab 2]], [[AGAI-L04 - Deep RL and Intro to Deep MARL|Lecture 4]], [[AGAI-L05 - Deep MARL, Part II|Lecture 5]], [[AGAI-L06a - From Deep MARL to Reasoning|Lecture 6a]].

## Game theory (5 MC, 1 open)

**The models, smallest first.** Each one contains the previous as a special case.

- **Normal-form game**: agents $I$, actions $A_i$, rewards $R_i$ on the joint action. One shot, simultaneous. "Two agents act once, payoffs depend only on the pair of actions" is this one.
- **Repeated normal-form game**: the same game many times. Policies can condition on the history of joint actions.
- **Stochastic game**: adds states $S$ and a transition function $T(s' \mid s, a)$ on the joint action $a$, plus a reward $R_i(s, a, s')$ per agent and an initial state distribution. Everyone sees the state.
- **POSG** (partially observable stochastic game): adds, per agent, observations $O_i$ and an observation function. Policies condition on the agent's own observation history.

Trap: a joint policy is never part of a game's definition, and observation functions belong only to the POSG.

**Payoff structures.** Zero-sum (one's gain is the other's loss), common-reward (everyone gets the same), general-sum (anything else: Prisoner's Dilemma, Stag Hunt).

**Solution concepts.**

- **Best response**: the policy that maximises my expected return against the others' fixed policies.
- **Nash equilibrium**: everyone best-responds at once, so nobody gains by deviating alone. Every finite game has one once mixed strategies are allowed.
- **Finding pure equilibria**: in each column mark the row player's best payoff, in each row the column player's best. A cell marked twice is a pure Nash equilibrium.
- **Mixed equilibrium in a 2x2 game**: find the partner's probability $q$ of playing C that makes C and D earn the same. Stag Hunt (4, 0, 3, 2): $4q = 3q + 2(1-q)$, so $q = 2/3$.
- **Limitations of Nash**: it can be worse for everyone than another outcome (Prisoner's Dilemma); a game can have several equilibria and Nash does not say which is played (Stag Hunt); it says nothing about how agents get there or what they do after a deviation; computing one can be hard in general games. False options to reject: "some finite games have none" (mixed strategies fix that) and "only two players".
- **Pareto optimal**: no other outcome makes someone better off without making someone worse off. It says nothing about fairness.
- **Welfare** = sum of returns; the **social optimum** maximises it. **Fairness** = product of returns, which favours equal splits.
- **Social dilemma**: self-interest leads to an outcome worse for the group. Prisoner's Dilemma is a dilemma of cooperation (the only equilibrium, (D, D), is the only outcome that is not Pareto optimal). Stag Hunt is a dilemma of coordination or trust ((C, C) and (D, D) are both equilibria).

**Repeated games.** Tit-for-Tat: cooperate first, then copy the opponent's last move. Win-Stay Lose-Shift: keep your move if it paid well, switch otherwise. In short games always-defect wins (it exploits once); in long games reciprocators pull ahead.

**Populations (evolutionary game theory).**

- Replicator equation: $\dot x = x\,(f_C - \bar f)$. With two strategies it becomes $\dot x = x(1-x)(f_C - f_D)$, with $f_C = xR + (1-x)S$ and $f_D = xT + (1-x)P$.
- What it assumes: **C becomes more common when it earns more than the population average.** It is not a best response, not "majority wins", not Pareto. The same equation describes natural selection, imitation and reinforcement learning on average.
- Outcomes: Prisoner's Dilemma goes to all D from any start. Stag Hunt tips at $x = 2/3$: above it all C, below it all D. Hawk-Dove settles in a stable mix of both.

**Open question (mock Q6), Prisoner's Dilemma with $T > R > P > S$:**

- (a) Against C, D earns $T > R$. Against D, D earns $P > S$. So D is the best response to both (dominant).
- (b) In any cell with a C, that player gains by switching to D, so (D, D) is the only pure equilibrium. Not Pareto optimal: (C, C) gives both $R > P$.
- (c) (C, D) pays $(S, T)$. The column player already has its maximum $T$, so helping the row player costs the column player: Pareto optimal. Yet the row player gets its worst payoff, so Pareto optimality ignores fairness.
- (d) (C, C) is the social optimum when $2R \ge T + S$.

Likely sibling: the same four parts for the Stag Hunt. Two pure equilibria (C, C) and (D, D), a mixed one at 2/3; only (C, C) is Pareto optimal and the social optimum; (D, D) is the safe, risk-dominant one.

## Multi-agent RL (5 MC, 1 open)

**RL basics.**

- MDP: states, actions, transitions $P$, rewards $R$, discount $\gamma$. Return $G_t = \sum_k \gamma^k r_{t+k+1}$.
- $Q(s, a)$: expected return after taking $a$ in $s$. Bellman: $Q(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q(s', a')]$.
- **Value iteration needs the model** (transition and reward functions). Q-learning learns from sampled experience without one.
- **Q-learning update**: $Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,]$. Off-policy. At a terminal state the target is just $r$.
- $\gamma$: how much the future counts ($\gamma = 0$ only cares about the next reward). $\alpha$: step size. $\varepsilon$-greedy: random action with probability $\varepsilon$.
- **Limits of tabular Q-learning**: the table grows with the number of state-action pairs; experience in one state does not carry over to similar states; rarely visited states stay unlearned. It does not need the transition function and handles stochastic rewards fine.

**MARL.**

- A stochastic game is the multi-agent MDP: my return depends on everyone's policies, but I control only mine. So "maximise your return" is replaced by equilibrium, welfare, no regret or robustness.
- Convergence is slippery: in matching pennies learners cycle around the mixed equilibrium, though their average policy matches it.
- **Non-stationarity**: the others learn, so my environment keeps changing; old experience goes stale and single-agent convergence guarantees break.
- **Credit assignment**: with a shared reward, whose action earned it? A lazy agent gets reinforced because a teammate succeeded.
- **Equilibrium selection**: even perfect learners can settle in a bad equilibrium; learners drift to safe ones because exploration mistakes are costly.
- **Scalability**: joint state and action spaces grow exponentially with the number of agents.

**Central against independent learning.**

- **Central Q-learning**: one Q-function over the state and the **joint action**, which controls all agents. Joint action space explodes, and it needs one scalar reward and central control.
- **Independent Q-learning**: each agent learns its own Q over its own actions and treats the others as part of the environment. Advantages: each table covers only its own actions, so it scales; agents do not need to observe each other's actions. Disadvantages: non-stationary, no convergence guarantee, can miscoordinate.

**Coin Game** (Lab 2): +1 for any coin, and the coin's owner gets −2 if the other agent takes it. Independent learners collect many coins but only about 56% of their own colour, because the −2 lands on the other agent and an agent's own reward ignores it.

**Open question (mock Q12), Stag Hunt (R 4, S 0, T 3, P 2) with independent Q-learners:**

- (a) Against a 50/50 partner: C earns $0.5 \cdot 4 + 0.5 \cdot 0 = 2$, D earns $0.5 \cdot 3 + 0.5 \cdot 2 = 2.5$.
- (b) D always earns at least 2; C pays only if the partner plays C at that same moment and earns 0 otherwise. Once both play D, $Q(D)$ goes to 2 and $Q(C)$ only gets updated during exploration, where it mostly earns 0. (D, D) is the risk-dominant equilibrium.
- (c) Leaving (C, C) takes only a few exploratory D moves by one agent: the partner's C then earns 0 and its $Q(C)$ drops below $Q(D)$. Leaving (D, D) needs $Q(C)$ to rise above $Q(D)$, which needs both agents to explore C in the same rounds. Far less likely.
- (d) Plays C first when $4b > 3b + 2(1-b)$, so $b > 2/3$. The $b = 0.9$ agent plays C, the $b = 0.5$ agent plays D. The first earns 0 and its $Q(C)$ falls fast; the second earns 3 and keeps D. They end at (D, D).

## Deep RL (7 MC, 1 open)

**Why tables fail.** Memory grows with the state-action pairs; many states are visited rarely or never; nothing learned in one row is used in another. Adding an irrelevant random binary input doubles the rows, and the two copies share nothing.

**Function approximation**: a network computes values or a policy from the observation with shared weights, so it generalises to similar states. Costs: training instability (moving targets, correlated data), possible divergence, no general convergence guarantee, more computation.

**DQN.**

- Input: the state. Output: one Q-value per discrete action.
- Loss $(y - Q(s,a;\theta))^2$ with target $y = r + \gamma \max_{a'} Q(s',a';\theta^-)$. If $s'$ is terminal, $y = r$. No gradient flows through $y$.
- Naive deep Q-learning has two problems: **moving targets** (the target depends on the weights being trained) and **correlated data** (consecutive steps look alike).
- **Target network** $\theta^-$: a copy updated only every so often, so the target holds still.
- **Replay buffer**: store transitions and train on random minibatches. It reduces correlation in a minibatch and lets each transition be used for several updates. It does not store weights, does not make DQN on-policy and does not replace the target network.
- DQN is off-policy, usually collects data $\varepsilon$-greedily, and needs discrete actions.

**Policy gradients** learn $\pi(a \mid s; \phi)$ directly: continuous actions and stochastic policies are natural.

- **REINFORCE**: the gradient estimate is $G_t \, \nabla_\phi \log \pi(a_t \mid s_t; \phi)$ with $G_t$ the Monte Carlo return. Unbiased, high variance, waits for the episode to end, on-policy.
- **Baseline**: subtract $V(s)$ from the return: lower variance, same expected gradient.
- **A2C** (actor-critic): the actor is the policy, the critic learns $V$. Advantage $A(s,a) = r + \gamma V(s') - V(s)$. If $A > 0$, action $a$ did better than expected and the actor makes it more likely. Lower variance and shorter rollouts than REINFORCE; the critic adds bias and hyperparameters.
- **PPO**: actor-critic that clips the ratio $\pi_\text{new}/\pi_\text{old}$ to $[1-\epsilon, 1+\epsilon]$, so one update cannot move the policy far. Stable, reuses each batch for several passes, still on-policy, and the clip is not a strict guarantee.

**Open question (mock Q21), Coin Game tables (1,458 rows, $k$ random lamps, 300 updates x 32 games x 16 steps):**

- (a) $N = 1{,}458$ for $k = 0$, $N = 1{,}458 \cdot 2^8$ for $k = 8$. Budget $B = 300 \cdot 32 \cdot 16$ decisions per agent.
- (b) The rows multiply but the budget stays the same, so with 8 lamps most rows are never visited and the rest only a few times. An unvisited row still holds zeros, so the agent acts randomly there. Rows that differ in one lamp share nothing, so every lesson has to be relearned for every lamp pattern. That is why it cannot discover the lamps do not matter.
- (c) A network computes values or a policy from the observation with shared weights: what it learns about the coin's position applies to all lamp patterns, and it can give the lamps little weight. New challenge: instability (moving targets, correlated data), possible divergence, no convergence guarantee, more computation.

## Deep MARL (8 MC, 1 open)

**Training and execution modes.**

- **CTCE** (centralised training and execution): one learner over the joint action. Good coordination; the joint action space grows exponentially, and distributed agents may not be able to share information.
- **DTDE** (decentralised training and execution): independent learners. Scales and suits agents that cannot communicate; suffers non-stationarity and bad equilibria.
- **CTDE** (centralised training with decentralised execution): extra information (global state, others' observations and actions) only during training; at execution each agent acts on local information. Fits "a simulator gives all data in training, the cars cannot share sensor data while driving". Typical form: a centralised critic in training, decentralised actors at execution.

**Independent deep learners.**

- **IDQN**: the replay buffer holds experience generated by the other agents' old policies, so it goes out of date. Fixes: a smaller buffer, or importance-sampling weights using the stored action probabilities of all agents. A larger buffer makes it worse.
- **Independent REINFORCE, A2C, PPO** are on-policy: they learn from recent experience generated by the others' current policies, so they adapt faster to learning partners.

**Centralised value functions (CTDE, shared reward).**

- **Value decomposition**: learn one utility per agent from its own observations, so that each agent's greedy action together gives the greedy joint action (the IGM property: individual-global-max).
- **VDN**: team value = sum of the utilities.
- **QMIX**: team value = a monotonic mix of the utilities (more utility never lowers the team value), weights produced from the global state. It fixes VDN's restrictive sum and captures interactions a sum cannot. (MARL book, not in the lecture: a monotonic mix still cannot represent a game where an agent's best action flips with the others'.)

**Agent modelling.** A deep agent model of agent $i$ predicts agent $j$'s actions from agent $i$'s own observation history; agent $i$ uses that prediction in its policy or value. **LOLA** goes further: it accounts for how its own update changes the other agent's next learning step.

**Sharing between similar agents.**

- **Weakly homogeneous**: swapping the agents' policies does not change the returns. **Strongly homogeneous**: the optimal joint policy consists of identical policies.
- **Parameter sharing**: one network for all agents, so all follow the same policy. The parameter count stays constant as agents are added, and the shared parameters learn from all agents' experience (sample efficiency). Needs strong homogeneity (or an agent index as input). It does not need agents to exchange observations at execution, and it works badly when the optimal policies differ.
- **Experience sharing**: each agent keeps its own network but also learns from all agents' trajectories (better sample efficiency), so policies can differ. Suits weakly homogeneous agents; on-policy methods need a correction (importance sampling).

**Search, self-play, populations.**

- **MCTS**: selection, expansion, simulation, backpropagation.
- **Self-play**: train against copies of yourself; needs symmetric roles. AlphaGo: a policy network (from expert moves, then RL) proposes moves and a value network evaluates new positions, both guiding MCTS. AlphaGo Zero drops the human games and learns from scratch; AlphaZero runs the same method on chess, shogi and Go.
- Plain self-play can cycle (rock-paper-scissors), hence **populations of policies**.
- **PSRO**: keep a population of policies per agent. The **meta-game is a normal-form game whose actions are the policies in the current populations**, with payoffs from simulated matches. A meta-solver (for example Nash) gives a mix over the population; an oracle trains a best response to that mix and adds it; repeat. AlphaStar's league is PSRO-like.
- RL and LLMs: AlphaGo has a fast System 1 (policy network, intuition) and a slow System 2 (search over rollouts); an LLM generating tokens is still System 1. LLM generation as RL: state = prompt and tokens so far, action = next token, policy = the LLM, reward when the full response is good.

**Open question (mock Q29), landmarks.** A: both agents must reach the same landmark. B: two different landmarks. Same actions and observations, swapping policies does not change the return.

- (a) Definitions above. A is strongly homogeneous, B is weakly homogeneous.
- (b) Parameter count stays constant as agents are added; the shared parameters learn from all agents' more diverse experience, which improves sample efficiency.
- (c) A. Its optimal policies are identical; identical policies in B could not reliably send the agents to different landmarks without knowing which agent is which.
- (d) Experience sharing: own networks, learning from each other's trajectories.

## Assignment 1 (1 open)

Mock Q30: (a) does your agent do particularly well against a specific Red agent (5 points), (b) which learning algorithm did you implement (5 points). Graded on whether the answer reflects what your group actually did. Name the algorithm and why you chose it, the Red agents it does well and badly against with one piece of evidence, and the reason for the difference.
