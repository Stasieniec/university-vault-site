---
type: lecture
course: 5204AGAI6Y
week: 2
lecture: 4
date: 2026-11-05
book_sections: ["Albrecht, Christianos and Schäfer, MARL, Ch 8", "Ch 9.1", "Ch 9.3"]
topics:
  - "[[Deep Reinforcement Learning]]"
  - "[[Function Approximation]]"
  - "[[Linear Function Approximation]]"
  - "[[Neural Network Function Approximation]]"
  - "[[Neural Networks]]"
  - "[[Convolutional Neural Networks]]"
  - "[[Recurrent Neural Network (RNN)]]"
  - "[[Deep Q-Network (DQN)]]"
  - "[[Target Network]]"
  - "[[Experience Replay]]"
  - "[[Off-Policy Learning]]"
  - "[[On-Policy Learning]]"
  - "[[Policy Gradient Methods]]"
  - "[[Policy Gradient Theorem]]"
  - "[[REINFORCE]]"
  - "[[Baseline]]"
  - "[[Actor-Critic]]"
  - "[[Advantage Function]]"
  - "[[Advantage Actor-Critic (A2C)]]"
  - "[[PPO]]"
  - "[[Centralised Training with Decentralised Execution]]"
  - "[[Independent Learning]]"
  - "[[Independent DQN]]"
status: complete
---

# AGAI-L04: Deep RL and Intro to Deep MARL

> [!abstract] In short
> Tables cannot generalise: every value is learned in isolation, so in large state spaces most rows go unvisited. Deep RL replaces the table with a neural network. DQN does this for Q-learning, adding a target network against moving targets and a replay buffer against correlated data. Value-based agents only express ε-greedy policies over discrete actions, so policy gradient methods learn the policy itself: REINFORCE weights log-probabilities by Monte Carlo returns (unbiased, high variance), A2C by a critic's advantage (lower variance, some bias), and PPO clips each update. With several agents, the design choice is what information each agent may use in training and in execution; independent deep learners are the decentralised baseline, and independent DQN's replay buffer goes stale as the others change.
>
> Feeds mock midterm questions 13 to 21 (the Deep RL section, including the open question 21 on the Coin Game) and the Deep MARL questions 22, 23 and 26 (stale replay in IDQN, choosing a training mode, independent REINFORCE against IDQN).

The MARL book writes time as a superscript ($s^t$, $a^t$, $r^t$) and the agent as a subscript ($Q_i$, $\pi_i$); this note follows it.

## Why tables stop working

[[AGAI-L03 - Intro to Multi-Agent Reinforcement Learning|Lecture 3]] stored values in a table: one row per state, one column per action, updated by

$$Q_i(s^t,a^t) \leftarrow Q_i(s^t,a^t) + \alpha\Big(\underbrace{r^t + \gamma\max_{a'}Q_i(s^{t+1},a')}_{\text{target}} - \underbrace{Q_i(s^t,a^t)}_{\text{estimate}}\Big)$$

That breaks down in complex environments for two reasons.

**It cannot generalise.** Each value is updated in isolation: learning about one row teaches nothing about any other, even a nearly identical one. So the agent has to visit every state-action pair, many times, to get accurate values. In a maze where the agent has walked through cell $s_1$ but never through the neighbouring cell $s_2$, both about equally far from the goal, the table treats $s_2$ as completely unknown, though its value is obviously close to that of $s_1$.

**It needs too much space.** The table grows linearly with the number of possible inputs, which itself grows exponentially with the number of state variables.

> [!example] Irrelevant lamps in the Coin Game ([[AGAI-Lab02 - Learning Agents|Lab 2]])
> Two tabular Q-learners collect coins on a 3×3 grid; the state (own cell, other's cell, coin cell, whose coin) has $9 \cdot 9 \cdot 9 \cdot 2 = 1{,}458$ rows. Add $k$ random lamps that switch on or off and have no effect on the game. Each lamp pattern is a different state to a table, so each lamp **doubles the rows**: $1{,}458 \cdot 2^k$, or $373{,}248$ with 8 lamps, and rows that differ only in a lamp share nothing. With a fixed budget, experience per row falls by $2^k$: with 8 lamps most rows are never visited, an unvisited row still holds zeros, and the agent acts at random there. The best policy ignores the lamps, but the table cannot notice that two rows call for the same action.

## Function approximation: learn a function instead of a table

The fix is [[Function Approximation|function approximation]]: learn a parameterised function $f(x;\theta)$ of some input $x$, and train the parameters $\theta$ so that $f$ approximates the optimal function $f^*$ well. Because all inputs share the same parameters, an update for one input also changes the output for similar inputs, which is exactly the generalisation the table lacked. $f$ can be a [[Value Function|value function]] (input a state, output the expected return) or a [[Policy|policy]] (input a state, output an action or a distribution over actions).

**Linear functions** compute $\hat v(s;\theta) = \theta^\top \mathbf{x}(s)$ from a feature vector $\mathbf{x}(s)$ ([[Linear Function Approximation]]). They are simple, generalise, and resist overfitting, but have **high bias**: they are not expressive enough for complex value functions.

**Deep neural networks** ([[Neural Network Function Approximation]], [[Neural Networks]]) are a universal method: they **learn feature representations of states automatically**, represent non-linear, complex functions, and are effective with high-dimensional data such as images and sequences. A **feedforward network** stacks layers, each a linear map followed by a non-linear activation $g$ (for example ReLU):

$$f(\mathbf{x};\theta) = f_3\big(f_2(f_1(\mathbf{x};\theta^1);\theta^2);\theta^3\big), \qquad f_k(\mathbf{z};\theta^k) = g_k(W_k\mathbf{z} + \mathbf{b}_k)$$

with $\theta^k = (W_k, \mathbf{b}_k)$ the weights and biases of layer $k$. It is trained by [[Gradient Descent|gradient descent]] on a loss, with gradients computed by backpropagation.

Two specialised architectures matter for agents:

- A [[Convolutional Neural Networks|convolutional neural network]] (CNN) moves small filters (say 3×3) over an image; the same filter weights are used at every position, so it detects a pattern wherever it appears. It suits image and video observations.
- A [[Recurrent Neural Network (RNN)|recurrent neural network]] (RNN) reads a sequence one element at a time and carries a hidden state, $h^t = f(x^t, h^{t-1})$, so its output can depend on the whole history. It suits a sequence of observations, which matters when one observation does not reveal the full state ([[Partial Observability]]).

With them an agent's mapping from observations to actions can take images, video or observation histories, and handle large and continuous action spaces: [[Deep Reinforcement Learning|deep reinforcement learning]].

## Deep Q-learning: Q-learning with a network

Replace the table $Q_i(s^t,a^t)$ by a network $Q_i(s^t,a^t;\theta)$. The standard design takes **the state as input and outputs one Q-value per discrete action**, $Q(s,a_1;\theta), \dots, Q(s,a_{|A|};\theta)$, so one forward pass gives every action's value and the greedy action is the largest output.

The tabular update moved the estimate towards the target. With a network that becomes a regression problem: minimise the squared difference between target and estimate,

> [!formula] Deep Q-learning loss
> $$\mathcal{L}(\theta) = \big(y^t - Q(s^t,a^t;\theta)\big)^2, \qquad y^t = \begin{cases} r^t & \text{if } s^{t+1} \text{ is terminal}\\ r^t + \gamma\max_{a'}Q(s^{t+1},a';\theta) & \text{otherwise}\end{cases}$$
> - $y^t$: the target, the same one-step estimate as in Q-learning; a terminal next state has no future value, so the target is just the reward
> - $Q(s^t,a^t;\theta)$: the network's current estimate for the action actually taken
>
> $\theta$ is updated by gradient descent on $\mathcal{L}$. The target is treated as a fixed number: **no gradient flows through $y^t$**, only through the estimate (standard practice, called a [[Semi-Gradient Methods|semi-gradient]] update). The network is pulled towards the target, never the other way round.

> [!example] One deep Q-learning step
> The network outputs $Q(s,a;\theta) = 2.0$ for the action taken; the agent gets $r = 1$ and the network gives $\max_{a'}Q(s',a';\theta) = 3.0$. With $\gamma = 0.9$: $y = 1 + 0.9 \times 3.0 = 3.7$, loss $(3.7 - 2.0)^2 = 2.89$. The gradient step raises $Q(s,a;\theta)$ towards 3.7, and because the weights are shared it also shifts the values of other states, including $Q(s',\cdot\,;\theta)$, the very values the target was built from.

The training loop is Lecture 3's Q-learning loop (ε-greedy action, act, observe, compute the target) with the table update replaced by one gradient step on this loss.

## What breaks: moving targets and correlated data

This naive version is unstable, for two reasons that both come from replacing the table.

**The moving target problem.** The target $y^t$ is computed with the same $\theta$ that is being updated. Because the network generalises, any update to $\theta$ changes many values at once, the targets included. In a table one entry changes per update, when its pair is visited; a network moves its targets after every step, so learning chases a target that keeps running away and can oscillate or diverge.

**Correlated consecutive experience.** Training on transitions in the order they happen gives data that is far from i.i.d. (independent and identically distributed): consecutive states are almost the same, and a stretch of an episode covers one small region of the state space. Gradient descent assumes roughly i.i.d. samples. With correlated data the network **overfits to recent experience** and suffers **catastrophic forgetting** of earlier estimates: after a hundred steps in one corridor it has overwritten what it knew about the rest of the maze.

Compared with the table, the network also loses the general convergence guarantee (training can diverge) and costs more computation per decision.

## DQN: a target network and a replay buffer

**[[Target Network|Target network]].** Keep a second copy of the network, with parameters $\bar\theta$, and compute targets with it:

$$y^t = r^t + \gamma\max_{a'}Q(s^{t+1},a';\bar\theta)$$

The copy is **updated less often**: every so many steps, $\bar\theta \leftarrow \theta$. Between copies the targets stay put, so the main network regresses towards a fixed goal for a while.

**[[Experience Replay|Replay buffer]].** Store every transition $(s^t,a^t,r^t,s^{t+1})$ in a buffer $\mathcal{D}$, and compute each loss on a **minibatch sampled uniformly** from it instead of the latest transition. A random minibatch mixes experience from many episodes and regions, so its samples are much less correlated. Extra benefits: each transition is **re-used** for several updates (better data efficiency), and averaging the loss over a batch gives **more stable gradients**.

> [!definition] DQN
> **Deep Q-networks (DQN)** (Mnih et al., Nature 2015) is the foundational deep RL algorithm based on Q-learning: **Q-learning + target network + replay buffer**. See [[Deep Q-Network (DQN)]].

```pseudo
Algorithm: Deep Q-Networks (DQN)
────────────────────────────────
Initialize value network Q with random parameters θ
Initialize target network with parameters θ̄ = θ                 // target network
Initialize an empty replay buffer D = {}                         // replay buffer
For every episode:
  For time step t = 0, 1, 2, ...:
    Observe current state s^t
    With probability ε: choose random action a^t ∈ A
    Otherwise:          choose a^t ∈ argmax_a Q(s^t, a; θ)
    Apply a^t; observe reward r^t and next state s^{t+1}
    Store transition (s^t, a^t, r^t, s^{t+1}) in D
    Sample random minibatch of B transitions (s^k, a^k, r^k, s^{k+1}) from D
    For each sampled k:
      If s^{k+1} is terminal: y^k ← r^k
      Else:                   y^k ← r^k + γ · max_a' Q(s^{k+1}, a'; θ̄)   // uses θ̄
    Loss L(θ) ← (1/B) Σ_{k=1..B} (y^k − Q(s^k, a^k; θ))²
    Update θ by minimising L(θ)
    In a set interval: θ̄ ← θ                                     // refresh target copy
```

**Why replay is allowed: DQN is off-policy.** The buffer holds transitions chosen by older, ε-greedy versions of the agent. That is fine for Q-learning: its target uses $\max_{a'}$, so it estimates the greedy policy's values whatever policy chose the stored actions ([[Off-Policy Learning]]; standard knowledge). The two fixes are complementary: the buffer does not remove the need for a target network, and the target network does not decorrelate the data.

DQN learned Atari games from screen pixels and the score: a CNN reads the frames, fully connected layers follow, and there is one output per joystick action.

## What a value-based agent cannot express

DQN's policy is derived from its values, and that mapping is restrictive.

**Limited policies.** An ε-greedy policy over three actions gives the greedy action $1 - \varepsilon + \varepsilon/3$ and every other action the same $\varepsilon/3$: $(1, 0, 0)$ at $\varepsilon = 0$, about $(0.93, 0.03, 0.03)$ at $0.1$, $(0.67, 0.17, 0.17)$ at $0.5$, uniform at $1$. A probabilistic policy can be anything, for example $(0.4, 0.1, 0.5)$, which favours two actions, or $(0.1, 0, 0.9)$, which never plays one. ε-greedy policies cannot represent these. In multi-agent settings this matters, because a good policy is often mixed: in matching pennies (Lecture 3) the equilibrium plays each side with probability 0.5, and any deterministic policy is exploited.

**Continuous actions.** DQN has one output per discrete action. A steering angle or a motor torque has infinitely many values, and computing $\max_a Q(s,a)$ over them is itself an optimisation problem at every step. It is much easier to learn a parameterised policy that **outputs a continuous value** directly.

So far networks approximated value functions. The alternative is to use a network to represent the policy directly: [[Policy Gradient Methods|policy gradient methods]].

## Policy gradients: learn the policy directly

A policy network with parameters $\phi$ takes the state and outputs one **preference** (logit) $l(a \mid s;\phi)$ per action; a [[Softmax Policy|softmax]] turns preferences into probabilities:

$$\pi(a \mid s;\phi) = \frac{e^{\,l(a \mid s;\phi)/\tau}}{\sum_{a'}e^{\,l(a' \mid s;\phi)/\tau}}$$

with temperature $\tau$. Any distribution over actions is now reachable, and for continuous actions the network can output, for example, the mean of a distribution instead.

The question policy gradients answer: **how should I change $\phi$ so that actions leading to better outcomes become more likely?** The performance of a policy is its value averaged over the states it visits:

$$J(\phi) = \sum_s d^{\pi_\phi}(s)\,V^{\pi_\phi}(s)$$

where $d^{\pi_\phi}(s)$ is the **visitation probability**, how likely the policy is to be in state $s$. To climb $\nabla_\phi J(\phi)$ we would seem to need $\nabla_\phi d^{\pi_\phi}(s)$, and that is hard: changing $\phi$ changes the action probabilities at every step, early actions change which states come later, the effects compound along trajectories, and the exact change **requires the environment's transition model**.

> [!formula] Policy gradient theorem
> $$\nabla_\phi J(\phi) \propto \sum_s d^{\pi_\phi}(s)\sum_a \pi(a \mid s;\phi)\Big[Q^\pi(s,a)\,\nabla_\phi\log\pi(a \mid s;\phi)\Big] = \mathbb{E}_{s \sim P(\cdot \mid \pi),\, a \sim \pi(\cdot \mid s;\phi)}\Big[Q^\pi(s,a)\,\nabla_\phi\log\pi(a \mid s;\phi)\Big]$$
> - $\nabla_\phi\log\pi(a \mid s;\phi)$: the direction in parameter space that makes $a$ more likely in $s$
> - $Q^\pi(s,a)$: how good $a$ is, the weight on that direction
> - the expectation runs over states and actions met while following $\pi$
>
> There is **no $\nabla_\phi d^{\pi_\phi}(s)$** in it, so no model is needed: the expectation can be estimated from sampled experience. $Q^\pi(s,a)$ still has to be estimated. See [[Policy Gradient Theorem]].

Intuition: push up the probability of each action you took, in proportion to how good it turned out to be. The log matters because $\nabla\log\pi = \nabla\pi/\pi$: actions the policy already takes often are sampled often, and dividing by $\pi$ stops them from being pushed up just for being frequent.

There are two ways to estimate $Q^\pi(s,a)$: with **Monte Carlo** returns (REINFORCE), or with **another network** (actor-critic methods: A2C, PPO).

## REINFORCE: estimate Q with the return

[[REINFORCE]] plays a whole episode with the current policy, then uses the discounted return from each step onwards as its estimate of $Q^\pi(s^t,a^t)$ ([[Monte Carlo Methods]]).

```pseudo
Algorithm: REINFORCE
────────────────────
Initialize policy network π with random parameters φ
For every episode:
  For time step t = 0, 1, ..., T−1:
    Observe current state s^t
    Sample action a^t ~ π(· | s^t; φ)             // exploration comes from sampling
    Apply a^t; observe reward r^t and next state s^{t+1}
  // after the episode: return from each step stands in for Q^π(s^t, a^t)
  Loss L(φ) ← −(1/T) Σ_{t=0..T−1} ( Σ_{τ=t..T−1} γ^{τ−t} r^τ ) · log π(a^t | s^t; φ)
  Update φ by minimising L(φ)                     // gradient ascent on J
```

The minus sign turns "increase $J$" into "minimise a loss". Writing $G_t = \sum_{\tau=t}^{T-1}\gamma^{\tau-t}r^\tau$ for the Monte Carlo return, REINFORCE's gradient estimate at time $t$ is $G_t\,\nabla_\phi\log\pi(a^t \mid s^t;\phi)$. With exact returns it is **unbiased**, it needs no critic, and it is simple. REINFORCE is **on-policy** ([[On-Policy Learning]]): each episode is used once and then discarded, because the gradient theorem's expectation is over the current policy's own behaviour.

**The problem: high variance.** Different trajectories can produce very different returns from the same state, because of random actions, random transitions and long episodes. The gradient estimates jump around, and training is unstable and slow.

**The fix: a [[Baseline|baseline]].** Subtract a function of the state from the return:

$$\nabla_\phi J(\phi) \propto \mathbb{E}\Big[\big(Q^\pi(s,a) - b(s)\big)\nabla_\phi\log\pi(a \mid s;\phi)\Big]$$

This leaves the expected gradient unchanged, since $\sum_a \pi(a \mid s)\nabla\log\pi(a \mid s) = \nabla\sum_a \pi(a \mid s) = \nabla 1 = 0$. But it reduces variance: if every return from a state is around 100, the raw estimate pushes every action up a lot and relies on small differences to sort them; with $b(s) \approx 100$, better-than-usual actions go up and worse-than-usual ones down.

## Actor-critic methods and A2C

[[Actor-Critic|Actor-critic]] methods learn two things: the **policy** (the **actor**) and a bootstrapped estimate of the state **value** (the **critic**), and use the critic in place of the Monte Carlo return.

- **Why it helps:** the update needs a single step instead of a full Monte Carlo rollout, so training is more efficient, and one-step estimates have **lower variance** than full returns.
- **The cost:** **bias from bootstrapping** ([[Bootstrapping]]). The critic's current local estimates may be inaccurate, and the actor learns from them anyway. The critic is also an extra component to train, and performance depends on the critic and rollout settings (extra hyperparameters).

[[Advantage Actor-Critic (A2C)|Advantage actor-critic (A2C)]] is the foundational actor-critic method. It guides the policy gradient with the **advantage** ([[Advantage Function]]):

> [!formula] Advantage
> $$\text{Adv}^\pi(s,a) = Q^\pi(s,a) - V^\pi(s), \qquad \text{Adv}(s^t,a^t) \approx r^t + \gamma V(s^{t+1}) - V(s^t)$$
> How much higher the expected return is when taking $a$ than when just following $\pi$. The estimate on the right needs only the critic $V$: $r^t + \gamma V(s^{t+1})$ is a one-step estimate of $Q^\pi(s^t,a^t)$, and $V(s^t)$ is the baseline. It is the [[TD Error|TD error]] of the critic.

If an action has a positive advantage, it did better than expected and its probability is pushed up; a negative advantage pushes it down.

```pseudo
Algorithm: A2C
──────────────
Initialize actor network π with random parameters φ
Initialize critic network V with random parameters θ
For every episode:
  For time step t = 0, 1, 2, ...:
    Observe current state s^t
    Sample action a^t ~ π(· | s^t; φ)
    Apply a^t; observe reward r^t and next state s^{t+1}
    If s^{t+1} is terminal:
      Adv(s^t, a^t) ← r^t − V(s^t; θ)
      Critic target y^t ← r^t
    Else:
      Adv(s^t, a^t) ← r^t + γ V(s^{t+1}; θ) − V(s^t; θ)     // bootstrapped advantage
      Critic target y^t ← r^t + γ V(s^{t+1}; θ)
    Actor loss  L(φ) ← −Adv(s^t, a^t) · log π(a^t | s^t; φ)
    Critic loss L(θ) ← (y^t − V(s^t; θ))²                  // regression, as in DQN
    Update φ by minimising the actor loss
    Update θ by minimising the critic loss
```

The critic is trained like DQN (squared error to a bootstrapped target), the actor like REINFORCE with the advantage in place of the return. The advantage can also use an **N-step return** (N rewards, then $V$), trading bias against variance. A2C is on-policy, so it cannot decorrelate its data with a replay buffer; the MARL book instead runs several environments in parallel and trains on a batch from all of them.

## PPO: limit how far one update moves the policy

Even with a critic, instability remains: a single policy gradient step can shift the policy a lot, and a bad step changes the data the agent collects next, which can make it worse. The solution is to **limit the change of the policy in a single update**. [[PPO|Proximal policy optimisation (PPO)]] does this, and is one of the most used policy gradient methods today.

> [!formula] PPO clipped loss
> $$\mathcal{L}(\phi) = -\min\Big(\rho(s^t,a^t)\,\text{Adv}(s^t,a^t),\;\; \text{clip}\big(\rho(s^t,a^t),\,1-\epsilon,\,1+\epsilon\big)\,\text{Adv}(s^t,a^t)\Big), \qquad \rho(s^t,a^t) = \frac{\pi(a^t \mid s^t;\phi)}{\pi(a^t \mid s^t;\phi_{\text{old}})}$$
> - $\rho$: the ratio between the new policy and the policy that collected the data (this definition is the MARL book's), an [[Importance Sampling|importance sampling]] weight; $\rho = 1$ means no change
> - $\text{clip}(\rho, 1-\epsilon, 1+\epsilon)$: $\rho$ cut off at the edges of $[1-\epsilon, 1+\epsilon]$
> - $\epsilon$: how far the ratio may move (a hyperparameter, unrelated to ε-greedy's ε)
> - the $\min$ takes the more pessimistic of the two terms

> [!example] What the clip does
> Take $\epsilon = 0.2$. A good action ($\text{Adv} > 0$) whose probability has already risen to $\rho = 1.5$: the terms are $1.5\,\text{Adv}$ and $1.2\,\text{Adv}$, the min picks the clipped $1.2\,\text{Adv}$, a constant, so there is no gradient pushing $\rho$ further up. A bad action ($\text{Adv} < 0$) already down to $\rho = 0.5$: the terms are $0.5\,\text{Adv}$ and $0.8\,\text{Adv}$, and since Adv is negative the min is the clipped $0.8\,\text{Adv}$, again constant. Inside $[0.8, 1.2]$ the loss is the plain advantage-weighted ratio and the gradient works normally.

Because the clip stops the policy from drifting far from the one that collected the data, PPO can make **several optimisation passes over the same batch** of rollouts. It remains **generally on-policy**: the batch is thrown away once the policy has moved on, and the clip is no strict guarantee against large or harmful changes.

## Comparing the deep RL methods

| | DQN | REINFORCE | A2C | PPO |
|---|---|---|---|---|
| Learns | $Q(s,a;\theta)$ | $\pi(a \mid s;\phi)$ | actor $\pi$, critic $V$ | actor $\pi$, critic $V$ |
| Weight on $\nabla\log\pi$ | none | MC return $G_t$ | bootstrapped Adv | clipped ratio × Adv |
| Data | off-policy, replay | on-policy, full episodes | on-policy, short rollouts | on-policy, several passes |
| Strength | data re-use | simple, unbiased | lower variance | stable updates |
| Weakness | discrete actions, ε-greedy | high variance, waits for episode end | critic bias, extra component, hyperparameters | more complex, still on-policy |

## Deep MARL: who sees what, and when

Lecture 3's setting stays ([[Stochastic Game]]): each agent $i$ has its own policy $\pi_i(a \mid s)$, transitions and rewards depend on the joint action, goals need not be aligned, and the four challenges remain (non-stationarity, equilibrium selection, [[Multi-Agent Credit Assignment|credit assignment]], the curse of dimensionality).

What deep MARL adds is a design choice about information. Each agent has to be **trained** (its parameters updated) and **executed** (it picks actions while acting), and each phase can be:

> [!definition] Centralised and decentralised
> **Centralised:** use information about the other agents (their observations, actions, policies, values, or a global state) to train an agent. **Decentralised:** condition only on the agent's own individual history $h_i^t$ (its own observations up to time $t$).

That gives three training and execution modes ([[Centralised Training with Decentralised Execution]]):

**Centralised training and execution (CTCE).** The centralised learning of Lecture 3: one policy over the joint action space, as if one agent controlled everyone. Challenges: it needs a joint reward, it does not scale with the number of agents (the joint action space grows exponentially), and by design agents may be distributed entities that cannot all be run from one place.

**Decentralised training and execution (DTDE).** Lecture 3's independent learning: each agent has access only to its own model and observations, in training and in execution. Challenges: it uses no information about the other agents, it suffers non-stationarity, it is hard to coordinate, and learning is unstable.

**Centralised training with decentralised execution (CTDE).** Train using shared information or a centralised critic, but have each agent act independently using only its local observations at execution time. The topic of [[AGAI-L05 - Deep MARL, Part II|Lecture 5]].

> [!example] Choosing a mode
> Autonomous cars cannot exchange all their sensor data while driving, but a simulator can provide everything during training. Execution must be decentralised and training can be centralised: CTDE. CTCE would need the impossible data exchange on the road; DTDE would throw away the simulator's information.

## Independent deep learners

Independent learning (DTDE) is a commonly used and competitive approach, and often it is simply the setting we have: when each organisation programs its own agent, nobody can train them centrally. Every single-agent algorithm above has an independent version: **IDQN, independent REINFORCE, IA2C, IPPO**. Each agent runs its own copy and treats the others as part of the environment ([[Independent Learning]]).

### Independent DQN and its stale replay buffer

In [[Independent DQN|independent DQN (IDQN)]] each agent $i$ trains its own action-value network $Q(\cdot\,;\theta_i)$, with its own target network and replay buffer, on its own observation history and reward (pseudocode as in the MARL book):

```pseudo
Algorithm: Independent DQN (n agents)
─────────────────────────────────────
For each agent i:
  Initialize value network θ_i, target network θ̄_i = θ_i, replay buffer D_i = {}
For every episode:
  For time step t = 0, 1, 2, ...:
    For each agent i: observe own history h_i^t;
      choose a_i^t ε-greedily from Q(h_i^t, · ; θ_i)
    Apply the joint action (a_1^t, ..., a_n^t)
    For each agent i: observe own reward r_i^t and next history h_i^{t+1}
    For each agent i:
      Store (h_i^t, a_i^t, r_i^t, h_i^{t+1}) in D_i      // the others' actions are not stored
      Sample minibatch of B transitions from D_i
      y^k ← r^k if terminal, else r^k + γ · max_a' Q(h^{k+1}, a'; θ̄_i)
      L(θ_i) ← (1/B) Σ_k (y^k − Q(h^k, a^k; θ_i))²
      Update θ_i by minimising L(θ_i)
      In a set interval: θ̄_i ← θ_i
```

**The new challenge: stored experience goes out of date.** A transition in $\mathcal{D}_i$ records what happened while the others still used their old policies. Once they change, the same action in the same state may lead elsewhere, so agent $i$ keeps training on an environment that no longer exists. DQN's off-policy property does not help: it covers agent $i$'s own old behaviour, while the others' old behaviour is part of agent $i$'s environment (Lecture 3's non-stationarity, stored and replayed).

- **Solution: smaller replay buffers**, which hold only recent transitions from policies close to the others' current ones. A larger buffer makes it worse.
- **A second remedy, from the MARL book (Ch 9):** store the agents' action probabilities with each transition and, when replaying it, weight its loss by an importance sampling ratio of the other agents' current to stored probabilities, so experience the others would now be unlikely to produce counts less.

### Independent policy gradient methods

Each agent $i$ follows its own policy gradient:

$$\nabla_{\phi_i}J(\phi_i) \propto \mathbb{E}_\pi\Big[u_i^t\,\nabla_{\phi_i}\log\pi(a_i^t \mid h_i^t;\phi_i)\Big]$$

where:
- $\phi_i$: agent $i$'s policy parameters
- $h_i^t$: agent $i$'s own observation history; $a_i^t$: its own action
- $u_i^t$: agent $i$'s utility estimate for that action: its Monte Carlo return in independent REINFORCE, its advantage in IA2C and IPPO
- $\mathbb{E}_\pi$: the expectation runs over the experience generated by **all** agents' current policies

**Advantage over IDQN: they always learn from the updated policies of the other agents.** They are on-policy, so each update uses fresh experience generated by the others' current policies, then discards it. Nothing goes stale in a buffer, so independent REINFORCE adapts faster to other learners than IDQN.

**Challenge:** in general, independent MARL **exacerbates** the sources of learning instability. Monte Carlo returns were already high-variance; now they also vary because the others keep changing.

- **Independent REINFORCE:** simple, decentralised, no critic, works with local or shared team rewards. High variance, worst with long episodes and shared team rewards.
- **Independent A2C (IA2C, IAC):** each agent has an actor and its own critic over its own observations. Lower variance and more frequent updates than REINFORCE. But the critic must learn in a changing environment, and with a shared team reward it **does not solve credit assignment**: it cannot tell which agent earned the reward.
- **Independent PPO (IPPO):** each agent clips its own update relative to its previous policy. Often stable and practical, several passes over its rollout data, and agents act on local information only. But each clip controls only that agent's change, with no guarantee that the **joint** policy changes safely, and the others' simultaneous updates keep the environment non-stationary.

Assignment 1 is this setting: one learning agent in a Coin Game against an opponent that is greedy, altruistic, reciprocating or itself learning. The opponent is not ours to train, so the learner is independent by construction.

## What to be able to answer

- Why tables fail; the lamps arithmetic ($1{,}458 \cdot 2^k$ rows against a fixed budget).
- Linear against neural approximation; CNNs and RNNs.
- DQN: input and output, loss, terminal target, no gradient through the target, moving targets and correlated data, target network and replay buffer, pseudocode, off-policy.
- Limits of value-based policies; softmax policy; the policy gradient theorem.
- REINFORCE, baselines, A2C, PPO's clip, and the comparison table.
- CTCE, DTDE, CTDE and when each fits; IDQN's stale replay and its fixes; why independent policy gradients adapt faster; IREINFORCE, IA2C, IPPO.

Model answers and the mock questions are below.

## Exam questions

> [!exam]- Mock Q13 (DRL 1): Why does tabular Q-learning fail in complex environments? Select all that apply. (A) A table cannot store values for discrete actions. (B) Many states are visited rarely or never during training. (C) What is learned in one row is not used in other rows. (D) The memory needed grows with the number of state-action pairs. (E) The update rule is undefined when rewards are stochastic.
> **Key points:** B, C and D.
>
> - (A) Wrong: a table is exactly a store of one value per discrete action in each state.
> - **(B) Correct:** the number of states explodes, so with a fixed budget many rows are rarely or never updated.
> - **(C) Correct:** no generalisation; each value is updated in isolation.
> - **(D) Correct:** the table grows with the number of state-action pairs.
> - (E) Wrong: with stochastic rewards the update is a running average and remains well defined.

> [!exam]- Mock Q14 (DRL 2): What does the replay buffer of DQN contribute? Select all that apply. (A) It stores earlier copies of the network weights. (B) It turns DQN into an on-policy method. (C) It reduces the correlation between the samples in a minibatch. (D) It removes the need for a target network. (E) It allows each transition to be used for several updates.
> **Key points:** C and E.
>
> - (A) Wrong: that is the target network; the buffer stores transitions $(s,a,r,s')$.
> - (B) Wrong: replaying old transitions is only possible because DQN is off-policy.
> - **(C) Correct:** uniform sampling mixes transitions from many episodes, breaking the correlation of consecutive steps.
> - (D) Wrong: the buffer addresses correlated data, the target network addresses moving targets; DQN uses both.
> - **(E) Correct:** a stored transition can be sampled many times, which re-uses experience.

> [!exam]- Mock Q15 (DRL 3): You add a random binary input that does not affect the game to the state of a Q-table. What is the effect? Select one answer. (a) The rows double, and values are not shared between rows that differ in this input. (b) The rows stay the same, because the table learns to ignore the input. (c) The rows double, but each update also copies the value to the matching row. (d) The rows stay the same, but the optimal policy of the game changes.
> **Key points:** (a).
>
> - **(a) Correct:** each state now comes in two versions, and a table has no mechanism to share values between rows.
> - (b) Wrong: a table cannot ignore an input; every distinct input is its own row.
> - (c) Wrong: a table update changes exactly one entry.
> - (d) Wrong: the input does not affect the game, so the optimal policy stays the same, and the table does grow.

> [!exam]- Mock Q16 (DRL 4): What are the input and output of a standard DQN? Select one answer. (a) Input: the state. Output: a probability for each discrete action. (b) Input: the state. Output: one Q-value for each discrete action. (c) Input: the state and an action. Output: a probability for that action. (d) Input: an action. Output: the value of each possible next state.
> **Key points:** (b).
>
> - (a) Wrong: probabilities per action are the output of a policy network.
> - **(b) Correct:** the state goes in and one output per discrete action gives $Q(s,a_1;\theta), \dots, Q(s,a_{|A|};\theta)$, so one pass gives all values.
> - (c) Wrong: DQN outputs values, and takes no action as input.
> - (d) Wrong: DQN does not predict next states.

> [!exam]- Mock Q17 (DRL 5): Which statements about DQN are correct? Select all that apply. (A) It directly learns a network that outputs action probabilities. (B) No gradient flows through the target y during an update. (C) It usually collects data with an ε-greedy policy. (D) It is an off-policy method. (E) It handles continuous actions without any changes.
> **Key points:** B, C and D.
>
> - (A) Wrong: that describes policy gradient methods; DQN learns Q-values.
> - **(B) Correct:** the target is computed with the target network and treated as a constant; only the estimate $Q(s,a;\theta)$ is differentiated.
> - **(C) Correct:** actions are chosen ε-greedily from the current Q-network.
> - **(D) Correct:** the max in its target learns the greedy policy's values from data collected by ε-greedy and older policies in the buffer.
> - (E) Wrong: it needs one output per discrete action and a max over actions.

> [!exam]- Mock Q18 (DRL 6): In DQN, what is the target y when the next state s' is terminal? Select one answer. (a) y = r (b) y = 0 (c) y = r + γ max_a' Q(s', a'; θ⁻) (d) y = max_a' Q(s', a'; θ⁻)
> **Key points:** (a).
>
> - **(a) Correct:** after a terminal state no future reward follows, so the target is just the last reward.
> - (b) Wrong: that drops the reward that was actually received.
> - (c) Wrong: that is the non-terminal target; it would add value that cannot be collected.
> - (d) Wrong: it ignores the reward and still bootstraps from the terminal state.

> [!exam]- Mock Q19 (DRL 7): In A2C, the estimate Adv(s, a) = r + γ V(s') − V(s) is positive. What does the actor update do? Select one answer. (a) It makes action a more likely, because a did better than expected. (b) It makes action a less likely, because the critic was too pessimistic. (c) It sets the probability of action a to 1 for this state. (d) It leaves the policy unchanged, and only the critic is updated.
> **Key points:** (a).
>
> - **(a) Correct:** the actor loss $-\text{Adv}\log\pi(a \mid s;\phi)$ with positive Adv is minimised by raising $\pi(a \mid s)$: the outcome beat the critic's expectation $V(s)$.
> - (b) Wrong: a positive advantage increases the probability.
> - (c) Wrong: a gradient step moves the probability a little.
> - (d) Wrong: actor and critic are both updated every step.

> [!exam]- Mock Q20 (DRL 8): Which quantity does REINFORCE use to estimate the policy gradient at time t? G_t is the Monte Carlo return. Select one answer. (a) G_t ∇_φ max_a Q(s_t, a; φ) (b) π(a_t | s_t; φ) ∇_φ G_t (c) G_t ∇_φ log π(a_t | s_t; φ) (d) ∇_φ (G_t − π(a_t | s_t; φ))²
> **Key points:** (c).
>
> - (a) Wrong: REINFORCE has no Q-network.
> - (b) Wrong: the return is a sampled number and has no gradient with respect to $\phi$.
> - **(c) Correct:** the policy gradient theorem with $Q^\pi(s_t,a_t)$ replaced by the return $G_t$.
> - (d) Wrong: that is a regression loss of a probability onto a return, which is meaningless.

> [!exam]- Mock Q21 (DRL 9, open, 10 points): Tabular Q-learners play the Coin Game, whose table has 1,458 rows. Each of k random lamps doubles the number of rows (as lamps can be on or off). Training plays 300 updates, each with 32 parallel games of 16 steps, and every step gives one decision per agent. Coins per episode are 14.5 with no lamps and 4.9 with 8 lamps, while random agents collect 3.7. (a) Compute the number of rows N for k = 0 and k = 8, and the budget B of decisions per agent. (3 points) (b) Use these numbers to explain the drop in performance. Why can't agents with tabular learning discover that the lamps do not matter? (3 points) (c) Explain how a neural network avoids this problem. Name one new challenge it introduces. (4 points)
> **Key points:** $N = 1{,}458$ for $k = 0$ and $1{,}458 \cdot 2^8$ for $k = 8$, $B = 300 \cdot 32 \cdot 16$; with 8 lamps most rows are never visited and the rest only a few times, unvisited rows hold zeros so the agent acts at random, rows grow while the budget stays; rows differing in one lamp share nothing, so each lesson must be relearned for every lamp pattern; a network computes values or a policy from the observation with shared weights, so what it learns about the coin's position applies to all lamp patterns and the lamps can get little weight; new challenges: instability from moving targets or correlated data, possible divergence, no general convergence guarantee, more computation.
>
> **(a)** Without lamps $N = 1{,}458$. With 8 lamps $N = 1{,}458 \cdot 2^8 = 373{,}248$. Each agent makes one decision per step, so $B = 300 \cdot 32 \cdot 16 = 153{,}600$ decisions.
>
> **(b)** Without lamps the budget gives about $153{,}600 / 1{,}458 \approx 105$ decisions per row, enough to learn useful values. With 8 lamps it is about $153{,}600 / 373{,}248 \approx 0.4$ per row: most rows are never visited, and the others only a few times. An unvisited row still holds its initial zeros, so the agent acts at random there, and rows visited once or twice have barely trained values. The number of rows grows by $2^k$ while the budget stays the same, so the probability of visiting each state becomes too low, and performance falls from 14.5 coins to 4.9, close to the 3.7 of random agents. The table cannot discover that the lamps do not matter, because two rows that differ in one lamp share nothing: whatever is learned in one row is not used in the other, so each lesson must be learned again for each of the other lamp patterns.
>
> **(c)** A neural network computes values or a policy as a function of the observation, with weights shared by all inputs. What it learns about the coin's position applies to all lamp patterns, and since the lamps carry no information about the return, it can learn to give them little weight. It generalises across the states the table kept apart. New challenges: instability from moving targets (the targets depend on the weights being trained) or from correlated data, possible divergence, no general convergence guarantee, and more computation per decision.

> [!exam]- Mock Q22 (DMARL 1): Which of these can reduce the problem of outdated experience in IDQN? Select all that apply. (A) Removing the target network (B) Storing importance sampling weights with the action probabilities of all agents (C) Using a larger replay buffer (D) Using a higher discount factor (E) Using a smaller replay buffer
> **Key points:** B and E.
>
> - (A) Wrong: the target network addresses moving targets; removing it destabilises learning and leaves the stale data in place.
> - **(B) Correct:** with the agents' action probabilities stored, a replayed transition can be reweighted by how likely the others are to act that way now.
> - (C) Wrong: a larger buffer keeps older experience from the others' older policies, which makes the problem worse.
> - (D) Wrong: the discount factor sets how far ahead the agent looks and has nothing to do with the age of the data.
> - **(E) Correct:** a smaller buffer holds only recent transitions, generated by policies close to the others' current ones.

> [!exam]- Mock Q23 (DMARL 2): Autonomous cars cannot exchange all sensor data while driving, but a simulator can provide all data during training. Which mode fits best? Select one answer. (a) Centralised training and execution (b) Decentralised training and execution (c) Centralised training with decentralised execution (d) Decentralised training with centralised execution
> **Key points:** (c).
>
> - (a) Wrong: centralised execution would need the sensor exchange that is impossible while driving.
> - (b) Wrong: it would ignore the full information the simulator offers during training.
> - **(c) Correct:** training can use the simulator's shared information, and each car then acts on its own sensors.
> - (d) Wrong: execution is exactly the phase that cannot be centralised here, and it is not one of the three standard modes.

> [!exam]- Mock Q26 (DMARL 5): Why can independent REINFORCE adapt faster to other learning agents than IDQN? Select one answer. (a) It learns from recent experience, generated by the current policies of the other agents. (b) It reads the current policy parameters of the other agents during training. (c) It reuses old experience, which covers more policies of the other agents. (d) It uses a centralised critic that is conditioned on the joint action of all agents.
> **Key points:** (a).
>
> - **(a) Correct:** REINFORCE is on-policy: each update uses the latest episodes and discards them, while IDQN's buffer replays experience from the others' old policies.
> - (b) Wrong: independent learners have no access to the others' parameters.
> - (c) Wrong: that is IDQN's replay, the source of the problem.
> - (d) Wrong: a centralised critic belongs to CTDE; independent REINFORCE has no critic at all.

> [!exam]- Explain the moving target problem and the problem of correlated experience in deep Q-learning, and how DQN solves each. Give DQN's pseudocode and explain why it can use a replay buffer.
> **Key points:** targets computed with the trained $\theta$, every update changes many values and hence the targets (unlike one table entry); consecutive transitions are not i.i.d., leading to overfitting to recent data and catastrophic forgetting; target network $\bar\theta$ updated less often; replay buffer with uniform minibatches (also re-use and stable gradients); off-policy because the max target is independent of the behaviour policy.
>
> In deep Q-learning the network is trained to minimise $(y^t - Q(s^t,a^t;\theta))^2$ with target $y^t = r^t + \gamma\max_{a'}Q(s^{t+1},a';\theta)$. Because the target uses the same parameters, and a network generalises, every gradient step changes the values of many states, including those the targets come from. In a table one entry changes per update; here the targets move after every step, which makes learning unstable. Second, transitions arrive in sequence, so consecutive samples are highly correlated and cover a small part of the state space. Gradient descent on such data overfits to recent experience and forgets what was learned elsewhere (catastrophic forgetting).
>
> DQN keeps a target network with parameters $\bar\theta$, used only to compute targets $r + \gamma\max_{a'}Q(s',a';\bar\theta)$ and copied from $\theta$ at set intervals, so targets stay fixed in between. It also stores every transition in a replay buffer and trains on minibatches sampled uniformly from it, which decorrelates the samples, lets each transition be used several times, and averages gradients over a batch.
>
> Pseudocode: initialise $\theta$, set $\bar\theta = \theta$, empty buffer. Each step: choose $a^t$ ε-greedily from $Q(s^t,\cdot\,;\theta)$, act, observe $r^t$ and $s^{t+1}$, store the transition, sample $B$ transitions, set $y^k = r^k$ if $s^{k+1}$ is terminal and $r^k + \gamma\max_{a'}Q(s^{k+1},a';\bar\theta)$ otherwise, minimise $\frac1B\sum_k(y^k - Q(s^k,a^k;\theta))^2$, and at set intervals copy $\bar\theta \leftarrow \theta$.
>
> The buffer contains actions chosen by older policies. That is allowed because Q-learning is off-policy: the max in its target estimates the greedy policy's values whatever policy produced the data.

> [!exam]- Derive the motivation for policy gradient methods, state the policy gradient theorem, and explain REINFORCE, including its main weakness and the fix.
> **Key points:** ε-greedy policies over discrete actions cannot express arbitrary or continuous policies; softmax policy network; $J(\phi) = \sum_s d^{\pi}(s)V^\pi(s)$ and $\nabla d^\pi$ needs the model; theorem $\nabla J \propto \mathbb{E}[Q^\pi\nabla\log\pi]$ without $\nabla d$; REINFORCE uses the Monte Carlo return; unbiased but high variance; baseline $b(s)$ keeps it unbiased and reduces variance.
>
> A value-based agent derives its policy from Q-values, usually ε-greedily, so it can only express "one favourite action plus uniform noise" and needs one output per discrete action. It cannot represent a policy like (0.4, 0.1, 0.5), mixed strategies needed in games such as matching pennies, or continuous actions. A policy network outputs preferences, turned into probabilities by a softmax, so any distribution is reachable.
>
> The objective is $J(\phi) = \sum_s d^{\pi_\phi}(s)V^{\pi_\phi}(s)$. Differentiating the visitation probabilities is hard: changing $\phi$ changes the action probabilities at every step, early actions change later states, and the exact effect needs the transition model. The policy gradient theorem avoids it: $\nabla_\phi J(\phi) \propto \mathbb{E}_{s,a \sim \pi}[Q^\pi(s,a)\nabla_\phi\log\pi(a \mid s;\phi)]$. It raises the probability of each action in proportion to its value, and it can be estimated from samples. $Q^\pi$ still has to be estimated.
>
> REINFORCE estimates it with the Monte Carlo return. It plays an episode with the current policy, then minimises $-\frac1T\sum_t G_t\log\pi(a^t \mid s^t;\phi)$ with $G_t = \sum_{\tau \ge t}\gamma^{\tau-t}r^\tau$. It is simple, needs no critic, and its gradient estimate is unbiased. Its weakness is high variance: trajectories from the same state can give very different returns, so gradients are noisy and training is unstable and slow; it also has to wait for the end of each episode. The fix is a baseline: use $G_t - b(s^t)$. Since $\sum_a\pi\nabla\log\pi = 0$, a state-dependent baseline leaves the expected gradient unchanged while reducing its variance. Using a learned $V(s)$ as the baseline leads to actor-critic methods.

> [!exam]- Explain how A2C and PPO improve on REINFORCE. Define the advantage, give the A2C actor and critic losses, and explain the PPO clipped objective.
> **Key points:** actor and critic; advantage $Q - V$ estimated by $r + \gamma V(s') - V(s)$; actor loss $-\text{Adv}\log\pi$, critic loss $(y - V)^2$; single-step updates, lower variance, bias from bootstrapping; PPO: a single step can shift the policy too far; ratio $\rho = \pi_\phi/\pi_{\text{old}}$, clip to $[1-\epsilon, 1+\epsilon]$, pessimistic min; several passes over a batch, still on-policy.
>
> A2C learns a policy (the actor) and a state-value estimate (the critic). It replaces the return by the advantage $\text{Adv}^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$, how much better $a$ is than following the policy, estimated in one step as $r^t + \gamma V(s^{t+1}) - V(s^t)$ (just $r^t - V(s^t)$ at a terminal step). The actor minimises $-\text{Adv}(s^t,a^t)\log\pi(a^t \mid s^t;\phi)$, raising the probability of actions that did better than expected and lowering the others; the critic minimises $(y^t - V(s^t;\theta))^2$ with $y^t = r^t + \gamma V(s^{t+1};\theta)$, like value-based regression. It updates after every step and has lower variance than REINFORCE, at the price of bias, since the critic's bootstrapped estimates may be wrong.
>
> Policy gradient steps can still change the policy a lot in one update, a source of instability. PPO limits the change. With $\rho = \pi(a \mid s;\phi)/\pi(a \mid s;\phi_{\text{old}})$, the ratio of new to old probability, it minimises $-\min(\rho\,\text{Adv}, \text{clip}(\rho, 1-\epsilon, 1+\epsilon)\,\text{Adv})$. Once a good action's ratio exceeds $1+\epsilon$, or a bad action's falls below $1-\epsilon$, the clipped term is constant and there is no gradient to move it further. This keeps updates stable and lets PPO run several optimisation passes over each batch of rollouts. It is more complex than REINFORCE, remains generally on-policy, and the clip is no strict guarantee against harmful changes.

> [!exam]- Describe the three training and execution modes of multi-agent deep RL, and explain how IDQN and independent policy gradient methods differ in their main weakness.
> **Key points:** centralised uses others' information, decentralised only the own history; CTCE: one policy over the joint action space, joint reward, does not scale, agents may be distributed; DTDE: independent learning, non-stationarity, poor coordination, unstable; CTDE: shared information or centralised critic in training, local observations in execution; IDQN's buffer holds experience from others' old policies (smaller buffers, importance weights); independent policy gradients learn from fresh on-policy data but have amplified variance; IA2C's critic does not solve credit assignment; IPPO's clip is per agent.
>
> Training an agent is centralised when it uses information about the other agents (their observations, actions, policies or a global state), decentralised when it conditions only on the agent's own history; the same distinction applies to execution. Centralised training and execution (CTCE) trains one policy over the joint action space, as if one agent controlled all; it needs a joint reward, does not scale because the joint action space grows exponentially, and is impossible when agents are distributed by design. Decentralised training and execution (DTDE) is independent learning: each agent has only its own model and observations; it uses nothing about the others, faces non-stationarity, coordinates poorly and learns unstably. Centralised training with decentralised execution (CTDE) uses shared information or a centralised critic during training and lets each agent act on its local observations at execution, for example cars trained in a simulator that cannot share sensor data on the road.
>
> In IDQN each agent trains its own DQN with its own replay buffer. The buffer stores transitions produced while the others used older policies; once they change, those transitions describe a different environment, and the agent keeps learning from them. Remedies: a smaller buffer, or storing the agents' action probabilities and reweighting replayed transitions by importance sampling. Independent policy gradient methods are on-policy, so each update uses experience generated by the others' current policies and nothing goes stale; they adapt faster. Their weakness is that independent learning amplifies existing instability: Monte Carlo returns in independent REINFORCE vary even more as the others change, IA2C's critic must track a moving environment and, with a shared team reward, does not solve credit assignment, and IPPO's clip bounds only each agent's own change, so the joint policy can still shift a lot.

## Flashcards

> [!card]- Why can a tabular Q-learner not generalise?
> Each value is updated in isolation; nothing learned in one row reaches another.

> [!card]- How does the size of a Q-table scale?
> Linearly with the number of possible inputs (state-action pairs).

> [!card]- What happens to a Q-table when an irrelevant binary input is added to the state?
> The rows double, and rows differing in that input share nothing.

> [!card]- How many Coin Game Q-table rows are there with 8 irrelevant lamps?
> $1{,}458 \cdot 2^8 = 373{,}248$.

> [!card]- What does a tabular Q-learner do in a state it never visited?
> All values are still zero, so it acts at random.

> [!card]- What is the goal of function approximation in RL?
> Learn parameters $\theta$ so that $f(x;\theta)$ approximates the optimal function $f^*$.

> [!card]- Which two functions can a function approximator represent in RL?
> A value function or a policy.

> [!card]- What is the weakness of linear function approximation?
> High bias: not expressive enough for complex functions.

> [!card]- What do deep networks offer over linear function approximation?
> They learn state features automatically and represent non-linear functions.

> [!card]- Which network type suits image observations?
> A convolutional neural network.

> [!card]- Which network type suits a sequence of observations?
> A recurrent neural network, which carries a hidden state.

> [!card]- What are the input and output of a standard DQN?
> Input the state; output one Q-value per discrete action.

> [!card]- What is the deep Q-learning loss?
> $(y^t - Q(s^t,a^t;\theta))^2$.

> [!card]- What is the DQN target when the next state is terminal?
> $y = r$.

> [!card]- Does gradient flow through the target in DQN?
> No: the target is treated as a constant; only the estimate is differentiated.

> [!card]- What is the moving target problem in deep Q-learning?
> Targets depend on $\theta$, so every update to $\theta$ also moves the targets.

> [!card]- Why do targets move faster with a network than with a table?
> A network update changes many values at once; a table update changes one entry.

> [!card]- What do correlated consecutive experiences cause in deep Q-learning?
> Overfitting to recent experience and catastrophic forgetting of earlier estimates.

> [!card]- What convergence property does deep Q-learning lose compared with tabular Q-learning?
> A general convergence guarantee: training can diverge.

> [!card]- How does a target network reduce instability?
> Targets use a copy $\bar\theta$ updated less often, so they stay fixed in between.

> [!card]- How does the replay buffer reduce correlation?
> Minibatches are sampled uniformly from many stored transitions.

> [!card]- Name two extra benefits of a replay buffer besides decorrelation.
> Re-use of experience; more stable gradients from batches.

> [!card]- What three ingredients make up DQN?
> Q-learning, a target network, a replay buffer.

> [!card]- Why can DQN learn from a replay buffer?
> It is off-policy: the max target learns greedy values from any behaviour data.

> [!card]- Why are ε-greedy policies restrictive?
> They give one favourite action and spread the rest uniformly; arbitrary distributions are impossible.

> [!card]- Why does DQN struggle with continuous actions?
> It needs one output per discrete action and a max over all actions.

> [!card]- How does a policy network turn preferences into action probabilities?
> A softmax over the preferences, with temperature $\tau$.

> [!card]- What does $d^{\pi_\phi}(s)$ denote in the policy objective?
> The visitation probability of state $s$ under the policy.

> [!card]- Why is $\nabla_\phi d^{\pi_\phi}(s)$ hard to compute?
> Effects of $\phi$ compound along trajectories; computing it needs the transition model.

> [!card]- What does the policy gradient theorem remove from the gradient of $J$?
> The derivative of the state visitation probabilities.

> [!card]- What is the policy gradient theorem, in expectation form?
> $\nabla_\phi J \propto \mathbb{E}[Q^\pi(s,a)\nabla_\phi\log\pi(a \mid s;\phi)]$.

> [!card]- Which two ways of estimating $Q^\pi$ give REINFORCE and actor-critic methods?
> Monte Carlo returns (REINFORCE); a learned critic network (actor-critic).

> [!card]- What quantity does REINFORCE use to estimate the policy gradient at time $t$?
> $G_t\nabla_\phi\log\pi(a_t \mid s_t;\phi)$.

> [!card]- Why does REINFORCE have high variance?
> Different trajectories from the same state give very different returns.

> [!card]- Why does a state baseline not bias the policy gradient?
> $\sum_a \pi\nabla\log\pi = \nabla\sum_a\pi = 0$, so the baseline term averages out.

> [!card]- What are the actor and the critic in actor-critic methods?
> Actor: the policy. Critic: a bootstrapped estimate of the state value.

> [!card]- What do actor-critic methods gain over REINFORCE?
> Single-step updates and lower variance.

> [!card]- What do actor-critic methods lose against REINFORCE?
> Unbiasedness: bootstrapping from the critic adds bias.

> [!card]- What is the advantage $\text{Adv}^\pi(s,a)$?
> $Q^\pi(s,a) - V^\pi(s)$: how much better $a$ is than following $\pi$.

> [!card]- How does A2C estimate the advantage in one step?
> $r^t + \gamma V(s^{t+1}) - V(s^t)$.

> [!card]- What is the A2C actor loss?
> $-\text{Adv}(s^t,a^t)\log\pi(a^t \mid s^t;\phi)$.

> [!card]- What does a positive advantage do to the action's probability in A2C?
> Raises it: the action did better than expected.

> [!card]- What problem does PPO address?
> A single policy gradient step can shift the policy too much.

> [!card]- What is the ratio $\rho$ in PPO?
> $\pi(a \mid s;\phi) / \pi(a \mid s;\phi_{\text{old}})$, new over old probability.

> [!card]- What range does PPO clip the ratio to?
> $[1-\epsilon, 1+\epsilon]$.

> [!card]- Why can PPO make several passes over one batch?
> The clip keeps the policy close to the one that collected the data.

> [!card]- Is PPO on-policy or off-policy?
> Generally on-policy: old data cannot be reused indefinitely.

> [!card]- What is centralised information in multi-agent training?
> Information about other agents: their observations, actions, policies or a global state.

> [!card]- Which training mode trains one policy over the joint action space?
> Centralised training and execution (CTCE).

> [!card]- Name the challenges of centralised training and execution.
> Needs a joint reward, does not scale, agents may be distributed.

> [!card]- Which training mode is independent learning?
> Decentralised training and execution (DTDE).

> [!card]- What does CTDE do at training and at execution?
> Trains with shared information or a centralised critic; acts on local observations.

> [!card]- Why is independent learning often the setting we actually have?
> Each organisation programs its own agent, so nobody can train them centrally.

> [!card]- What is the new challenge of replay buffers in IDQN?
> Stored experience goes stale once the other agents change their policies.

> [!card]- Which change to the replay buffer reduces stale experience in IDQN?
> Make it smaller, so it holds only recent transitions.

> [!card]- How can importance sampling reduce stale replay in IDQN?
> Store action probabilities; reweight replayed transitions by the others' current over stored probabilities.

> [!card]- Why do independent policy gradient methods adapt faster than IDQN?
> They learn on-policy from fresh experience of the others' current policies.

> [!card]- What does independent learning do to sources of instability like Monte Carlo variance?
> It exacerbates them, since the other agents' changes add noise.

> [!card]- Why does an IA2C critic not solve credit assignment with a shared reward?
> It estimates the team's value, which says nothing about whose action earned it.

> [!card]- What does each IPPO agent's clip fail to guarantee?
> That the joint policy changes safely.

> [!card]- True or false: the replay buffer stores earlier copies of the network weights.
> **False.** It stores transitions; the copied weights are the target network.

> [!card]- True or false: DQN is an off-policy method.
> **True.** Its max target learns greedy values from ε-greedy and older data.

> [!card]- True or false: the replay buffer removes the need for a target network.
> **False.** They fix different problems: correlation and moving targets.

> [!card]- True or false: with exact Monte Carlo returns, REINFORCE's gradient estimate is biased.
> **False.** It is unbiased; its problem is high variance.

> [!card]- True or false: actor-critic methods have higher variance than REINFORCE.
> **False.** They have lower variance and some bias from bootstrapping.

> [!card]- True or false: PPO's clipping strictly guarantees that no update is harmful.
> **False.** Clipping limits the ratio but is no strict guarantee.

> [!card]- True or false: a larger replay buffer reduces outdated experience in IDQN.
> **False.** A smaller buffer does; a larger one keeps older experience.

> [!card]- True or false: in CTDE, agents use shared information at execution time.
> **False.** Shared information is used only during training.

> [!card]- Multiple selection: which are true of the DQN target network? (A) Its parameters are copied from the main network at set intervals. (B) It computes the targets $y$. (C) It decorrelates minibatches. (D) It receives gradient updates every step.
> **A and B.** C: that is the replay buffer. D: it is only copied, never trained directly.

> [!card]- Multiple selection: which are true of REINFORCE? (A) It needs no critic. (B) It updates after every step. (C) It uses Monte Carlo returns. (D) It reuses old episodes from a replay buffer.
> **A and C.** B: it waits for the episode to end. D: it is on-policy.

> [!card]- Multiple selection: which are disadvantages of A2C compared with REINFORCE? (A) The critic is an extra component to train. (B) Bootstrapping can introduce bias. (C) It must wait until an episode ends. (D) Performance depends on critic and rollout settings.
> **A, B and D.** C: A2C can update from single steps or short rollouts.

> [!card]- Multiple selection: which are challenges of decentralised training and execution? (A) Non-stationarity. (B) Joint action space growing exponentially. (C) Hard to coordinate. (D) Unstable learning.
> **A, C and D.** B: that is the problem of centralised training and execution.

## Links

- **Course:** [[AGAI - Overview|Course overview]]
- **Previous lecture:** [[AGAI-L03 - Intro to Multi-Agent Reinforcement Learning]] (tabular Q-learning, centralised and independent Q-learning, the MARL challenges)
- **Next lecture:** [[AGAI-L05 - Deep MARL, Part II]] (CTDE, centralised critics, parameter and experience sharing)
- **Lab:** [[AGAI-Lab02 - Learning Agents]] (the Coin Game, irrelevant lamps, tabular Q-learning against PPO)
- **RL background in depth:** [[RL-L05 - Tabular to Approximation]], [[RL-L08 - Deep RL Value-Based]] (DQN), [[RL-L09 - Policy Gradient Methods]] (policy gradient theorem, REINFORCE, actor-critic), [[RL-L10 - Advanced Policy Search]] (PPO), [[RL-L13 - Partial Observability]] (RNNs and histories)
- **Concepts introduced here:** [[Centralised Training with Decentralised Execution]], [[Independent DQN]]
- **Source:** Agentic Artificial Intelligence, Lecture 4, "Intro to Deep RL and Deep MARL", Roman Chiva Gil, MSc AI, UvA, 5 November 2026
- **Reading:** Albrecht, Christianos and Schäfer, *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches* (MIT Press, 2024), Chapter 8 (deep RL: function approximation, DQN, policy gradients, REINFORCE, A2C, PPO) and Chapter 9.1 and 9.3 (training and execution modes, independent deep learners)
- **Papers:** Mnih et al., "Human-level control through deep reinforcement learning", Nature, 2015 (DQN)
- **Mock exam:** Mock Midterm 2026, questions 13 to 21 (Deep RL) and 22, 23, 26 (Deep MARL)
