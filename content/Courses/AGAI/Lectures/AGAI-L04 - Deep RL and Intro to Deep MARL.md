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
> A Q-table learns each value in isolation, so in a large state space most rows are never visited. A neural network's shared weights generalise. DQN is Q-learning with a network, a target network (against moving targets) and a replay buffer (against correlated data). Value-based agents only express ε-greedy policies over discrete actions, so policy gradient methods learn the policy itself: REINFORCE (Monte Carlo return, high variance), A2C (critic's advantage, lower variance, some bias), PPO (clipped updates). In MARL, training and execution can each be centralised or decentralised; IDQN's replay buffer goes stale as the other agents change.
>
> Feeds mock questions 13 to 21 (Deep RL, including the open question 21 on the Coin Game) and 22, 23 and 26 (stale replay in IDQN, choosing a training mode, independent REINFORCE against IDQN).

## Why tables stop working

Tabular Q-learning ([[AGAI-L03 - Intro to Multi-Agent Reinforcement Learning|Lecture 3]]) keeps one row per state and one column per action, and moves each entry towards a target (reward plus discounted value of the best next action):

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\Big(\underbrace{r_t + \gamma\max_{a'}Q(s_{t+1},a')}_{\text{target}} - \underbrace{Q(s_t,a_t)}_{\text{estimate}}\Big)$$

In complex environments this fails for two reasons.

- **It cannot generalise.** Each value is updated in isolation, so learning about one state teaches nothing about a similar one, and the agent has to visit every state-action pair many times. As the number of states explodes, many are visited rarely or never. An unvisited row still holds its initial zeros, so the agent acts at random there.
- **It needs too much space.** The table grows linearly with the number of possible inputs (state-action pairs), which grows exponentially with the number of state variables.

Stochastic rewards and discrete actions are no problem for a table.

> [!example] Irrelevant lamps in the Coin Game ([[AGAI-Lab02 - Learning Agents|Lab 2]])
> A state is (own cell, other agent's cell, coin cell, whether the coin has the agent's colour) on a 3×3 grid: $9 \cdot 9 \cdot 9 \cdot 2 = 1{,}458$ rows. Add $k$ lamps that switch on and off at random and never affect the game. To a table, observations that differ in one lamp are different states, so each lamp **doubles the rows**: $N = 1{,}458 \cdot 2^k$, which is $373{,}248$ for $k = 8$, and rows that differ only in a lamp share nothing.
>
> The budget is fixed in decisions per agent: $B = \text{updates} \times \text{parallel games} \times \text{steps} = 300 \cdot 32 \cdot 16 = 153{,}600$, about 105 decisions per row without lamps and 0.4 with 8 lamps. Most rows are never visited, and the agents play close to random.

## Function approximation: learn a function instead of a table

> [!definition] Function approximation
> Learn a parameterised function $f(x;\theta)$ of some input $x$, and train the parameters $\theta$ so that $f$ accurately approximates the optimal function $f^*$. $f$ can be a [[Value Function|value function]] (state in, expected return out) or a [[Policy|policy]] (state in, action or action probabilities out).

All inputs share the parameters, so an update for one input also changes the output for similar inputs: the generalisation the table lacked. What a network learns about the coin's position applies under every lamp pattern, and it can give the lamps little weight.

- **Linear function** ([[Linear Function Approximation]]): a weighted sum of state features. Simple, generalises, resists overfitting. Problem: high bias, not expressive enough.
- **Deep neural network** ([[Neural Network Function Approximation]]): a universal method. Learns feature representations of states automatically, represents non-linear, complex functions, and handles high-dimensional data such as images ([[Convolutional Neural Networks|CNNs]]) and sequences of observations ([[Recurrent Neural Network (RNN)|RNNs]], which carry a hidden state). This is [[Deep Reinforcement Learning|deep reinforcement learning]].

The price of a network: less stable training (next two sections), no general convergence guarantee (it can diverge), and more computation per decision.

## Deep Q-learning: Q-learning with a network

Replace the table by a network $Q(s,a;\theta)$. The standard design takes **the state as input and outputs one Q-value per discrete action**: one forward pass gives every action's value, and the greedy action is the largest output. (A network that outputs a probability per action is a policy network.)

The table update becomes a regression: minimise the squared difference between target and estimate.

> [!formula] Deep Q-learning loss and target
> $$\mathcal{L}(\theta) = \big(y_t - Q(s_t,a_t;\theta)\big)^2, \qquad y_t = \begin{cases} r_t & \text{if } s_{t+1} \text{ is terminal}\\ r_t + \gamma\max_{a'}Q(s_{t+1},a';\theta) & \text{otherwise}\end{cases}$$
> - $y_t$: the target, as in Q-learning. After a terminal state no reward follows, so it is just $r_t$.
> - $Q(s_t,a_t;\theta)$: the network's estimate for the action taken.

$\theta$ is updated by gradient descent on $\mathcal{L}$. The target is treated as a fixed number: **no gradient flows through $y_t$**, only through the estimate. Actions are still chosen ε-greedily.

## What breaks: moving targets and correlated data

> [!definition] Moving target problem
> The targets depend on $\theta$, so any update to $\theta$ changes the targets. Because a network generalises, targets move faster than in tabular methods, where one state-action value changes at a time, when visited. Learning chases a moving goal and can oscillate or diverge.

> [!definition] Correlation of consecutive experience
> Training data are not independent and identically distributed (i.i.d.): consecutive steps look alike and cover a small region of the state space. This can make the value function overfit to recent experience and cause catastrophic forgetting of previously learned estimates.

## DQN: a target network and a replay buffer

**[[Target Network|Target network]], against moving targets.** Keep a copy of the network, $\bar\theta$ (also written $\theta^-$), and compute targets with it: $y_t = r_t + \gamma\max_{a'}Q(s_{t+1},a';\bar\theta)$. The copy gets no gradient steps; every so many steps it is overwritten, $\bar\theta \leftarrow \theta$. In between, the targets stay fixed.

**[[Experience Replay|Replay buffer]], against correlated data.** Store every transition $(s_t,a_t,r_t,s_{t+1})$, and compute each loss on a minibatch sampled uniformly from the buffer. A random minibatch mixes many episodes, so its samples are much less correlated. Extra advantages: each transition is re-used for several updates, and batches give more stable gradients.

The two fixes solve different problems, so DQN needs both.

> [!definition] DQN
> Deep Q-Networks (DQN) is a foundational deep RL algorithm based on Q-learning: **Q-learning + target network + replay buffer**. See [[Deep Q-Network (DQN)]].

```pseudo
Algorithm: DQN
Initialise Q-network θ, target network θ̄ ← θ, empty replay buffer D
For each time step t:
  Choose a_t ε-greedily from Q(s_t, · ; θ)
  Apply a_t, observe r_t and s_{t+1}, store (s_t, a_t, r_t, s_{t+1}) in D
  Sample a random minibatch of transitions (s, a, r, s') from D
  For each: y ← r if s' is terminal, else r + γ · max_a' Q(s', a'; θ̄)
  Gradient step on the mean of (y − Q(s, a; θ))²       // no gradient through y
  Every so many steps: θ̄ ← θ
```

**DQN is off-policy** ([[Off-Policy Learning]]). The buffer holds actions chosen by older ε-greedy versions of the agent, and the max in the target estimates the greedy policy's values whatever policy chose them. That is what makes replay possible.

## What a value-based agent cannot express

- **Limited policies.** ε-greedy over three actions gives the greedy action $1 - \varepsilon + \varepsilon/3$ and each other action $\varepsilon/3$: one favourite plus uniform noise. A policy like $(0.4, 0.1, 0.5)$ is out of reach. In games good policies are often mixed (matching pennies needs 50/50).
- **Continuous actions.** DQN has one output per discrete action, and a max over a continuous action (a steering angle) is an optimisation problem at every step. It is easier to learn a policy that outputs a continuous value.

So policy gradient methods use a network to represent the policy directly.

## Policy gradients: learn the policy directly

In a [[Policy Gradient Methods|policy gradient method]] a network with parameters $\phi$ outputs a preference per action, and a softmax turns them into probabilities $\pi(a \mid s;\phi)$. Any distribution is reachable, and the agent explores by sampling. For continuous actions the network can output, say, the mean of a distribution.

The aim: change $\phi$ so that actions leading to better outcomes become more likely. Differentiating the expected return directly would need to know how changing $\phi$ changes which states are visited, which requires the transition model. The [[Policy Gradient Theorem|policy gradient theorem]] removes that term: the gradient is an average, over the current policy's own experience, of $Q^\pi(s,a)\,\nabla_\phi\log\pi(a \mid s;\phi)$. Here $\nabla_\phi\log\pi$ is the direction that makes $a$ more likely in $s$, and $Q^\pi(s,a)$, how good $a$ is, weights it. So samples can estimate the gradient without a model.

$Q^\pi$ still has to be estimated: by Monte Carlo returns (REINFORCE) or by another network (actor-critic: A2C, PPO). Because the average is over the current policy's own experience, these methods are **on-policy** ([[On-Policy Learning]]): data is used to update the policy that generated it, then thrown away.

## REINFORCE: estimate Q with the return

[[REINFORCE]] plays a whole episode, then uses the discounted return from each step, $G_t = \sum_{\tau \ge t}\gamma^{\tau-t}r_\tau$, as its estimate of $Q^\pi(s_t,a_t)$ ([[Monte Carlo Methods|Monte Carlo]]).

> [!formula] REINFORCE gradient estimate
> $$G_t\,\nabla_\phi\log\pi(a_t \mid s_t;\phi)$$
> $G_t$ is a sampled number with no gradient of its own; it scales the push towards the action taken.

Advantages: simple, no critic, and with exact Monte Carlo returns the gradient estimate is **unbiased**.

**Problem: high variance.** Different trajectories from the same state can give very different returns (random actions, random transitions, long episodes), so gradients are unstable and learning is slow. It also has to wait until the episode ends.

**Fix: a [[Baseline|baseline]].** Use $G_t - b(s_t)$. A state-dependent baseline leaves the expected gradient unchanged but reduces variance: if all returns from a state are around 100, the raw estimate pushes every action up; with $b(s) \approx 100$, better-than-usual actions go up and worse ones down. A learned $V(s)$ as baseline leads to actor-critic methods.

## Actor-critic methods and A2C

[[Actor-Critic|Actor-critic]] methods learn the policy (the **actor**) and a bootstrapped estimate of the state value (the **critic**), and use the critic in place of the Monte Carlo return.

- **Benefits:** single steps replace full rollouts (efficient training), and variance is lower than Monte Carlo.
- **Costs:** bias from bootstrapping (the critic's estimates may be inaccurate), an extra component to train, and performance depends on critic and rollout settings (extra hyperparameters).

[[Advantage Actor-Critic (A2C)|A2C]] guides the policy gradient with the [[Advantage Function|advantage]]:

> [!formula] Advantage and A2C updates
> $$\text{Adv}(s,a) = Q(s,a) - V(s) \;\approx\; r + \gamma V(s') - V(s)$$
> - how much better taking $a$ is than just following the policy; $r + \gamma V(s')$ estimates $Q(s,a)$, $V(s)$ is the baseline (at a terminal $s'$: $r - V(s)$)
> - **actor:** minimise $-\text{Adv}(s,a)\log\pi(a \mid s;\phi)$. Positive advantage: the action did better than expected and becomes more likely. Negative: less likely.
> - **critic:** minimise $\big(y - V(s;\theta)\big)^2$ with $y = r + \gamma V(s';\theta)$, a regression as in DQN

Both networks update every step. A2C is on-policy, so it uses no replay buffer.

## PPO: limit how far one update moves the policy

A single policy gradient update can still shift the policy a lot, and a bad step changes the data collected next: a source of instability. [[PPO|Proximal policy optimisation (PPO)]], one of the most used policy gradient methods, limits how much the policy can change in one update.

> [!formula] PPO clipped loss
> $$\mathcal{L}(\phi) = -\min\Big(\rho\,\text{Adv},\;\; \text{clip}\big(\rho,\,1-\epsilon,\,1+\epsilon\big)\,\text{Adv}\Big), \qquad \rho = \frac{\pi(a \mid s;\phi)}{\pi(a \mid s;\phi_{\text{old}})}$$
> - $\rho$: new over old probability of the action; $\rho = 1$ means no change
> - clip keeps $\rho$ in $[1-\epsilon, 1+\epsilon]$ ($\epsilon$ is unrelated to ε-greedy); the min takes the more pessimistic term

Once a good action's ratio exceeds $1+\epsilon$ (or a bad one's drops below $1-\epsilon$), the clipped term is a constant, so no gradient pushes it further. Inside the range the update is the ordinary advantage-weighted one.

Pros: often stable updates, and it can reuse a batch for several optimisation passes because the policy stays close to the one that collected it. Cons: more complicated than REINFORCE, still generally on-policy (cannot reuse old data indefinitely), and clipping is no strict guarantee against large or harmful changes.

## Comparing the deep RL methods

| | DQN | REINFORCE | A2C | PPO |
|---|---|---|---|---|
| Learns | Q-values | policy | policy + critic | policy + critic |
| Data | off-policy, replay | on-policy, full episodes | on-policy, single steps | on-policy, few passes |
| Strength | re-uses data | simple, unbiased | lower variance | stable updates |
| Weakness | discrete actions, ε-greedy | high variance | critic bias | more complex |

## Deep MARL: who sees what, and when

Lecture 3's setting and challenges stay ([[Stochastic Game]]; non-stationarity, equilibrium selection, [[Multi-Agent Credit Assignment|credit assignment]], curse of dimensionality). Deep MARL adds a choice for both training and execution:

> [!definition] Centralised and decentralised
> **Centralised:** use information about the other agents (their observations, actions, policies, values, or a global state) to train an agent. **Decentralised:** condition only on the agent's own individual history.

Three modes ([[Centralised Training with Decentralised Execution]]):

- **Centralised training and execution (CTCE):** Lecture 3's centralised learning, one policy over the joint action space. Challenges: needs a joint reward, does not scale with the number of agents, and agents may be distributed by design.
- **Decentralised training and execution (DTDE):** Lecture 3's independent learning; each agent has only its own model and observations. Challenges: no information about the others, non-stationarity, hard to coordinate, unstable learning.
- **Centralised training with decentralised execution (CTDE):** train with shared information or a centralised critic, then each agent acts on its local observations only. The topic of [[AGAI-L05 - Deep MARL, Part II|Lecture 5]].

> [!example] Choosing a mode
> Autonomous cars cannot exchange all sensor data while driving, but a simulator provides all data in training: CTDE. CTCE would need the impossible exchange on the road; DTDE would waste the simulator's information.

## Independent deep learners

Independent learning (DTDE) is commonly used and competitive, and often it is the setting we have, as when each organisation programs its own agent (Assignment 1). Each method above has an independent version (IDQN, independent REINFORCE, IA2C, IPPO) in which each agent treats the others as part of the environment ([[Independent Learning]]).

### Independent DQN and its stale replay buffer

In [[Independent DQN|IDQN]] each agent trains its own Q-network, target network and replay buffer on its own observations and rewards.

**New challenge: stored experience may no longer be relevant once the other agents change their policies.** The agent trains on an environment that no longer exists. Being off-policy does not help: it covers the agent's own old actions, while the others' behaviour is part of its environment.

Remedies:

- **A smaller replay buffer**, holding only recent transitions from policies close to the others' current ones. A larger buffer makes it worse.
- **Importance sampling weights** (MARL book, Ch 9): store all agents' action probabilities with each transition and, on replay, weight it by how likely the others are to act that way now.

The target network and the discount factor have nothing to do with the age of the data.

### Independent policy gradient methods

**Advantage over IDQN: they always learn from the updated policies of the other agents.** Being on-policy, each update uses recent experience from the others' current policies and then discards it, so independent REINFORCE adapts faster to other learners than IDQN.

**Challenge:** independent MARL exacerbates sources of learning instability, such as the variance of Monte Carlo returns, which now also vary as the others change.

- **Independent REINFORCE:** simple, decentralised, no critic, local or shared team rewards. High variance, worst with long episodes and shared team rewards.
- **IA2C:** each agent has an actor and its own critic; lower variance, more frequent updates. But the critic learns in a changing environment, and with a shared team reward it does not solve credit assignment.
- **IPPO:** each agent clips its own update; often stable, several passes, acts on local information. But it does not guarantee that the joint policy changes safely, the others' updates keep the environment non-stationary, and it remains generally on-policy.

## What to be able to answer

- Why tables fail, and the lamps arithmetic; what a network gains and costs.
- DQN: input and output, terminal target, no gradient through the target, what each fix does, off-policy.
- Why policy gradients; the REINFORCE gradient, the advantage, the A2C updates, the PPO clip; variance and baselines.
- Which training mode fits a scenario; IDQN's stale buffer and remedies; why independent policy gradients adapt faster.

## Exam questions

> [!exam]- Mock Q21 (open, 10 points): Tabular Q-learners play the Coin Game, whose table has 1,458 rows. Each of k random lamps doubles the number of rows (as lamps can be on or off). Training plays 300 updates, each with 32 parallel games of 16 steps, and every step gives one decision per agent. Coins per episode are 14.5 with no lamps and 4.9 with 8 lamps, while random agents collect 3.7. (a) Compute the number of rows N for k = 0 and k = 8, and the budget B of decisions per agent. (3 points) (b) Use these numbers to explain the drop in performance. Why can't agents with tabular learning discover that the lamps do not matter? (3 points) (c) Explain how a neural network avoids this problem. Name one new challenge it introduces. (4 points)
> **Key points:** $N = 1{,}458$ and $1{,}458 \cdot 2^8$, $B = 300 \cdot 32 \cdot 16$; rows grow while the budget stays, so most rows are never visited and unvisited rows hold zeros (random play); rows that differ in one lamp share nothing, so each lesson is relearned per lamp pattern; a network maps observations to values or a policy with shared weights, so learning about the coin's position applies to all lamp patterns and the lamps get little weight; new challenge: instability from moving targets or correlated data, divergence, no convergence guarantee, more computation.
>
> **(a)** $N = 1{,}458$ for $k = 0$, and $N = 1{,}458 \cdot 2^8 = 373{,}248$ for $k = 8$. $B = 300 \cdot 32 \cdot 16 = 153{,}600$ decisions per agent.
>
> **(b)** With 8 lamps, most rows are never visited, and the others are visited only a few times (about 0.4 decisions per row, against about 105 without lamps). An unvisited row still holds zeros, so the agent acts at random there. The number of rows grows but the budget remains, so the probability of visiting each state becomes too low, and performance drops from 14.5 coins to 4.9, close to random (3.7). Two rows that differ in one lamp share nothing, so each lesson must be learned again for each of the other lamp patterns. That is why the table cannot discover that the lamps do not matter.
>
> **(c)** A network computes values or a policy as a function of the observation, with shared weights. What it learns about the coin's position applies to all lamp patterns, and it can give the lamps little weight. New challenges: instability from moving targets or correlated data, possible divergence, no general convergence guarantee, and more computation needed.

> [!exam]- A deep Q-learner on the Coin Game first computes its targets with the network it is training and learns from each transition right after it happens. Its values oscillate, and after long stretches in one part of the grid it forgets what it knew elsewhere. (a) Name and explain the two problems. (4 points) (b) Explain how DQN's target network and replay buffer each address one of them. (4 points) (c) Why may DQN learn from buffer transitions whose actions were chosen by older versions of the agent? (2 points)
> **Key points:** moving targets: targets depend on $\theta$, and a network update changes many values at once, the targets included; correlated experience: consecutive transitions are not i.i.d., causing overfitting to recent data and catastrophic forgetting; target network: copy $\bar\theta$ updated less often keeps targets fixed; replay buffer: uniform minibatches from many episodes decorrelate samples (and re-use data); off-policy: the max target estimates greedy values whatever policy acted.
>
> **(a)** Moving targets: the target $r + \gamma\max_{a'}Q(s',a';\theta)$ uses the same parameters that are updated. A network generalises, so each step changes many values, including the targets, while a table changes one entry at a time. Learning chases a moving goal and oscillates. Correlated experience: consecutive transitions are similar and come from one region, so they are not i.i.d. The network overfits to recent experience and catastrophically forgets earlier estimates.
>
> **(b)** The target network is a copy $\bar\theta$ used only to compute the targets and overwritten with $\theta$ every so many steps, so the targets stay fixed in between. The replay buffer stores all transitions and trains on minibatches sampled uniformly from it, which mixes many episodes and regions; it also re-uses each transition and gives more stable batch gradients. Each fix solves only its own problem, so DQN uses both.
>
> **(c)** DQN is off-policy: the max in its target estimates the greedy policy's values, whichever policy chose the stored actions.

> [!exam]- A robot arm must choose a continuous torque, and a good policy may need to be stochastic. (a) Give two reasons why DQN is a poor fit. (2 points) (b) Give the quantity REINFORCE uses to estimate the policy gradient, and explain its main weakness. (3 points) (c) How does A2C reduce this weakness, and what does it cost? (3 points) (d) What problem does PPO's clipping address? (2 points)
> **Key points:** DQN needs one output per discrete action and a max over actions; ε-greedy gives only one favourite action plus uniform noise; REINFORCE uses $G_t\nabla_\phi\log\pi(a_t \mid s_t;\phi)$, unbiased but high variance; A2C replaces $G_t$ by a critic's advantage $r + \gamma V(s') - V(s)$, lower variance and single-step updates, but bias from bootstrapping and an extra network with hyperparameters; PPO clips the new-to-old probability ratio so one update cannot move the policy too far.
>
> **(a)** DQN has one output per discrete action, and a max over a continuous torque is an optimisation problem at every step. Its ε-greedy policy is always one favourite action plus uniform noise, so it cannot represent an arbitrary stochastic policy.
>
> **(b)** $G_t\,\nabla_\phi\log\pi(a_t \mid s_t;\phi)$, with $G_t$ the Monte Carlo return. It is unbiased, but returns from the same state vary a lot between trajectories, so the gradient estimates have high variance and training is unstable and slow. It also waits for the end of each episode.
>
> **(c)** A2C learns a critic $V$ and weights the log-probability by the advantage $r + \gamma V(s') - V(s)$, where $V(s)$ acts as a baseline. This lowers the variance and allows an update after each step. The cost is bias from bootstrapping on possibly inaccurate critic estimates, plus an extra component to train with its own settings.
>
> **(d)** A single policy gradient update can shift the policy a lot, which destabilises training. PPO clips the ratio of new to old action probability to $[1-\epsilon, 1+\epsilon]$, so there is no gradient pushing the policy further once it has moved that far.

> [!exam]- In a Coin Game, your agent learns while the opponent, built by another team, is also learning. (a) Which training and execution mode is this, and why? (2 points) (b) Your agent uses IDQN. Explain why its replay buffer becomes a problem, and give two remedies. (4 points) (c) Why can independent REINFORCE adapt faster to the opponent, and what is its main weakness here? (4 points)
> **Key points:** DTDE, since nobody can train both agents centrally and each uses only its own observations; buffer transitions reflect the opponent's old policy, so the agent trains on an environment that no longer exists, and off-policy learning only covers its own old actions; smaller buffer, importance sampling weights from stored action probabilities; REINFORCE is on-policy and learns from fresh experience of the opponent's current policy; independent learning makes the already high variance of Monte Carlo returns worse.
>
> **(a)** Decentralised training and execution (independent learning): the opponent is not ours to train, so each agent trains and acts on its own observations and treats the other as part of the environment.
>
> **(b)** The buffer stores transitions generated while the opponent used older policies. Once the opponent changes, the same action in the same situation can lead elsewhere, so the agent keeps learning from outdated experience. DQN being off-policy does not fix this, because the opponent's behaviour is part of the environment. Remedies: a smaller replay buffer, which keeps only recent transitions; or storing the agents' action probabilities with each transition and reweighting replayed transitions with importance sampling weights.
>
> **(c)** Independent REINFORCE is on-policy: each update uses recent episodes, generated by the opponent's current policy, and discards them, so nothing goes stale. Its weakness is variance: Monte Carlo returns are already noisy, and the opponent's changing policy makes them even less consistent.

## Flashcards

> [!card]- Why can a tabular Q-learner not generalise?
> Each value is updated in isolation; nothing learned in one row reaches another.

> [!card]- What does a tabular Q-learner do in a state it never visited?
> Its values are still zero, so it acts at random.

> [!card]- What happens to a Q-table when an irrelevant binary input is added to the state?
> The rows double, and rows that differ in that input share nothing.

> [!card]- How does a neural network avoid the irrelevant-lamps problem?
> Shared weights: what it learns about positions applies under every lamp pattern.

> [!card]- Name the costs of replacing a Q-table with a neural network.
> Instability, possible divergence, no convergence guarantee, more computation.

> [!card]- What are the input and output of a standard DQN?
> Input the state; output one Q-value per discrete action.

> [!card]- What is the DQN target when the next state is terminal?
> $y = r$.

> [!card]- Which part of the deep Q-learning loss is differentiated?
> Only the estimate $Q(s,a;\theta)$; the target $y$ is treated as a constant.

> [!card]- What is the moving target problem in deep Q-learning?
> Targets depend on $\theta$, so every update to $\theta$ also moves the targets.

> [!card]- What does training on correlated consecutive experience cause?
> Overfitting to recent experience and catastrophic forgetting of earlier estimates.

> [!card]- Which DQN component counters moving targets?
> The target network, a copy updated less often.

> [!card]- How does DQN's replay buffer reduce correlation?
> It samples minibatches uniformly from many stored transitions.

> [!card]- Why can DQN learn from a replay buffer of old actions?
> It is off-policy: the max target ignores which policy chose the action.

> [!card]- Why does DQN struggle with continuous actions?
> One output per discrete action; a max over infinitely many actions is an optimisation per step.

> [!card]- Why does the policy gradient theorem need no environment model?
> It has no term for how state visit probabilities change.

> [!card]- What quantity does REINFORCE use to estimate the policy gradient at time $t$?
> $G_t\nabla_\phi\log\pi(a_t \mid s_t;\phi)$, with $G_t$ the Monte Carlo return.

> [!card]- Why does REINFORCE have high variance?
> Returns from the same state differ a lot between trajectories.

> [!card]- What does subtracting a state baseline do to REINFORCE's gradient?
> Lowers its variance without changing its expected value.

> [!card]- What do actor-critic methods lose against REINFORCE?
> Unbiasedness: bootstrapping from the critic adds bias.

> [!card]- How does A2C estimate the advantage in one step?
> $r + \gamma V(s') - V(s)$.

> [!card]- In A2C, what does a positive advantage do to the action's probability?
> Raises it: the action did better than expected.

> [!card]- What problem does PPO's clipping address?
> One update can shift the policy too far and destabilise training.

> [!card]- What is the ratio $\rho$ that PPO clips?
> New over old probability of the action, clipped to $[1-\epsilon, 1+\epsilon]$.

> [!card]- Which training mode trains with shared information but acts on local observations?
> Centralised training with decentralised execution (CTDE).

> [!card]- Name the challenges of centralised training and execution.
> Needs a joint reward, does not scale, agents may be distributed.

> [!card]- Why does IDQN's replay buffer go stale?
> Its transitions reflect the other agents' old policies.

> [!card]- Why do independent policy gradient methods adapt faster than IDQN?
> They are on-policy and learn from the others' current policies.

> [!card]- True or false: the replay buffer stores earlier copies of the network weights.
> **False.** It stores transitions; the weight copy is the target network.

> [!card]- True or false: with exact Monte Carlo returns, REINFORCE's gradient estimate is biased.
> **False.** It is unbiased; its problem is high variance.

> [!card]- True or false: PPO's clipping strictly guarantees that no update is harmful.
> **False.** It limits the ratio but is no strict guarantee.

> [!card]- True or false: a larger replay buffer reduces outdated experience in IDQN.
> **False.** A smaller buffer does; a larger one keeps older experience.

> [!card]- Multiple selection: which are on-policy? (A) DQN. (B) REINFORCE. (C) A2C. (D) PPO.
> **B, C and D.** A: DQN is off-policy and replays old transitions.

> [!card]- Multiple selection: which are disadvantages of A2C compared with REINFORCE? (A) The critic is an extra component to train. (B) Bootstrapping can introduce bias. (C) It must wait until an episode ends. (D) Performance depends on critic and rollout settings.
> **A, B and D.** C: A2C can update after a single step.

> [!card]- Multiple selection: which are challenges of decentralised training and execution? (A) Non-stationarity. (B) Joint action space growing exponentially. (C) Hard to coordinate. (D) Unstable learning.
> **A, C and D.** B: that is a problem of centralised training and execution.

## Links

- **Course:** [[AGAI - Overview|Course overview]]
- **Previous lecture:** [[AGAI-L03 - Intro to Multi-Agent Reinforcement Learning]] (tabular Q-learning, centralised and independent Q-learning, the MARL challenges)
- **Next lecture:** [[AGAI-L05 - Deep MARL, Part II]] (CTDE, centralised critics, parameter and experience sharing)
- **Lab:** [[AGAI-Lab02 - Learning Agents]] (the Coin Game, irrelevant lamps, tabular Q-learning against PPO)
- **RL background in depth:** [[RL-L05 - Tabular to Approximation]], [[RL-L08 - Deep RL Value-Based]] (DQN), [[RL-L09 - Policy Gradient Methods]] (policy gradient theorem, REINFORCE, actor-critic), [[RL-L10 - Advanced Policy Search]] (PPO)
- **Source:** Agentic Artificial Intelligence, Lecture 4, "Intro to Deep RL and Deep MARL", Roman Chiva Gil, MSc AI, UvA, 5 November 2026
- **Reading:** Albrecht, Christianos and Schäfer, *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches* (MIT Press, 2024), Chapter 8 and Chapter 9.1 and 9.3
- **Mock exam:** Mock Midterm 2026, questions 13 to 21 (Deep RL) and 22, 23, 26 (Deep MARL)
