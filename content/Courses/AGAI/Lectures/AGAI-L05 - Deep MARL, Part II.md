---
type: lecture
course: 5204AGAI6Y
week: 3
lecture: 5
date: 2026-11-10
book_sections: ["Albrecht, Christianos and Schäfer, MARL, Ch 9"]
topics:
  - "[[Centralised Training with Decentralised Execution]]"
  - "[[Policy Gradient Methods]]"
  - "[[Advantage Actor-Critic (A2C)]]"
  - "[[Multi-Agent Credit Assignment]]"
  - "[[Individual-Global-Max Property]]"
  - "[[Value Decomposition Networks (VDN)]]"
  - "[[QMIX]]"
  - "[[Agent Modelling]]"
  - "[[Parameter Sharing]]"
  - "[[Monte Carlo Tree Search (MCTS)]]"
  - "[[AlphaGo Zero]]"
  - "[[Policy Space Response Oracles (PSRO)]]"
status: complete
---

# AGAI-L05: Deep MARL, Part II

> [!abstract] In short
> Independent deep learners ignore each other, and one central learner over the joint action does not scale. How can agents use information about each other while learning and still act alone? Mainly by centralised training with decentralised execution (CTDE): a critic that sees everyone's information trains decentralised actors, and value decomposition (VDN, QMIX) learns one utility per agent so that each agent's greedy action adds up to the best joint action (the IGM property). Agents can also model the others' policies, or shape how the others learn (LOLA). Similar agents can share one network (parameter sharing) or one replay buffer (experience sharing). In large two-player zero-sum games, Monte Carlo tree search plus self-play (AlphaZero) trains one policy against itself, and PSRO (from the MARL book) trains populations of policies.
>
> Feeds mock questions 23 (training modes), 24 (parameter sharing), 25 (agent models), 27 (experience against parameter sharing), 28 (PSRO meta-game) and the open question 29 (homogeneous agents).

## What this builds on

[[AGAI-L04 - Deep RL and Intro to Deep MARL|Lecture 4]] ran one deep learner ([[Deep Q-Network (DQN)|DQN]], [[Advantage Actor-Critic (A2C)|A2C]], PPO) per agent, each treating the others as part of the environment. This [[Independent Learning|independent learning]] ([[Independent DQN|IDQN]], IA2C, IPPO) scales, but it is non-stationary and cannot coordinate.

Notation: agent $i$ usually sees only part of the state, so it conditions on its **observation history** $h_i$ (often summarised by a [[Recurrent Neural Network (RNN)|recurrent network]]); $h$ is the joint history of all agents and $\hat h$ the full history including states; $z$ is **centralised information**, anything beyond $h_i$ available during training (global state, joint history, others' observations); $a_{-i}$ are the actions of all agents except $i$.

## Who sees what, and when: training and execution modes

Two distinctions. **Training** is optimising the policy and exploring; **execution** is selecting actions once the policy is in use. A phase is **centralised** when agent $i$ uses centrally shared information of others (joint history, observations, global state, joint policy), and **decentralised** when agent $i$ relies only on its own experience. Combining them gives three modes.

**Centralised training and execution (CTCE).** Shared information is used throughout: local observation histories, agent policies, learned value functions. Its strength is coordination, and in a partially observable stochastic game (POSG, a [[Stochastic Game|stochastic game]] where each agent sees only part of the state) pooling everyone's observations removes much of the [[Partial Observability|partial information]]. Its weaknesses: the joint action space grows exponentially with the number of agents, and agents may be distributed entities by design that cannot share a controller.

**Decentralised training and execution (DTDE).** Agents never rely on centrally shared information: independent learning, and [[Agent Modelling|agent modelling]] from an agent's own observations. It scales and applies to distributed agents that cannot communicate. But training suffers from non-stationarity, agents can settle in suboptimal equilibria, and an agent cannot tell which changes are due to other agents.

**Centralised training with decentralised execution (CTDE).** Shared information is used **only during training**. Each agent ends up with a policy that needs only its own history, so execution needs no central controller and suits distributed settings with limited runtime communication, while training accounts for the interactions between agents and improves coordination. Its weaknesses: centralised training can still be a bottleneck, and coordination is hard when local information is not enough at execution time. See [[Centralised Training with Decentralised Execution]].

> [!example] Choosing a mode
> Autonomous cars cannot exchange all their sensor data while driving, but a simulator can provide all data during training. Execution has to be decentralised, training can be centralised: CTDE.

CTDE can be applied to both families from Lecture 4: policy gradient methods (centralised critics) and value-based methods (value decomposition).

## Centralised critics: CTDE for policy gradient

The single-agent [[Policy Gradient Theorem|policy gradient theorem]] says which direction to move the policy parameters $\phi$:

$$\nabla_\phi J(\phi) \propto \mathbb{E}_{s \sim P(\cdot \mid \pi),\, a \sim \pi(\cdot \mid s;\phi)}\Big[Q^\pi(s,a)\,\nabla_\phi \log\pi(a \mid s;\phi)\Big]$$

$\nabla_\phi\log\pi$ points towards making action $a$ more likely, and $Q^\pi(s,a)$ scales that push by how valuable $a$ is: highly valuable actions become more probable. With several agents the value of my action depends on what the others do at the same moment, which gives the multi-agent version:

> [!formula] Multi-agent policy gradient theorem
> $$\nabla_{\phi_i} J(\phi_i) \propto \mathbb{E}_{\hat h \sim P(\hat h \mid \pi),\; a_i \sim \pi_i,\; a_{-i} \sim \pi_{-i}}\Big[Q_i^\pi\big(\hat h, \langle a_i, a_{-i}\rangle\big)\,\nabla_{\phi_i}\log\pi_i(a_i \mid h_i;\phi_i)\Big]$$
> - $\phi_i$: agent $i$'s policy parameters; $\pi_{-i}$: the other agents' policies
> - $\hat h \sim P(\hat h \mid \pi)$: full histories as generated by the **joint** policy
> - $Q_i^\pi(\hat h, \langle a_i, a_{-i}\rangle)$: agent $i$'s expected return given the full history and the **joint action**

The **evaluation** of an action uses the joint history and joint action, and is needed only to compute the gradient during training. The **policy** $\pi_i(a_i \mid h_i;\phi_i)$ conditions only on agent $i$'s own history and is what the agent carries into execution. That split is exactly CTDE.

### The centralised actor-critic

- **Actor:** stays decentralised, $\pi(\cdot \mid h_i^t;\phi_i)$, because it is used for execution.
- **Critic:** uses centralised information, $V(h_i, z;\theta_i)$, because it is only used for training.

The critic is an ordinary network whose input is $h_i$ concatenated with $z$. The actor never sees $z$, but its update uses it indirectly: the advantage in the actor loss comes from the centralised critic.

```pseudo
Algorithm: Centralised A2C with synchronous environments (n agents, K parallel environments)
──────────────────────────────────────────────────────────────────────────────────────────
Initialize n actor networks φ_1, ..., φ_n and n critic networks θ_1, ..., θ_n
Initialize K parallel environments
For time step t = 0, 1, ...:
  Get observations o_i^{t,k} for each agent i and environment k
  Get centralised information z^{t,k} for each environment k
  Sample actions a_i^{t,k} ~ π(· | h_i^{t,k}; φ_i)            // actors: own history only
  Apply actions; collect rewards r_i^{t,k}, next observations, next z^{t+1,k}
  For each agent i:
    For each environment k:
      y_i^{t,k} ← r_i^{t,k} + γ V(h_i^{t+1,k}, z^{t+1,k}; θ_i)   // critic target; just r_i^{t,k} if terminal
      Adv^{t,k} ← y_i^{t,k} − V(h_i^{t,k}, z^{t,k}; θ_i)          // advantage
    Actor loss  L(φ_i) ← −(1/K) Σ_k Adv^{t,k} · log π(a_i^{t,k} | h_i^{t,k}; φ_i)
                // Adv is a constant; the minus makes minimising raise log π where Adv > 0
    Critic loss L(θ_i) ← (1/K) Σ_k (y_i^{t,k} − V(h_i^{t,k}, z^{t,k}; θ_i))²
    Update φ_i and θ_i by minimising their losses
```

The actor loss is the sampled version of the multi-agent policy gradient, with the advantage standing in for $Q_i^\pi$. Multi-agent [[PPO]] (MAPPO) uses the same centralised value critic.

**Why is a CTDE version of [[REINFORCE]] rare?** REINFORCE has no critic: it weights the policy gradient by the sampled return. Its only learned component is the policy, which must stay decentralised for execution, so centralised information has nowhere to go. CTDE needs a training-only component, and in policy gradient methods that is the critic.

### Value critics and action-value critics

A **centralised value critic** $V(h_i, z;\theta_i)$ outputs one number (used by MAPPO and the centralised A2C above). A **centralised action-value critic** also takes the other agents' actions $a_{-i}$ as input and outputs one value per own action $a_i$. Because it shows what would have happened had agent $i$ acted differently while the others kept their actions, it supports **counterfactual reasoning**. (From the MARL book: COMA, counterfactual multi-agent policy gradient, uses such a critic for credit assignment. Its advantage compares the value of agent $i$'s action with the policy-weighted average over its alternative actions, the others' actions held fixed.)

## What a centralised value function still cannot do

Centralised action-value functions tackle three MARL dilemmas: **non-stationarity** (the other agents' information is an input, so their changes are no longer invisible), **coordination**, and **partial observability**. One key challenge remains in cooperative, common-reward games: **credit assignment** ([[Multi-Agent Credit Assignment]]). From one shared reward and one team value it is hard to learn the team's best joint policy, because the team value never says which agent's action made the difference. And decentralised action selection is still needed at execution.

The ideal: learn an **individual** action-value function per agent such that, if each agent follows its own one greedily, the outcome equals that of the optimal joint value function.

## Splitting the team value into utilities

A **centralised action-value function** is the expected return given everyone's histories, the centralised information and the joint action:

$$Q(h, z, a;\theta) = \mathbb{E}\Big[\sum_{\tau = t}^{\infty}\gamma^{\tau - t}r^\tau \,\Big|\, h^t, z^t, a^t\Big]$$

**Value decomposition** learns instead one **individual utility function** per agent, conditioned only on that agent's history and action:

$$Q(h_i, a_i;\theta_i) \approx \mathbb{E}\Big[\sum_{\tau = t}^{\infty}\gamma^{\tau - t}r^\tau \,\Big|\, h_i^t, a_i^t\Big]$$

It is called a **utility** because, once trained through the decomposition, it need not equal any actual expected return: only its role in rebuilding the team value matters.

> [!example] A linear game and its decomposition
> Two agents each choose A or B and share the reward:
>
> | | A | B |
> |---|---|---|
> | **A** | 1 | 5 |
> | **B** | 5 | 9 |
>
> VDN (below) learns the utilities $Q_1(A) = 0.88$, $Q_1(B) = 4.88$ and $Q_2(A) = 0.12$, $Q_2(B) = 4.12$, which reproduce every cell as a sum: $0.88 + 0.12 = 1$, $0.88 + 4.12 = 5$, $4.88 + 0.12 = 5$, $4.88 + 4.12 = 9$. Agent 1's greedy choice is B (4.88 against 0.88), agent 2's is B (4.12 against 0.12), and (B, B) is the best joint action. The split is not unique: adding $c$ to both of agent 1's utilities and subtracting $c$ from agent 2's changes nothing.

How do we judge whether a set of utilities is good? The criterion is the **Individual-Global-Max property**.

> [!definition] Individual-Global-Max (IGM) property
> An algorithm satisfies IGM if
> $$\arg\max_{a_1, \dots, a_n} Q(s, a_1, \dots, a_n) = \Big(\arg\max_{a_1}Q_1(o_1, a_1),\; \dots,\; \arg\max_{a_n}Q_n(o_n, a_n)\Big)$$
> The **joint action maximising the joint action-value function** equals the **set of individual actions maximising the individual action-value functions**. See [[Individual-Global-Max Property]].

IGM has two implications:

1. **Decentralised execution:** each agent can follow the greedy policy with respect to its own utility, and together they play the greedy joint action.
2. **Efficiency:** the greedy joint action is obtained by computing the greedy individual actions. With 5 agents of 10 actions each, that is 5 maximisations over 10 values instead of one over $10^5$ joint actions. Training needs this too, because every Q-learning target contains a max over the next joint action.

How can the joint value be decomposed in practice? Linearly (VDN), monotonically (QMIX), or beyond both (QTRAN).

## VDN: the team value as a sum

Assume the shared reward is a sum of individual reward terms, $r^t = \bar r_1^t + \dots + \bar r_n^t$, which nobody observes. Then by linearity of expectation:

> [!formula] Linear value decomposition
> $$\begin{aligned}Q(h^t, z^t, a^t;\theta) &= \mathbb{E}_{\hat h^t \sim \Pr(\cdot \mid \pi)}\Big[\sum_{\tau=t}^{\infty}\gamma^{\tau-t}r^\tau \,\Big|\, h^t = \sigma(\hat h^t), z^t, a^t\Big]\\ &= \mathbb{E}_{\hat h^t}\Big[\sum_{\tau=t}^{\infty}\gamma^{\tau-t}\Big(\sum_{i \in I}\bar r_i^\tau\Big) \,\Big|\, h^t, z^t, a^t\Big]\\ &= \sum_{i \in I}\mathbb{E}_{\hat h^t}\Big[\sum_{\tau=t}^{\infty}\gamma^{\tau-t}\bar r_i^\tau \,\Big|\, h^t, z^t, a^t\Big]\\ &= \sum_{i \in I} Q(h_i^t, a_i^t;\theta_i)\end{aligned}$$
> - $I$: the set of agents; $\sigma(\hat h^t)$: the joint observation history extracted from the full history
> - second line: substitute the sum of individual rewards; third line: swap the two sums (linearity of expectation)
> - last line: the modelling step, approximating each agent's share by a utility of its own history and action

[[Value Decomposition Networks (VDN)|Value decomposition networks (VDN)]] learn exactly this: agent-level utilities whose **sum predicts the team value**.

```pseudo
Algorithm: Value decomposition networks (VDN)
─────────────────────────────────────────────
Initialize n utility networks θ_1, ..., θ_n and target networks θ̄_i ← θ_i
Initialize a shared replay buffer D
For time step t = 0, 1, 2, ...:
  Collect current observations o_1^t, ..., o_n^t
  For each agent i:
    With probability ε: random action a_i^t
    Otherwise:          a_i^t ∈ argmax_{a_i} Q(h_i^t, a_i; θ_i)     // own history only
  Apply actions; collect shared reward r^t and next observations
  Store (h^t, a^t, r^t, h^{t+1}) in D
  Sample a mini-batch of B transitions (h^k, a^k, r^k, h^{k+1}) from D
  If s^{k+1} is terminal: y^k ← r^k
  Else:                   y^k ← r^k + γ Σ_i max_{a_i'} Q(h_i^{k+1}, a_i'; θ̄_i)
                          // max of the sum = sum of the maxes: IGM makes this cheap
  Loss L(θ) ← (1/B) Σ_k ( y^k − Σ_i Q(h_i^k, a_i^k; θ_i) )²
  Update all θ_i by minimising L(θ)                     // gradients flow through the sum
  At a set interval, update target networks θ̄_i ← θ_i
```

Every utility is trained only through the sum, against one team target, so the networks themselves discover how to split the credit. VDN satisfies IGM because the maximum of a sum of terms in separate variables is the sum of their maxima. It is CTDE: the loss and buffer are shared, and each agent acts on its own utility.

**The problem: additivity is restrictive.** Some tasks depend on specific combinations of actions, for example two agents that must choose matching actions to succeed. A sum of individual values cannot capture that interaction.

> [!example] A matching task VDN cannot represent (example from outside the course material)
> Reward 10 if both agents pick the same action, 0 otherwise. A sum would need $Q_1(A) + Q_2(A) = Q_1(B) + Q_2(B) = 10$, so the four utilities total 20, and $Q_1(A) + Q_2(B) = Q_1(B) + Q_2(A) = 0$, so they total 0. No sum fits. The least-squares fit values every joint action at 5, so the greedy choice is a tie and the agents learn no reason to match.

## QMIX: any monotonic mix

The point of the decomposition was IGM, and IGM does not need linearity. **Monotonicity is enough**: if raising any agent's utility can never lower the joint value, then each agent maximising its own utility maximises the joint value.

> [!formula] Monotonic value decomposition
> $$Q(h, z, a;\theta) = f_{\text{mix}}\big(Q(h_1, a_1;\theta_1), \dots, Q(h_n, a_n;\theta_n);\,\theta_{\text{mix}}\big)$$
> $$\forall i \in I,\; \forall a \in A:\quad \frac{\partial Q(h, z, a;\theta)}{\partial Q(h_i, a_i;\theta_i)} > 0$$
> - $f_{\text{mix}}$: the **mixing network**, monotonic in every utility
> - VDN is the special case $f_{\text{mix}} = $ sum, so QMIX can represent everything VDN can, and more

[[QMIX]] learns the mixing function. Its parameters $\theta_{\text{mix}}$ are produced by a **hypernetwork** (a network that outputs the weights of another network), $\theta_{\text{mix}} = f_{\text{hyper}}(z;\theta_{\text{hyper}})$, which takes the centralised information $z$ (typically the global state) as input. The mix can therefore change from state to state while staying monotonic in the utilities. (Beyond the lecture: the original QMIX enforces this with non-negative mixing weights, an absolute value on the hypernetwork's outputs. Strictly that gives $\partial Q / \partial Q_i \geq 0$, which also suffices for IGM.)

The algorithm is VDN with the sum replaced by the mix, both in the target and in the estimate:

```pseudo
Algorithm: QMIX (lines that differ from VDN)
────────────────────────────────────────────
Also initialize a hypernetwork θ_hyper
Each step also collect centralised information z^t, and store (h^t, z^t, a^t, r^t, h^{t+1}, z^{t+1}) in D
For a sampled transition k, if s^{k+1} is not terminal:
  θ_mix^{k+1} ← f_hyper(z^{k+1}; θ_hyper)
  y^k ← r^k + γ f_mix( max_{a_1'} Q(h_1^{k+1}, a_1'; θ̄_1), ..., max_{a_n'} Q(h_n^{k+1}, a_n'; θ̄_n); θ_mix^{k+1} )
θ_mix^k ← f_hyper(z^k; θ_hyper)
Value estimate Q(h^k, z^k, a^k; θ) ← f_mix( Q(h_1^k, a_1^k; θ_1), ..., Q(h_n^k, a_n^k; θ_n); θ_mix^k )
Loss L(θ) ← (1/B) Σ_k ( y^k − Q(h^k, z^k, a^k; θ) )²     // θ: utilities and hypernetwork together
```

Taking each agent's max inside $f_{\text{mix}}$ is valid only because $f_{\text{mix}}$ is monotonic: that is IGM again.

> [!example] QMIX on the linear game
> On the game with rewards 1, 5, 5, 9, QMIX learns utilities $Q_1(A) = 0.19$, $Q_1(B) = 0.96$ and $Q_2(A) = -0.21$, $Q_2(B) = 0.68$. The mix reproduces all four rewards exactly, but the utilities do not add up to them (0.19 − 0.21 is about 0, yet the mixed value is 1), because the mixing network transforms them. Each agent's greedy action is still B, so QMIX picks the optimal (B, B), as VDN does.

Monotonicity is still a restriction (from the MARL book): it cannot represent a game where one agent's best action flips depending on the other's. In the matching task, if agent 2's utility favours A, monotonicity makes (B, A) worth at least (B, B), which is wrong. QTRAN goes beyond linear and monotonic decompositions; it is named in the course but not covered.

## Modelling the other agents

So far the other agents' actions have been inferred implicitly from training data, through a centralised critic or a decomposition. **[[Agent Modelling|Agent modelling]]** makes it explicit: model the other agents' policies, so that an agent can condition its action selection on them and adapt when their behaviour changes.

In **deep joint-action learning**, agent $i$ maintains a model $\hat\pi_j^i$ of the policy of each other agent $j$, a network with parameters $\phi_j^i$:

- **Input:** agent $i$'s observation history $h_i$
- **Output:** a probability distribution over agent $j$'s actions

It is trained by supervised learning on the actions agent $i$ has seen $j$ take:

> [!formula] Agent model loss
> $$\mathcal{L}(\phi_j^i) = -\log\hat\pi_j^i\big(a_j^t \mid h_i^t;\phi_j^i\big)$$
> the negative log-likelihood (cross-entropy) of $j$'s observed action $a_j^t$ under the model: "according to the model, are the observed actions likely?"

Agent $i$ also learns a value over **joint** actions, $Q(h_i, \langle a_i, a_{-i}\rangle;\theta_i)$, and combines it with the models into an expected value of its own action (definition from the MARL book):

$$AV(h_i, a_i;\theta_i) = \sum_{a_{-i}} Q\big(h_i, \langle a_i, a_{-i}\rangle;\theta_i\big)\prod_{j \neq i}\hat\pi_j^i\big(a_j \mid h_i;\phi_j^i\big)$$

the joint-action value averaged over what the models predict the others will do.

```pseudo
Algorithm: Deep joint-action learning with agent models
───────────────────────────────────────────────────────
Initialize for each agent i: value network θ_i, target network θ̄_i ← θ_i,
  agent models φ^i_{-i} = {φ^i_j : j ≠ i}, replay buffer D_i
For time step t = 0, 1, 2, ...:
  Collect observations o_1^t, ..., o_n^t
  For each agent i:
    With probability ε: random action a_i^t
    Otherwise:          a_i^t ∈ argmax_{a_i} AV(h_i^t, a_i; θ_i)
  Apply joint action (a_1^t, ..., a_n^t); collect rewards r_1^t, ..., r_n^t and next observations
  For each agent i:
    Store (h_i^t, a^t, r_i^t, h_i^{t+1}) in D_i          // a^t is the joint action
    Sample a mini-batch of B transitions from D_i
    If s^{k+1} is terminal: y_i^k ← r_i^k
    Else:                   y_i^k ← r_i^k + γ max_{a_i'} AV(h_i^{k+1}, a_i'; θ̄_i)
    Critic loss L(θ_i)      ← (1/B) Σ_k ( y_i^k − Q(h_i^k, ⟨a_i^k, a_{-i}^k⟩; θ_i) )²
    Model loss  L(φ^i_{-i}) ← Σ_{j≠i} (1/B) Σ_k −log π̂^i_j(a_j^k | h_i^k; φ^i_j)
    Update θ_i and φ^i_{-i} by minimising their losses
    At a set interval, update θ̄_i ← θ_i
```

Each agent needs $n - 1$ extra networks, one per other agent, and must observe the others' actions. In return it reacts to the others' current behaviour as its models estimate it, where an independent learner sees only their past behaviour averaged into its rewards.

## Shaping how the others learn: LOLA

Agent modelling predicts what the others do. **Opponent shaping** goes one step further: the others are learning, so an agent can steer their learning to its own advantage. The well-known approach is **LOLA** (Learning with Opponent-Learning Awareness, Foerster et al. 2017): each agent explicitly models the anticipated learning update of its opponent and shapes it.

> [!formula] Naive policy gradient against LOLA (two agents $i$ and $j$)
> Naive learner:
> $$\theta_i^{t+1} = \theta_i^t + \alpha\nabla_{\theta_i}J_i(\theta_i^t, \theta_j^t)$$
> LOLA assumes the opponent will take its own gradient step:
> $$\theta_j^{t+1}(\theta_i^t) = \theta_j^t + \beta\nabla_{\theta_j}J_j(\theta_i^t, \theta_j^t)$$
> and uses that anticipated policy in its own update:
> $$\theta_i^{t+1} = \theta_i^t + \alpha\nabla_{\theta_i}J_i\Big(\theta_i^t, \theta_j^{t+1}(\theta_i^t)\Big)$$
> - $J_i$, $J_j$: each agent's expected return as a function of both agents' parameters; $\alpha$, $\beta$: the two learning rates
> - the opponent's next parameters are written as a function of $\theta_i^t$, because the opponent's gradient depends on agent $i$'s current policy

By the chain rule, differentiating $J_i(\theta_i, \theta_j^{t+1}(\theta_i))$ gives the naive gradient plus a **shaping term**: how agent $i$'s parameters change the opponent's learning step, times what that change is worth to agent $i$. This requires knowing (or estimating) the opponent's parameters and learning rule.

## Many similar agents: sharing parameters or experience

Every method so far gives each agent its own network $\theta_i$, and one DQN-sized network already has millions of parameters. Scalability and sample efficiency can improve when the agents are similar.

Agents are **homogeneous** when their policies can be interchanged while still preserving the overall result in solving the task, regardless of which agent is executing which policy. Two strengths:

> [!definition] Strongly and weakly homogeneous agents
> - **Weakly homogeneous:** for any joint policy, permuting the policies between agents leads to exactly the same outcomes (returns).
> - **Strongly homogeneous:** the optimal joint policy consists of identical policies for all agents.
>
> In the MARL book, strong homogeneity is weak homogeneity plus identical optimal policies.

> [!example] One landmark or two
> Two agents with the same actions and observations; swapping their policies does not change the return, so both environments below are weakly homogeneous. In **environment A** both must reach the **same** landmark: the optimal joint policy has both agents doing the same thing, so A is also strongly homogeneous. In **environment B** they must reach **two different** landmarks: the optimal policies differ (one agent per landmark), so B is only weakly homogeneous.

### Parameter sharing (strongly homogeneous agents)

All agents use **the same set of parameter values** in their networks: one network $\theta$ instead of $n$. During execution each agent runs its own copy on its own observations, so nothing has to be exchanged. Benefits:

- the **number of parameters stays constant** as agents are added;
- the shared parameters are **updated with the experience of all agents**, more diverse experience per update, which improves sample efficiency.

The cost is that all agents are **constrained to the same policy**, which fits environment A. Agents still behave differently when they see different observations, but in symmetric situations identical policies act identically: in environment B, they could not reliably send the agents to different landmarks. A common fix (from the MARL book) is to add the **agent index** to each agent's observation, which lets one network behave like different policies for different agents. See [[Parameter Sharing]].

### Experience sharing (weakly homogeneous agents)

Each agent keeps **its own network**, but all agents train on a **centralised buffer of experiences** collected by every agent. Weak homogeneity is what makes this valid: since swapping policies does not change outcomes, agent $j$'s experience is equally relevant experience for agent $i$. It is **more expensive** (there are still $n$ different policies to train), yet it **applies to less restrictive domains**, because the learned policies can differ. (From the MARL book: on-policy algorithms need a correction when learning from another agent's trajectories, which a different policy generated; shared experience actor-critic, SEAC, uses importance sampling weights.)

## Large zero-sum games: search and self-play

The last setting is specific: **zero-sum games with two agents and fully observable states and actions**, the turn-taking board games chess, shogi, backgammon and Go. Their challenges are a **sparse reward** (win or lose only at the end), a **large action space** (large branching factor: many legal moves per position) and a **long horizon**. Learning needs an efficient way to explore the game tree.

### Monte Carlo tree search

[[Monte Carlo Tree Search (MCTS)|MCTS]] simulates possible outcomes of the game from the current position and updates value estimates of the actions at each state it visits. Each simulation has three steps:

1. **Simulation and selection:** walk down the tree from the root, choosing an action at each node (ExploreAction) until reaching a state not yet in the tree.
2. **Expansion:** add that new leaf node $\hat s^l$ to the tree (InitializeNode). From it, a random rollout plays the game to the end; that game's outcome is $u$.
3. **Backpropagation:** send the simulation's outcome $u$ back up along the visited path, updating every node on it.

> [!formula] MCTS statistics
> $$N(s,a) \leftarrow N(s,a) + 1, \qquad Q(s,a) \leftarrow Q(s,a) + \frac{G - Q(s,a)}{N(s,a)}$$
> - $N(s,a)$: how often action $a$ was tried in $s$; $G$: the return of this simulation (the outcome $u$ above)
> - the update keeps $Q(s,a)$ equal to the **average return** over all simulations through $(s,a)$ (an incremental mean)
> - selection uses ε-greedy or [[Upper Confidence Bound|UCB]] on these statistics; UCB prefers actions that look good or have been tried rarely

### Self-play: AlphaZero

**[[Self-Play|Self-play]]** trains an agent's policy against itself: **the same policy chooses the actions for each agent**. This requires agents with **symmetrical roles** that can build **egocentric observations** (the board seen from the side of the player to move: "my pieces", "the opponent's pieces"), so one policy can play either side. Zero-sum board games typically have symmetric roles: the agents are direct opponents, each trying to defeat the other, and generally have access to the same types of actions.

[[AlphaGo Zero|AlphaZero]] combines MCTS with self-play (the details are from the MARL book): a neural network guides the tree search and evaluates new leaves in place of random playouts, the games it plays against itself provide the training data, and the improved network makes the next search stronger.

**Beyond games.** Games are a proxy for studying reasoning, the bridge from MARL to agentic AI: Thore Graepel's opinion piece "Don't be fooled: LLMs don't reason" (October 2026) argues that today's AI still lacks the machinery behind AlphaGo's win over Lee Sedol.

## Populations of policies: PSRO

(From MARL book Ch 9; Lecture 5 does not cover PSRO, [[AGAI-L06a - From Deep MARL to Reasoning|Lecture 6]] does.) Self-play trains only against the current version of oneself, so in games with **non-transitive** strategies (B beats A, C beats B, A beats C, as in rock-paper-scissors) it can cycle and forget how to beat old strategies. [[Population-Based Training|Population-based training]] trains against a population of policies instead.

**[[Policy Space Response Oracles (PSRO)]]** keeps a population of policies for each agent and repeats:

1. **Build the meta-game:** a [[Normal-Form Game|normal-form game]] whose **actions are the policies in the current populations**, and whose payoffs are the expected returns when those policies play each other, estimated by simulating episodes.
2. **Solve the meta-game** with a meta-solver, for example a [[Nash Equilibrium|Nash equilibrium]], giving each agent a probability distribution over its population.
3. **Oracle step:** for each agent, train a new policy with deep RL as a [[Best Response|best response]] to the other agents playing their meta-game distributions.
4. **Add** the new policies to the populations.

Each new policy must beat a mixture of the earlier ones, so the populations do not just chase each other's latest version.

## How the methods compare

| Method | Mode | Shared during training | What it addresses |
|---|---|---|---|
| Centralised critic (A2C, MAPPO) | CTDE | $z$ in the critic | non-stationarity, coordination |
| VDN | CTDE | team loss over a sum | credit assignment, IGM |
| QMIX | CTDE | team loss, mix from $z$ | credit assignment, richer IGM |
| Joint-action learning with agent models | DTDE | nothing (observes others' actions) | adapting to others' policies |
| Parameter sharing | CTDE | one network | parameters, sample efficiency |
| Experience sharing | CTDE | one replay buffer | sample efficiency, distinct policies |
| MCTS + self-play | self-play | one policy for both sides | sparse reward, huge game trees |

## What to be able to answer

- The three training and execution modes, their pros and cons, and which fits a scenario.
- The multi-agent policy gradient theorem; centralised A2C; why CTDE REINFORCE is rare.
- What centralised value functions leave unsolved; utilities, IGM and its two implications.
- VDN (derivation, algorithm, where additivity fails); QMIX (monotonicity, hypernetwork).
- Agent models (input, output, loss, $AV$); LOLA's update.
- Weak and strong homogeneity; parameter against experience sharing.
- MCTS steps and updates; self-play's requirements; the PSRO meta-game.

Model answers are in the exam questions.

## Exam questions

> [!exam]- Mock exam Q23 (DMARL 2, 2 points, select one): Autonomous cars cannot exchange all sensor data while driving, but a simulator can provide all data during training. Which mode fits best? (a) Centralised training and execution (b) Decentralised training and execution (c) Centralised training with decentralised execution (d) Decentralised training with centralised execution
> **Key points:** (c) centralised training with decentralised execution.
>
> The simulator makes all data available during training, so training can be centralised; the cars cannot exchange all their data while driving, so each must act on its own information at execution. (a) needs full data exchange while driving, (b) wastes the simulator's data, and (d) is backwards.

> [!exam]- Mock exam Q24 (DMARL 3, 2 points, select all that apply): Which statements about parameter sharing are correct? (A) It requires agents to exchange their observations during execution. (B) It works best when the optimal policies of the agents differ. (C) It constrains all agents to the same policy. (D) The number of parameters stays constant as the number of agents grows. (E) The shared parameters are updated with the experience of all agents.
> **Key points:** C, D and E.
>
> C: all agents use one set of parameter values, so one policy. D: there is one network however many agents there are. E: every agent's experience trains the shared network. A is false: each agent runs its own copy of the network on its own observations. B is false: it suits strongly homogeneous agents, whose optimal policies are identical.

> [!exam]- Mock exam Q25 (DMARL 4, 2 points, select one): What does a deep agent model of agent i learn to do? (a) Copy the true parameters of agent j at every step of execution (b) Split the common reward into one utility for each agent (c) Predict the actions of agent j from the observation history of agent i (d) Replace the value function of agent i with a smaller network
> **Key points:** (c).
>
> The model $\hat\pi_j^i(a_j \mid h_i;\phi_j^i)$ takes agent $i$'s observation history as input, outputs a distribution over agent $j$'s actions, and is trained on $j$'s observed actions with the loss $-\log\hat\pi_j^i(a_j^t \mid h_i^t)$. (a) agent $i$ cannot see $j$'s parameters; (b) is value decomposition; (d) describes nothing in agent modelling.

> [!exam]- Mock exam Q27 (DMARL 6, 2 points, select one): Why can experience sharing be preferred over parameter sharing? (a) It needs less computation per step, as batches become smaller. (b) Agents keep their own networks, so their learned policies can differ. (c) It needs fewer network parameters as the number of agents grows. (d) It needs no correction when applied to on-policy algorithms.
> **Key points:** (b).
>
> Experience sharing only pools the replay data; each agent still trains its own network, so it works for weakly homogeneous agents whose optimal policies differ. (a) and (c) are wrong: it is the more expensive option, with $n$ networks to train. (d) is wrong: on-policy learning from other agents' trajectories needs a correction such as importance sampling.

> [!exam]- Mock exam Q28 (DMARL 7, 2 points, select one): In Policy Space Response Oracles (PSRO), what is the meta-game? (a) A stochastic game whose states are the observation histories of the agents (b) A normal-form game whose actions are the policies in the current populations (c) A common-reward game whose value function is decomposed into utilities (d) A search tree whose nodes are expanded by Monte Carlo tree search
> **Key points:** (b).
>
> PSRO keeps a population of policies per agent. The meta-game is the normal-form game in which each agent "chooses" one of its policies and the payoffs are the expected returns of those policies against each other, estimated by simulation. Solving it gives a distribution over each population, against which a new best-response policy is trained. (c) describes value decomposition and (d) MCTS.

> [!exam]- Mock exam Q29 (DMARL 8, open, 10 points): In environment A, two agents must reach the same landmark. In environment B, two agents must reach two different landmarks. In both, the agents have the same actions and observations, and swapping their policies does not change the return. (a) Define weakly and strongly homogeneous agents. Classify the agents in A and in B. (3 points) (b) Give two benefits of parameter sharing. (3 points) (c) Which environment (A or B) can use parameter sharing without observing agent indices? Explain. (2 point) (d) Which alternative suits the other environment, to still use some agent homogeneity to improve learning efficiency? (2 points)
> **Key points:** weakly homogeneous = permuting policies does not change the returns, strongly = the optimal joint policy consists of identical policies; A strongly, B weakly homogeneous; parameters stay constant as agents are added, and shared parameters learn from all agents' more diverse experience (sample efficiency); A, since its optimal policies are identical, while identical policies in B could not reliably send the agents to different landmarks; experience sharing.
>
> (a) Agents are weakly homogeneous when permuting the policies between agents does not change the returns. They are strongly homogeneous when the optimal joint policy consists of identical policies. In both environments swapping policies leaves the return unchanged, so both are at least weakly homogeneous. In A the best joint policy is for both agents to go to the same landmark, an identical policy for each, so A is strongly homogeneous. In B the agents must split between two landmarks, so their optimal policies differ: B is weakly homogeneous.
>
> (b) With parameter sharing the number of parameters stays constant as agents are added, because all agents use one network. And the shared parameters learn from the more diverse experience of all agents, which improves sample efficiency.
>
> (c) A. Its optimal policies are identical, so one shared policy is exactly what is needed. In B identical policies with the same observations could not reliably send the agents to different landmarks; the shared network would need the agent index to tell the agents apart.
>
> (d) Experience sharing, in which the agents keep their own networks but learn from each other's trajectories. Weak homogeneity makes the other agent's experience relevant, and separate networks let the two agents learn different policies.

> [!exam]- Describe the three training and execution modes in MARL, with the advantages and disadvantages of each.
> **Key points:** centralised = uses others' shared information, decentralised = own experience only; CTCE: coordination and help with partial observability, but exponential joint action space and impossible for distributed agents; DTDE: scalable and works without communication, but non-stationary, suboptimal equilibria, cannot attribute changes to other agents; CTDE: shared information only in training, no central controller at execution, accounts for interactions, but training can be a bottleneck and local information may not suffice.
>
> Training is optimising the policy and exploring; execution is selecting actions. A phase is centralised when agent $i$ uses centrally shared information about the others (joint history, observations, global state, joint policy), decentralised when it relies only on its own experience.
>
> Centralised training and execution uses shared information throughout: observation histories, policies, value functions. It improves coordination, and in partially observable stochastic games the pooled observations remove much of the partial information. But the joint action space grows exponentially with the number of agents, and agents may be distributed entities by design.
>
> Decentralised training and execution uses no shared information: independent learning or agent modelling. It scales and works for distributed agents that cannot communicate, but training is affected by non-stationarity, agents can select suboptimal equilibria, and an agent cannot identify which changes are due to other agents.
>
> Centralised training with decentralised execution uses shared information only during training, as in an actor-critic with a centralised critic and decentralised actors. Execution needs no central controller, which suits distributed settings with limited runtime communication, and training can account for the interactions between agents. But centralised training can still be a bottleneck, and coordination is hard when local information is not sufficient during execution.

> [!exam]- State the multi-agent policy gradient theorem and explain how it leads to a centralised actor-critic. Describe centralised A2C and explain why a CTDE version of REINFORCE is uncommon.
> **Key points:** $\nabla_{\phi_i}J \propto \mathbb{E}[Q_i^\pi(\hat h, \langle a_i, a_{-i}\rangle)\nabla_{\phi_i}\log\pi_i(a_i \mid h_i;\phi_i)]$; evaluation on joint history and joint action, policy on own history; actor $\pi(\cdot \mid h_i;\phi_i)$ decentralised, critic $V(h_i, z;\theta_i)$ centralised and training-only; advantage $r_i + \gamma V(h_i', z') - V(h_i, z)$ in the actor loss, squared TD error for the critic; REINFORCE has no critic, so nothing can be centralised.
>
> The single-agent theorem weights $\nabla_\phi\log\pi(a \mid s)$ by $Q^\pi(s,a)$, so valuable actions become more likely. With several agents the value of agent $i$'s action depends on what the others do, so the theorem becomes $\nabla_{\phi_i}J(\phi_i) \propto \mathbb{E}_{\hat h, a_i \sim \pi_i, a_{-i} \sim \pi_{-i}}[Q_i^\pi(\hat h, \langle a_i, a_{-i}\rangle)\nabla_{\phi_i}\log\pi_i(a_i \mid h_i;\phi_i)]$. Actions are evaluated on the full history and the joint action, while the policy conditions only on agent $i$'s own history.
>
> That split is CTDE. The evaluation is only needed to compute gradients during training, so it can use centralised information; the policy is used at execution, so it stays decentralised. In the centralised actor-critic, each agent has an actor $\pi(\cdot \mid h_i;\phi_i)$ and a critic $V(h_i, z;\theta_i)$ whose input adds centralised information $z$, such as the joint history or global state.
>
> Centralised A2C runs $K$ parallel environments. Each step, every agent samples its action from its actor, the environments return rewards, observations and the next $z$. For each agent the advantage is $r_i + \gamma V(h_i', z';\theta_i) - V(h_i, z;\theta_i)$ (just $r_i - V$ at a terminal state); the actor's loss is minus the advantage-weighted log-probability of its actions, averaged over environments; the critic regresses on the target $r_i + \gamma V(h_i', z')$ with a squared loss.
>
> REINFORCE has no critic: it weights the gradient by sampled Monte Carlo returns. Its only learned component is the policy, which must stay decentralised for execution, so there is no training-only part into which centralised information could go.

> [!exam]- What problem does value decomposition solve? Define the IGM property, give its two implications, and check it for the decomposition $Q_1(A) = 0.88$, $Q_1(B) = 4.88$, $Q_2(A) = 0.12$, $Q_2(B) = 4.12$ of the common-reward game (A,A) = 1, (A,B) = (B,A) = 5, (B,B) = 9.
> **Key points:** centralised values handle non-stationarity, coordination and partial observability, but in common-reward games credit assignment remains and execution must be decentralised; learn per-agent utilities $Q(h_i, a_i;\theta_i)$; IGM: argmax of the joint value = tuple of individual argmaxes; implications: greedy decentralised execution, and cheap computation of the greedy joint action; the sums reproduce 1, 5, 5, 9 and both agents choose B, giving the optimal (B, B).
>
> A centralised action-value function $Q(h, z, a)$ conditions on everyone's information and the joint action. That deals with non-stationarity, coordination and partial observability, but in cooperative, common-reward games it is still hard to learn the team's best joint policy from one shared reward, because nothing says which agent's action mattered (credit assignment), and agents still have to select actions on their own at execution. Value decomposition learns one utility $Q(h_i, a_i;\theta_i)$ per agent, from its own history and action, such that greedy individual actions give the optimal joint action.
>
> The IGM property: $\arg\max_{a_1,\dots,a_n}Q(s, a_1, \dots, a_n) = (\arg\max_{a_1}Q_1(o_1,a_1), \dots, \arg\max_{a_n}Q_n(o_n,a_n))$. The joint action that maximises the joint value is the set of individually maximising actions. Implications: each agent can act greedily on its own utility for decentralised execution; and the greedy joint action, needed in every Q-learning target, is found by $n$ small maximisations instead of one over the exponentially large joint action space.
>
> Check: $0.88 + 0.12 = 1$, $0.88 + 4.12 = 5$, $4.88 + 0.12 = 5$, $4.88 + 4.12 = 9$, so the sum reproduces every reward. Agent 1 prefers B (4.88 > 0.88), agent 2 prefers B (4.12 > 0.12), and (B, B) = 9 is the maximum of the joint table, so IGM holds.

> [!exam]- Derive the linear value decomposition behind VDN, give the VDN algorithm, and explain how QMIX generalises it. What can neither represent?
> **Key points:** assume $r = \sum_i \bar r_i$; linearity of expectation gives $Q(h, z, a) = \sum_i Q(h_i, a_i;\theta_i)$; VDN: greedy local actions, shared buffer, target $r + \gamma\sum_i\max Q(h_i', a_i';\bar\theta_i)$, loss on $y - \sum_i Q_i$; additivity fails when success needs specific combinations (matching); QMIX: monotonic $f_{\text{mix}}$ with $\partial Q/\partial Q_i > 0$, weights from a hypernetwork on $z$, VDN is the special case; monotonic mixes still cannot represent best actions that flip with the other agent's action (QTRAN goes further).
>
> If the shared reward is a sum of individual terms $r^t = \bar r_1^t + \dots + \bar r_n^t$, the joint value is the expected discounted sum of $\sum_i \bar r_i$. Swapping the sums by linearity of expectation turns it into a sum of $n$ expectations, each the discounted return of one agent's term, and approximating each by a utility of that agent's history and action gives $Q(h, z, a;\theta) = \sum_i Q(h_i, a_i;\theta_i)$.
>
> VDN learns this. Each agent picks its action ε-greedily from its own utility. Transitions with the shared reward go into a shared replay buffer. For a mini-batch, the target is $r$ at terminal states and otherwise $r + \gamma\sum_i\max_{a_i'}Q(h_i', a_i';\bar\theta_i)$ with target networks; the loss is $(y - \sum_i Q(h_i, a_i;\theta_i))^2$, minimised jointly over all utilities, and target networks are refreshed at intervals. The sum satisfies IGM, and the network decides how to split credit.
>
> The additive assumption is restrictive: when two agents must choose matching actions, the value of an action depends on the partner's action, and no sum of individual values captures that.
>
> QMIX only requires monotonicity: $Q(h, z, a) = f_{\text{mix}}(Q(h_1, a_1), \dots, Q(h_n, a_n);\theta_{\text{mix}})$ with $\partial Q/\partial Q_i > 0$ for every agent. Raising a utility never lowers the joint value, so IGM still holds. The mixing parameters come from a hypernetwork $f_{\text{hyper}}(z;\theta_{\text{hyper}})$ fed with centralised information, and both the target and the estimate pass the utilities through $f_{\text{mix}}$. With the sum as mixing function it reduces to VDN. A monotonic mix still cannot represent games where one agent's best action flips depending on the other's choice, such as the matching task; methods like QTRAN go beyond it.

> [!exam]- Explain agent modelling in deep joint-action learning: what the agent model is, how it is trained and how it is used to select actions. Then explain how LOLA goes further.
> **Key points:** goal: explicitly model the others' policies to condition on them and adapt to changes; $\hat\pi_j^i(a_j \mid h_i;\phi_j^i)$, input $i$'s observation history, output distribution over $j$'s actions; loss $-\log\hat\pi_j^i(a_j^t \mid h_i^t)$; joint-action value $Q(h_i, \langle a_i, a_{-i}\rangle)$ averaged with the models gives $AV(h_i, a_i)$, used for greedy choice and targets; LOLA anticipates the opponent's gradient step $\theta_j + \beta\nabla_{\theta_j}J_j$ and differentiates through it.
>
> Independent learners and centralised critics infer the other agents' behaviour only implicitly from training data. Agent modelling models the others' policies explicitly, so an agent can condition its action selection on them and adapt when they change.
>
> In deep joint-action learning, agent $i$ keeps one model $\hat\pi_j^i$ per other agent $j$, a network with parameters $\phi_j^i$ whose input is agent $i$'s observation history and whose output is a probability distribution over $j$'s actions. It is trained by minimising $-\log\hat\pi_j^i(a_j^t \mid h_i^t;\phi_j^i)$, the negative log-likelihood of the actions $j$ was actually seen taking.
>
> Agent $i$ also learns a value over joint actions $Q(h_i, \langle a_i, a_{-i}\rangle;\theta_i)$, trained like DQN on the observed joint actions. To choose its own action it averages that value over the models' predictions: $AV(h_i, a_i) = \sum_{a_{-i}}Q(h_i, \langle a_i, a_{-i}\rangle)\prod_{j \neq i}\hat\pi_j^i(a_j \mid h_i)$. It acts ε-greedily on $AV$, and the Q-learning target is $r_i + \gamma\max_{a_i'}AV(h_i', a_i';\bar\theta_i)$.
>
> LOLA uses the fact that the others are learning. A naive learner updates $\theta_i \leftarrow \theta_i + \alpha\nabla_{\theta_i}J_i(\theta_i, \theta_j)$, treating the opponent as fixed. LOLA anticipates the opponent's step $\theta_j^{t+1}(\theta_i) = \theta_j + \beta\nabla_{\theta_j}J_j(\theta_i, \theta_j)$ and updates with $\nabla_{\theta_i}J_i(\theta_i, \theta_j^{t+1}(\theta_i))$. Because the opponent's next parameters depend on $\theta_i$, the gradient gains a shaping term: agent $i$ chooses its update partly for how it steers the opponent's learning.

> [!exam]- Why are large two-player zero-sum board games hard for RL, and how do MCTS and self-play (AlphaZero) address this? State the conditions self-play needs.
> **Key points:** two agents, fully observable, turn-taking (chess, shogi, backgammon, Go); sparse reward, large branching factor, long horizon, so efficient game-tree exploration is needed; MCTS: selection, expansion, backpropagation, $N \leftarrow N + 1$, $Q \leftarrow Q + (G - Q)/N$, ε-greedy or UCB; self-play: one policy chooses actions for both agents; needs symmetric roles and egocentric observations; AlphaZero: network-guided MCTS plus self-play.
>
> These games give a reward only at the end (win or lose), have many legal moves per position and last many moves, so random exploration almost never reaches informative outcomes and the tree is far too big to enumerate.
>
> Monte Carlo tree search builds a partial tree from the current position by repeated simulations. In selection it walks down the tree, choosing actions ε-greedily or by UCB on the current statistics; in expansion it adds the first new state reached as a leaf; in backpropagation it sends the simulation's outcome back up the path, updating $N(s,a) \leftarrow N(s,a) + 1$ and $Q(s,a) \leftarrow Q(s,a) + (G - Q(s,a))/N(s,a)$, so each $Q$ is the average outcome through that action. Effort concentrates on promising lines.
>
> Self-play trains an agent's policy against itself: the same policy chooses actions for each agent. That needs symmetric roles and egocentric observations, so one policy can play either side from its own perspective. Zero-sum board games fit: the agents are direct opponents with the same types of actions. AlphaZero combines the two: a neural network guides the tree search and evaluates positions, games of self-play generate training data, and the improved network strengthens the next search.

## Flashcards

> [!card]- What makes a training or execution phase centralised in MARL?
> Agent $i$ uses centrally shared information of others: joint history, observations, global state or joint policy.

> [!card]- What does CTDE use centralised information for?
> Only training; at execution each agent acts on its own history.

> [!card]- Why does CTCE scale badly?
> The joint action space grows exponentially with the number of agents.

> [!card]- Name the disadvantages of decentralised training and execution.
> Non-stationarity, suboptimal equilibria, unable to identify changes due to other agents.

> [!card]- Name the disadvantages of CTDE.
> Centralised training can be a bottleneck; coordination is hard when local information is insufficient at execution.

> [!card]- In the multi-agent policy gradient theorem, what does the action evaluation condition on?
> The full history and the joint action of all agents.

> [!card]- In the multi-agent policy gradient theorem, what does the policy term condition on?
> Only agent $i$'s own observation history.

> [!card]- In a centralised actor-critic, which network gets the centralised information $z$?
> The critic $V(h_i, z;\theta_i)$; the actor $\pi(\cdot \mid h_i;\phi_i)$ stays decentralised.

> [!card]- Why is a CTDE version of REINFORCE uncommon?
> REINFORCE has no critic, and its policy must stay decentralised, so nothing can be centralised.

> [!card]- What does a centralised action-value critic take as input beyond $h_i$ and $z$?
> The other agents' actions $a_{-i}$; it outputs one value per own action.

> [!card]- What does a centralised action-value critic make possible?
> Counterfactual reasoning: comparing my action with alternatives while the others' actions stay fixed.

> [!card]- Which MARL problem remains for centralised value functions in common-reward games?
> Credit assignment: learning the team's best joint policy from one shared reward.

> [!card]- Why are the per-agent functions in value decomposition called utilities?
> They need not equal any actual expected return; only their combination into the team value matters.

> [!card]- What does the IGM property state?
> The joint action maximising the joint value equals the tuple of individually maximising actions.

> [!card]- What is the name for "the argmax of the joint value equals the individual argmaxes"?
> The Individual-Global-Max (IGM) property.

> [!card]- Name the two implications of the IGM property.
> Greedy decentralised execution; cheap computation of the greedy joint action.

> [!card]- What assumption about the reward gives VDN's linear decomposition?
> The shared reward is a sum of individual reward terms, $r = \sum_i \bar r_i$.

> [!card]- What is VDN's training target?
> $r + \gamma\sum_i\max_{a_i'}Q(h_i', a_i';\bar\theta_i)$, the reward plus the discounted sum of greedy utilities.

> [!card]- Why does VDN satisfy IGM?
> The maximum of a sum of separately chosen terms is the sum of their maxima.

> [!card]- Why can VDN not represent a task where two agents must match actions?
> An action's value depends on the partner's action, which a sum of individual values cannot express.

> [!card]- What condition does QMIX's mixing function satisfy?
> Monotonicity: $\partial Q / \partial Q_i > 0$ for every agent's utility.

> [!card]- Why is monotonicity enough for IGM?
> Raising any utility never lowers the joint value, so individual maxima give the joint maximum.

> [!card]- Where do QMIX's mixing weights come from?
> A hypernetwork that takes the centralised information $z$ as input.

> [!card]- How is VDN a special case of QMIX?
> VDN is QMIX with the sum as mixing function.

> [!card]- What can a monotonic decomposition still not represent?
> A game where one agent's best action flips depending on the other's action.

> [!card]- What is the goal of agent modelling?
> To explicitly model the other agents' policies, conditioning action choice on them and adapting to changes.

> [!card]- What are the input and output of agent $i$'s model of agent $j$?
> Input: agent $i$'s observation history. Output: a distribution over agent $j$'s actions.

> [!card]- What loss trains an agent model $\hat\pi_j^i$?
> $-\log\hat\pi_j^i(a_j^t \mid h_i^t;\phi_j^i)$: negative log-likelihood of $j$'s observed actions.

> [!card]- In deep joint-action learning, what is $AV(h_i, a_i)$?
> The joint-action value averaged over the agent models' predicted actions of the others.

> [!card]- What does LOLA anticipate that a naive policy gradient learner ignores?
> The opponent's next learning step, which depends on the agent's own current parameters.

> [!card]- What is opponent shaping?
> Steering how other learning agents update, to one's own advantage.

> [!card]- When are agents weakly homogeneous?
> When permuting policies between agents, for any joint policy, leaves the outcomes unchanged.

> [!card]- When are agents strongly homogeneous?
> When the optimal joint policy consists of identical policies for all agents.

> [!card]- Which kind of homogeneity suits parameter sharing, and which experience sharing?
> Parameter sharing: strongly homogeneous. Experience sharing: weakly homogeneous.

> [!card]- How does parameter sharing work?
> All agents use the same parameter values in their networks.

> [!card]- How does experience sharing work?
> Each agent keeps its own network but trains on a centralised buffer of all agents' experience.

> [!card]- Why is experience sharing more expensive than parameter sharing?
> It still trains a separate network per agent.

> [!card]- How can one shared network still give agents different behaviour?
> Add the agent index to each agent's observation.

> [!card]- Which three challenges make large zero-sum board games hard?
> Sparse reward, large branching factor, long horizon.

> [!card]- Name the three steps of each MCTS simulation.
> Simulation and selection, expansion, backpropagation.

> [!card]- What does the MCTS update $Q \leftarrow Q + (G - Q)/N$ keep $Q(s,a)$ equal to?
> The average return over all simulations through $(s,a)$.

> [!card]- What does self-play require of the game?
> Symmetric roles and egocentric observations.

> [!card]- What is the PSRO meta-game?
> A normal-form game whose actions are the policies in the current populations.

> [!card]- What does PSRO's oracle step do?
> Trains a new policy per agent as a best response to the others' meta-game distribution.

> [!card]- True or false: CTDE agents need a central controller to choose actions at execution.
> **False.** CTDE uses shared information only in training; execution is decentralised.

> [!card]- True or false: In centralised A2C the actor conditions on the centralised information $z$.
> **False.** The critic conditions on $z$; the actor conditions only on $h_i$.

> [!card]- True or false: VDN's utilities must each equal an agent's expected return.
> **False.** Only their sum must match the team value; the split is not unique.

> [!card]- True or false: QMIX's mixing function must be monotonic in each agent's utility.
> **True.** Monotonicity is what preserves IGM.

> [!card]- True or false: QMIX's mixing function must be linear in each agent's utility.
> **False.** It only has to be monotonic; linear is the VDN special case.

> [!card]- True or false: A deep agent model copies the true parameters of the other agent.
> **False.** It predicts the other's actions from the modelling agent's own observation history.

> [!card]- True or false: Parameter sharing requires agents to exchange observations during execution.
> **False.** Each agent runs its own copy of the shared network on its own observations.

> [!card]- True or false: With parameter sharing the number of parameters grows linearly with the number of agents.
> **False.** It stays constant: one network for all agents.

> [!card]- True or false: Every strongly homogeneous environment is also weakly homogeneous.
> **True.** In the MARL book, strong homogeneity is weak homogeneity plus identical optimal policies.

> [!card]- True or false: Experience sharing needs no correction for on-policy algorithms.
> **False.** Other agents' trajectories come from other policies, so a correction such as importance sampling is needed.

> [!card]- True or false: In self-play the same policy chooses the actions of both players.
> **True.** That is the definition of self-play.

> [!card]- Multiple selection: which methods use centralised information only during training? (A) Centralised A2C (B) VDN (C) QMIX (D) Independent DQN
> **A, B and C.** D uses no centralised information at all.

> [!card]- Multiple selection: which are true of the IGM property? (A) Each agent can act greedily on its own utility (B) It requires a linear decomposition (C) It makes the greedy joint action cheap to compute (D) QMIX satisfies it through monotonic mixing
> **A, C and D.** B: monotonic decompositions also satisfy IGM.

> [!card]- Multiple selection: which are true of parameter sharing? (A) It suits strongly homogeneous agents (B) All agents train on each other's experience (C) It lets agents learn very different policies without agent indices (D) It keeps the parameter count constant as agents are added
> **A, B and D.** C: without indices, agents share one policy and act alike in symmetric situations.

> [!card]- Multiple selection: which are true of agent modelling in deep joint-action learning? (A) Agent $i$ keeps one model per other agent (B) Models are trained on the others' observed actions (C) Models need the others' true parameters (D) Action selection averages joint-action values over the models
> **A, B and D.** C: models are learned from observed behaviour.

## Links

- **Course:** [[AGAI - Overview|Course overview]]
- **Previous lecture:** [[AGAI-L04 - Deep RL and Intro to Deep MARL]] (DQN, REINFORCE, A2C, PPO, training modes, independent DQN and independent policy gradient)
- **Earlier:** [[AGAI-L03 - Intro to Multi-Agent Reinforcement Learning]] (non-stationarity, credit assignment, equilibrium selection, independent Q-learning)
- **Lab:** [[AGAI-Lab02 - Learning Agents]] (independent learners, the limits of tabular RL, PPO)
- **Concepts introduced here:** [[Individual-Global-Max Property]], [[Value Decomposition Networks (VDN)]], [[QMIX]], [[Agent Modelling]], [[Parameter Sharing]], [[Policy Space Response Oracles (PSRO)]]
- **Related concepts:** [[Centralised Training with Decentralised Execution]], [[Independent DQN]], [[Actor-Critic]], [[Policy Gradient Methods]], [[Value Function]], [[Q-Learning]], [[Multi-Agent System]], [[Deep Reinforcement Learning]], [[Epsilon-Greedy Policy]]
- **Source:** Agentic Artificial Intelligence, Lecture 5, "Deep MARL", Roman Chiva Gil, MSc AI, UvA, 10 November 2026
- **Reading:** Albrecht, Christianos and Schäfer, *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches* (MIT Press, 2024), Chapter 9 (exam material for the midterm): training and execution modes, multi-agent policy gradient, value decomposition, agent modelling, homogeneous agents, self-play and population-based training
- **Papers named:** Foerster et al., "Learning with Opponent-Learning Awareness" (AAMAS 2017); Graepel, "Don't be fooled: LLMs don't reason" (opinion piece, 2 October 2026)
- **Mock exam:** AGAI Mock Midterm 2026 with annotated solutions, Deep MARL questions 22 to 29
