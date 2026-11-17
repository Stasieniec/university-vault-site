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
> Independent learners ignore each other; one central learner over the joint action does not scale. Centralised training with decentralised execution (CTDE) uses everyone's information while learning and lets each agent act alone: a centralised critic with decentralised actors, or value decomposition into one utility per agent whose greedy actions give the best joint action (IGM), as a sum (VDN) or a monotonic mix (QMIX). Agents can also model the others' policies, or shape how the others learn (LOLA). Similar agents can share one network (parameter sharing) or one pool of experience (experience sharing). For two-player zero-sum board games, Monte Carlo tree search plus self-play (AlphaZero) trains one policy against itself.
>
> Feeds mock questions 23 (training modes), 24 (parameter sharing), 25 (agent models), 27 (experience against parameter sharing) and the open question 29 (homogeneous agents). Q22 and Q26 (independent DQN and independent REINFORCE) are in [[AGAI-L04 - Deep RL and Intro to Deep MARL|Lecture 4]]; Q28 (PSRO) is in [[AGAI-L06a - From Deep MARL to Reasoning|Lecture 6]].

Notation used throughout: $h_i$ is agent $i$'s own observation history, $z$ is **centralised information** (anything beyond $h_i$ that is available in training: the global state, the joint history, the other agents' observations), and $a_{-i}$ are the actions of everyone except $i$.

## Who sees what, and when: training and execution modes

**Training** is optimising the policy and exploring; **execution** is selecting actions once the policy is in use. A phase is **centralised** when an agent uses centrally shared information of the others (joint history, observations, global state, joint policy), and **decentralised** when it relies only on its own experience. That gives three modes.

**Centralised training and execution (CTCE).** Shared information is used all the time.
- Advantages: better coordination; in a partially observable stochastic game, pooling everyone's observations removes much of the partial information.
- Disadvantages: the joint action space grows exponentially with the number of agents; agents may be distributed entities by design, with no way to share a controller.

**Decentralised training and execution (DTDE).** No shared information at any point: independent learning, or agent modelling from one's own observations.
- Advantages: scales; works for distributed agents that cannot communicate.
- Disadvantages: training suffers from non-stationarity; agents can settle in suboptimal equilibria; an agent cannot tell which changes are due to the other agents.

**Centralised training with decentralised execution (CTDE).** Shared information is used **only during training**; each trained policy needs only the agent's own history. The standard example is an actor-critic with a centralised critic (training only) and decentralised actors (execution).
- Advantages: no central controller at execution; suits distributed settings with limited runtime communication; training accounts for the interactions between agents, which improves coordination.
- Disadvantages: centralised training can still be a bottleneck; coordination is hard when local information is not enough at execution.

> [!example] Choosing a mode
> Autonomous cars cannot exchange all their sensor data while driving, but a simulator provides all the data during training. Execution has to be decentralised, training can be centralised: CTDE. ("Decentralised training with centralised execution" is backwards: it drops shared information where it is free.)

See [[Centralised Training with Decentralised Execution]]. CTDE is applied to both families from Lecture 4: policy gradient methods (centralised critics) and value-based methods (value decomposition).

## Centralised critics: CTDE for policy gradient

The single-agent [[Policy Gradient Theorem|policy gradient]] pushes up the log-probability of an action in proportion to how valuable it is. With several agents, the value of my action depends on what the others do at the same moment, so the evaluation uses the joint action:

> [!formula] Multi-agent policy gradient theorem
> $$\nabla_{\phi_i} J(\phi_i) \propto \mathbb{E}\Big[\,Q_i^\pi\big(\hat h, \langle a_i, a_{-i}\rangle\big)\;\nabla_{\phi_i}\log\pi_i(a_i \mid h_i;\phi_i)\Big]$$
> The **evaluation** $Q_i^\pi$ conditions on the full history $\hat h$ and the **joint action**; the **policy** $\pi_i$ conditions only on agent $i$'s own history $h_i$.

The evaluation is only needed to compute gradients during training, so it may use centralised information. The policy is what the agent carries into execution, so it stays decentralised. That split is CTDE. In the **centralised actor-critic**:

- **Actor** $\pi(a_i \mid h_i;\phi_i)$: decentralised, used for execution.
- **Critic** $V(h_i, z;\theta_i)$: takes the agent's history plus centralised information $z$, used only in training. Its advantage estimate $r_i + \gamma V(h_i', z') - V(h_i, z)$ drives the actor's update, so the actor benefits from $z$ without ever seeing it.

**Why is a CTDE version of [[REINFORCE]] rare?** REINFORCE has no critic: it weights the gradient by the sampled return. Its only learned part is the policy, which must stay decentralised, so there is nowhere to put the centralised information.

**Value critic or action-value critic.** A centralised **value** critic $V(h_i, z)$ outputs one number (MAPPO, A2C). A centralised **action-value** critic also takes the other agents' actions $a_{-i}$ as input and outputs one value per own action. It can say what would have happened had agent $i$ acted differently while the others kept their actions, which supports **counterfactual reasoning**.

## Splitting the team value into utilities

A centralised action-value function $Q(h, z, a)$ over the joint action tackles three MARL problems: non-stationarity, coordination and partial observability. Two things remain in cooperative, common-reward games:

- **[[Multi-Agent Credit Assignment|Credit assignment]]**: one shared reward never says which agent's action made the difference.
- Execution still has to be **decentralised**, and a max over the joint action grows exponentially.

The ideal is one **individual utility** $Q(h_i, a_i;\theta_i)$ per agent, from its own history and action, such that every agent acting greedily on its own utility produces the optimal joint action. This is **value decomposition**. "Utility" because, once trained through the decomposition, it need not equal any real expected return: only its role in rebuilding the team value matters.

> [!definition] Individual-Global-Max (IGM) property
> $$\arg\max_{a}\, Q(h, z, a) = \Big(\arg\max_{a_1}Q(h_1, a_1),\; \dots,\; \arg\max_{a_n}Q(h_n, a_n)\Big)$$
> The **joint action that maximises the joint action-value function** equals the **set of individual actions that maximise the individual utility functions**. See [[Individual-Global-Max Property]].

Why IGM matters, two implications:

1. **Decentralised execution:** each agent can follow the greedy policy on its own utility, and together they play the greedy joint action.
2. **Efficiency:** the greedy joint action is found by $n$ small maximisations (one per agent) instead of one over the exponentially many joint actions. Training needs this too, because every Q-learning target contains a max over the next joint action.

How to decompose in practice: linearly (VDN), monotonically (QMIX), or beyond both (QTRAN, named but not covered).

## VDN: the team value as a sum

[[Value Decomposition Networks (VDN)|Value decomposition networks (VDN)]] assume the shared reward is a sum of individual (unobserved) reward terms, $r = \bar r_1 + \dots + \bar r_n$. By linearity of expectation the team value then splits into a sum too:

> [!formula] VDN
> $$Q(h, z, a;\theta) = \sum_{i} Q(h_i, a_i;\theta_i)$$
> Learn agent-level utilities whose **sum predicts the team value**.

Training is DQN on the sum. All agents store transitions in one shared replay buffer; each acts ε-greedily on its own utility; the target is $r + \gamma \sum_i \max_{a_i'} Q(h_i', a_i';\bar\theta_i)$ (just $r$ at a terminal state), and the loss is $\big(y - \sum_i Q(h_i, a_i;\theta_i)\big)^2$. The gradient flows through the sum into every agent's network, so the networks themselves discover how to split the credit. A sum satisfies IGM: the max of a sum of separately chosen terms is the sum of their maxima.

> [!example] A game VDN fits
> Common reward: (A, A) = 1, (A, B) = (B, A) = 5, (B, B) = 9. VDN learns $Q_1(A) = 0.88$, $Q_1(B) = 4.88$, $Q_2(A) = 0.12$, $Q_2(B) = 4.12$. Every sum matches ($0.88 + 0.12 = 1$, $4.88 + 4.12 = 9$, and so on), both agents greedily pick B, and (B, B) is the best joint action.

**Limitation: additivity is restrictive.** Some tasks depend on specific combinations of actions, for example two agents that must choose **matching** actions to succeed. There the value of my action depends on the partner's action, and a sum of individual values cannot capture that interaction.

## QMIX: any monotonic mix

IGM does not need linearity. **Monotonicity is enough**: if raising any agent's utility can never lower the joint value, then each agent maximising its own utility maximises the joint value.

> [!formula] QMIX
> $$Q(h, z, a;\theta) = f_{\text{mix}}\big(Q(h_1, a_1;\theta_1), \dots, Q(h_n, a_n;\theta_n);\,\theta_{\text{mix}}\big), \qquad \frac{\partial Q(h, z, a)}{\partial Q(h_i, a_i)} > 0 \text{ for every agent } i$$
> $f_{\text{mix}}$ is the **mixing network**, monotonic in every utility.

[[QMIX]] is VDN with the sum replaced by a learned mix, in the target and in the estimate. The mixing weights $\theta_{\text{mix}}$ come from a **hypernetwork** (a network that outputs another network's weights) whose input is the centralised information $z$, typically the global state. So the mix can change from state to state while staying monotonic. VDN is the special case where $f_{\text{mix}}$ is the sum, so QMIX can represent everything VDN can, and more (it can combine utilities nonlinearly).

So QMIX is the answer to VDN's restrictive additivity: a monotonic mix captures interactions between the agents that a plain sum cannot.

**Limitation (MARL book, not in the lecture):** monotonicity still cannot represent a game where one agent's best action **flips depending on the other agent's action**, such as a strict matching task. Both VDN and QMIX keep each agent's preference over its own actions independent of what the others pick.

## Modelling the other agents

Centralised critics and decompositions infer the other agents' behaviour only implicitly, from training data. **[[Agent Modelling|Agent modelling]]** makes it explicit: model the other agents' policies, so an agent can condition on them and adapt when their behaviour changes.

In **deep joint-action learning**, agent $i$ keeps one model $\hat\pi_j^i$ per other agent $j$:

- **Input:** agent $i$'s own observation history $h_i$.
- **Output:** a probability distribution over agent $j$'s actions.
- **Training:** supervised, on the actions $j$ was actually seen taking, with loss $-\log\hat\pi_j^i(a_j \mid h_i)$ ("according to the model, are the observed actions likely?").

Agent $i$ also learns a value over joint actions $Q(h_i, \langle a_i, a_{-i}\rangle)$ and scores its own action by averaging that value over what its models predict the others will do (MARL book). This costs $n - 1$ extra networks per agent, and the agent must observe the others' actions. In return it reacts to their current behaviour.

## Shaping how the others learn: LOLA

**Opponent shaping** goes one step further: the others are learning, so steer their learning to your advantage. **LOLA** (Learning with Opponent-Learning Awareness) assumes the opponent will take its own gradient step, and differentiates its own return through that anticipated step:

$$\theta_i \leftarrow \theta_i + \alpha\,\nabla_{\theta_i} J_i\big(\theta_i,\; \theta_j + \Delta\theta_j(\theta_i)\big)$$

A naive learner treats $\theta_j$ as fixed. LOLA's extra term says how its own parameters change the opponent's next update. It needs to know or estimate the opponent's parameters and learning rule.

## Many similar agents: sharing parameters or experience

Every method so far gives each agent its own network, and one DQN-sized network already has millions of parameters. When agents are similar, they can share. Agents are **homogeneous** when their policies can be interchanged while still preserving the overall result, regardless of which agent executes which policy.

> [!definition] Weakly and strongly homogeneous agents
> - **Weakly homogeneous:** for any joint policy, permuting the policies between agents leads to exactly the same outcomes (returns).
> - **Strongly homogeneous:** the optimal joint policy consists of identical policies for all agents.

> [!example] One landmark or two
> Two agents with the same actions and observations, and swapping their policies does not change the return. In **environment A** both must reach the **same** landmark: the optimal policies are identical, so A is strongly homogeneous. In **environment B** they must reach **two different** landmarks: the optimal policies differ (one agent per landmark), so B is only weakly homogeneous.

### Parameter sharing (strongly homogeneous agents)

All agents use **the same set of parameter values** in their networks: one network instead of $n$. Each agent runs its own copy on its own observations, so nothing is exchanged at execution. See [[Parameter Sharing]].

- **Benefits:** the number of parameters **stays constant** as agents are added; the shared parameters are **updated with the experience of all agents**, more diverse experience per update, which improves sample efficiency.
- **Limitation:** all agents are **constrained to the same policy**. Agents with different observations still act differently, but in symmetric situations they act identically. In environment B a shared policy cannot reliably send the two agents to different landmarks. The usual fix (MARL book) is to add the **agent index** to each agent's observation, so one network can behave differently per agent.

### Experience sharing (weakly homogeneous agents)

Each agent **keeps its own network**, but all train on a **centralised buffer of experiences** collected by every agent. Weak homogeneity is what makes this valid: since swapping policies does not change outcomes, agent $j$'s experience is relevant experience for agent $i$.

- **Benefits:** each agent learns from every agent's experience, which improves sample efficiency; applies to less restrictive domains, because the learned policies can differ (environment B).
- **Limitation:** more expensive, since there are still $n$ networks to train. On-policy algorithms need a correction when learning from another agent's trajectories, because a different policy generated them (MARL book: importance sampling weights).

## Large zero-sum games: search and self-play

The last setting: **two-player zero-sum games with fully observable states and actions**, the turn-taking board games chess, shogi, backgammon and Go. They are hard because of a **sparse reward** (win or lose only at the end), a **large action space** (many legal moves per position, a large branching factor) and a **long horizon**. Learning needs an efficient way to explore the game tree.

### Monte Carlo tree search

[[Monte Carlo Tree Search (MCTS)|MCTS]] builds a partial tree from the current position by repeated simulations of the game, and keeps value estimates of the actions at each state it visits. Each simulation:

1. **Selection (and simulation):** walk down the tree from the root, choosing an action at each node, until reaching a state not yet in the tree.
2. **Expansion:** add that state as a new leaf. In plain MCTS a **random rollout** then plays the game to the end, giving an outcome $G$.
3. **Backpropagation:** send $G$ back up the visited path and update every $(s, a)$ on it:

$$N(s,a) \leftarrow N(s,a) + 1, \qquad Q(s,a) \leftarrow Q(s,a) + \frac{G - Q(s,a)}{N(s,a)}$$

$N(s,a)$ counts how often $a$ was tried in $s$, and the update keeps $Q(s,a)$ equal to the **average outcome** of the simulations through $(s,a)$. Selection uses ε-greedy or [[Upper Confidence Bound|UCB]] on these statistics (UCB prefers actions that look good or have been tried rarely), so effort concentrates on promising lines.

### Self-play: AlphaZero

**[[Self-Play|Self-play]]**: **the same policy chooses the actions for each agent**, so the agent trains against itself. This needs agents with **symmetrical roles** that can build **egocentric observations** (the board from the mover's side: "my pieces" and "the opponent's pieces"), so one policy can play either side. Zero-sum board games usually fit: the agents are direct opponents with the same types of actions.

**AlphaZero** = MCTS + self-play + deep learning: a network guides the tree search, the games it plays against itself are the training data, and the improved network makes the next search stronger. How AlphaGo's networks plug into MCTS, what happens when roles are not symmetric (population-based training, PSRO) and AlphaStar are in [[AGAI-L06a - From Deep MARL to Reasoning|Lecture 6]].

## How the methods compare

| Method | Mode | What is shared or centralised | What it addresses |
|---|---|---|---|
| Independent learners | DTDE | nothing | scalability |
| Centralised critic (A2C, MAPPO) | CTDE | $z$ in the critic | non-stationarity, coordination |
| VDN | CTDE | team loss over a sum | credit assignment, IGM |
| QMIX | CTDE | team loss, mix from $z$ | same, richer than a sum |
| Agent models | DTDE | nothing (observes others' actions) | adapting to others |
| Parameter sharing | CTDE | one network | parameters, sample efficiency |
| Experience sharing | CTDE | one replay buffer | sample efficiency, distinct policies |
| MCTS + self-play | self-play | one policy for both sides | sparse reward, huge game trees |

## What to be able to answer

- Pick CTCE, DTDE or CTDE for a scenario; the advantages and disadvantages of each.
- What a centralised critic sees and why the actor does not; why CTDE REINFORCE is rare.
- IGM and its two implications; VDN (sum) and QMIX (monotonic mix, hypernetwork on $z$); what each cannot represent.
- What a deep agent model learns (input, output, loss); LOLA in one line.
- Weak and strong homogeneity; parameter sharing against experience sharing (mock Q29).
- MCTS steps and update; what self-play needs.

## Exam questions

> [!exam]- Mock exam Q29 (open, 10 points): In environment A, two agents must reach the same landmark. In environment B, two agents must reach two different landmarks. In both, the agents have the same actions and observations, and swapping their policies does not change the return. (a) Define weakly and strongly homogeneous agents. Classify the agents in A and in B. (3 points) (b) Give two benefits of parameter sharing. (3 points) (c) Which environment (A or B) can use parameter sharing without observing agent indices? Explain. (2 point) (d) Which alternative suits the other environment, to still use some agent homogeneity to improve learning efficiency? (2 points)
> **Key points:** weakly homogeneous = permuting the policies between agents does not change the returns; strongly = the optimal joint policy consists of identical policies; A strongly, B weakly homogeneous; parameter count stays constant as agents are added, and the shared parameters learn from all agents' more diverse experience (sample efficiency); A, because its optimal policies are identical; experience sharing.
>
> (a) Weakly homogeneous: permuting the policies between agents does not change the returns. Strongly homogeneous: the optimal joint policy consists of identical policies. A is strongly homogeneous (both agents should do the same thing, go to the one landmark). B is weakly homogeneous (swapping policies changes nothing, but the optimal policies differ, one agent per landmark).
>
> (b) The number of parameters stays constant as agents are added. The shared parameters learn from the more diverse experience of all agents, which improves sample efficiency.
>
> (c) A. Its optimal policies are identical, so one shared policy is exactly what is needed. In B, identical policies on the same kind of observations could not reliably send the agents to different landmarks.
>
> (d) Experience sharing: the agents keep their own networks but learn from each other's trajectories. Weak homogeneity makes the other agent's experience relevant, and separate networks let the policies differ.

> [!exam]- Warehouse robots are trained in a simulator that shows the full map and every robot's position. Once deployed, each robot sees only its own sensors and the radio link is unreliable. (a) Define centralised and decentralised training and execution. (b) Which mode fits, and why? (c) Give one advantage and one disadvantage of that mode. (d) Why not centralised training and execution?
> **Key points:** centralised = uses shared information of the others, decentralised = own experience only; CTDE, since shared information exists only in training; advantage: no central controller at execution / training accounts for interactions; disadvantage: training bottleneck / hard to coordinate when local information is insufficient; CTCE would need the radio at execution and a joint action space exponential in the number of robots.
>
> (a) Training is optimising the policy and exploring; execution is selecting actions. A phase is centralised when an agent uses centrally shared information of the others (joint observations, global state, joint policy), decentralised when it uses only its own experience.
>
> (b) CTDE. The simulator's full information is available during training, so training can be centralised (for example a critic that sees the whole map). At execution each robot must act on its own sensors.
>
> (c) Advantage: no central controller at execution, and training still accounts for how the robots interact, which helps coordination. Disadvantage: when a robot's local information is not enough at execution, coordination is hard; centralised training can also be a bottleneck.
>
> (d) CTCE would need all robots' information while acting, which the unreliable radio cannot provide, and its joint action space grows exponentially with the number of robots.

> [!exam]- Two agents share a common reward. (a) Which problem remains for a centralised action-value function in common-reward games? (b) Define the IGM property and give its two implications. (c) Give VDN's decomposition and check IGM for $Q_1(A) = 0.88$, $Q_1(B) = 4.88$, $Q_2(A) = 0.12$, $Q_2(B) = 4.12$ on the game (A, A) = 1, (A, B) = (B, A) = 5, (B, B) = 9. (d) Give a task VDN cannot represent, and say what QMIX changes.
> **Key points:** credit assignment (and decentralised action selection is still needed); IGM: argmax of the joint value = tuple of individual argmaxes; implications: greedy decentralised execution, cheap greedy joint action; VDN: $Q = \sum_i Q_i$, sums reproduce 1, 5, 5, 9 and both choose B = optimal; matching actions; QMIX uses a monotonic mixing network with weights from a hypernetwork on $z$.
>
> (a) Credit assignment: one shared reward does not say which agent's action made the difference. Also, agents still have to choose actions on their own at execution.
>
> (b) IGM: the joint action that maximises the joint action-value equals the set of individual actions that maximise each agent's utility. So each agent can act greedily on its own utility (decentralised execution), and the greedy joint action costs one small max per agent instead of a max over all joint actions.
>
> (c) VDN: $Q(h, z, a) = \sum_i Q(h_i, a_i)$. Sums: $0.88 + 0.12 = 1$, $0.88 + 4.12 = 5$, $4.88 + 0.12 = 5$, $4.88 + 4.12 = 9$. Both agents prefer B, and (B, B) = 9 is the best joint action, so IGM holds.
>
> (d) Two agents that must choose matching actions: the value of an action depends on the partner's action, which no sum captures. QMIX replaces the sum by a mixing network that only has to be monotonic in each utility ($\partial Q / \partial Q_i > 0$), with weights produced by a hypernetwork from the global state, so it can capture interactions a sum cannot while keeping IGM.

> [!exam]- (a) In a centralised actor-critic, which network receives the centralised information, and why not the other? (b) Why is a CTDE version of REINFORCE uncommon? (c) What does a deep agent model of agent $i$ learn: input, output and training signal? (d) What does LOLA add to a naive policy gradient learner?
> **Key points:** critic $V(h_i, z)$ gets $z$ because it is used only in training; the actor $\pi(a_i \mid h_i)$ must run at execution on local information; REINFORCE has no critic, only a policy that must stay decentralised; agent model: input agent $i$'s observation history, output a distribution over agent $j$'s actions, trained on $j$'s observed actions (negative log-likelihood); LOLA anticipates the opponent's next learning step and shapes it.
>
> (a) The critic. It is only used to compute the actor's update during training, so it may see $z$ (joint history, global state). The actor is what each agent runs at execution, so it conditions on its own history only; it benefits from $z$ through the critic's advantage.
>
> (b) REINFORCE weights the gradient by sampled returns and has no critic. Its only learned part is the policy, which must stay decentralised, so centralised information has nowhere to go.
>
> (c) Agent $i$'s model of agent $j$ takes agent $i$'s observation history as input and outputs a probability distribution over $j$'s actions. It is trained by supervised learning on the actions $j$ was seen taking, with loss $-\log\hat\pi_j^i(a_j \mid h_i)$.
>
> (d) LOLA assumes the opponent will take its own gradient step and optimises its return against the opponent's anticipated updated policy, so its update also shapes how the opponent learns.

## Flashcards

> [!card]- When is a training or execution phase called centralised?
> When an agent uses centrally shared information of the others: joint history, observations, global state or joint policy.

> [!card]- Why does centralised training and execution scale badly?
> The joint action space grows exponentially with the number of agents.

> [!card]- Name the disadvantages of decentralised training and execution.
> Non-stationarity, suboptimal equilibria, cannot tell which changes come from other agents.

> [!card]- Name the disadvantages of CTDE.
> Centralised training can be a bottleneck; coordination is hard when local information is insufficient at execution.

> [!card]- Which mode fits agents that cannot communicate when acting but have full information in a simulator?
> Centralised training with decentralised execution (CTDE).

> [!card]- In a centralised actor-critic, which network gets the centralised information $z$?
> The critic $V(h_i, z)$; the actor $\pi(a_i \mid h_i)$ uses only the agent's own history.

> [!card]- In the multi-agent policy gradient, what does the action evaluation condition on?
> The full history and the joint action of all agents.

> [!card]- Why is a CTDE version of REINFORCE uncommon?
> It has no critic, and its policy must stay decentralised, so nothing can be centralised.

> [!card]- What does a centralised action-value critic support that a value critic does not?
> Counterfactual reasoning: my alternative actions with the others' actions held fixed.

> [!card]- Which problem remains for a centralised action-value function in common-reward games?
> Credit assignment: which agent's action earned the shared reward.

> [!card]- What does the IGM property state?
> The argmax of the joint value equals the tuple of each agent's argmax of its own utility.

> [!card]- Name the two implications of IGM.
> Greedy decentralised execution; cheap computation of the greedy joint action.

> [!card]- What does VDN assume about the reward?
> The shared reward is a sum of individual reward terms.

> [!card]- Why does VDN satisfy IGM?
> The max of a sum of separately chosen terms is the sum of their maxima.

> [!card]- Why can VDN not learn a task where agents must match actions?
> An action's value depends on the partner's action, which a sum cannot express.

> [!card]- What condition must QMIX's mixing network satisfy?
> Monotonicity: $\partial Q / \partial Q_i > 0$ for every agent's utility.

> [!card]- Where do QMIX's mixing weights come from?
> A hypernetwork whose input is the centralised information $z$ (the global state).

> [!card]- What can neither VDN nor QMIX represent?
> A game where one agent's best action flips with the other agent's action.

> [!card]- What are the input and output of agent $i$'s deep model of agent $j$?
> Input: agent $i$'s observation history. Output: a distribution over agent $j$'s actions.

> [!card]- How is a deep agent model trained?
> Supervised, on the other agent's observed actions (negative log-likelihood).

> [!card]- What does LOLA anticipate that a naive learner ignores?
> The opponent's next learning step.

> [!card]- When are agents weakly homogeneous?
> When permuting policies between agents, for any joint policy, leaves the returns unchanged.

> [!card]- When are agents strongly homogeneous?
> When the optimal joint policy consists of identical policies.

> [!card]- Which homogeneity does parameter sharing need, and which experience sharing?
> Parameter sharing: strong. Experience sharing: weak.

> [!card]- Name two benefits of parameter sharing.
> Constant parameter count as agents are added; every agent's experience trains the shared network.

> [!card]- Why can experience sharing be preferred over parameter sharing?
> Each agent keeps its own network, so the learned policies can differ.

> [!card]- What lets one shared network give different agents different behaviour?
> Adding the agent index to each agent's observation.

> [!card]- Name the three steps of an MCTS simulation.
> Selection, expansion, backpropagation.

> [!card]- What does self-play require of the game?
> Symmetric roles and egocentric observations, so one policy can play both sides.

> [!card]- True or false: CTDE agents need a central controller to choose actions at execution.
> **False.** Shared information is used only in training.

> [!card]- True or false: QMIX's mixing function must be linear in each utility.
> **False.** It must be monotonic; linear (a sum) is the VDN special case.

> [!card]- True or false: parameter sharing requires agents to exchange observations during execution.
> **False.** Each agent runs its own copy of the shared network on its own observations.

> [!card]- Multiple selection: which statements about parameter sharing are correct? (A) All agents are constrained to the same policy (B) It works best when optimal policies differ (C) The parameter count stays constant as agents are added (D) The shared parameters learn from all agents' experience
> **A, C and D.** B: it suits strongly homogeneous agents, whose optimal policies are identical.

> [!card]- Multiple selection: which are true of the IGM property? (A) Each agent can act greedily on its own utility (B) It requires a linear decomposition (C) It makes the greedy joint action cheap to compute (D) QMIX satisfies it through monotonic mixing
> **A, C and D.** B: a monotonic mix also satisfies IGM.

> [!card]- Multiple selection: which methods use centralised information only during training? (A) Centralised A2C (B) VDN (C) QMIX (D) Independent DQN
> **A, B and C.** Independent DQN uses no centralised information at all.

## Links

- **Course:** [[AGAI - Overview|Course overview]]
- **Previous lecture:** [[AGAI-L04 - Deep RL and Intro to Deep MARL]] (DQN, REINFORCE, A2C, PPO; independent DQN and its stale replay buffer, independent policy gradient)
- **Next:** [[AGAI-L06a - From Deep MARL to Reasoning]] (AlphaGo, AlphaZero, population-based training, PSRO, AlphaStar)
- **Earlier:** [[AGAI-L03 - Intro to Multi-Agent Reinforcement Learning]] (non-stationarity, credit assignment, equilibrium selection)
- **Lab:** [[AGAI-Lab02 - Learning Agents]]
- **Concepts:** [[Centralised Training with Decentralised Execution]], [[Individual-Global-Max Property]], [[Value Decomposition Networks (VDN)]], [[QMIX]], [[Agent Modelling]], [[Parameter Sharing]], [[Multi-Agent Credit Assignment]], [[Actor-Critic]], [[Advantage Actor-Critic (A2C)]], [[Policy Gradient Methods]], [[Independent DQN]], [[Monte Carlo Tree Search (MCTS)]], [[Self-Play]], [[AlphaGo Zero]]
- **Source:** Agentic Artificial Intelligence, Lecture 5, "Deep MARL", MSc AI, UvA, 10 November 2026
- **Reading:** Albrecht, Christianos and Schäfer, *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches* (MIT Press, 2024), Chapter 9
- **Paper named:** Foerster et al., "Learning with Opponent-Learning Awareness" (AAMAS 2017)
- **Mock exam:** AGAI Mock Midterm 2026, Deep MARL questions 23 to 25, 27 and 29
