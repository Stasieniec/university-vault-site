---
type: concept
aliases: [agent modelling, agent modeling, Agent Modeling, opponent modelling, opponent modeling, Opponent Modelling, agent models, agent model, opponent shaping, LOLA, deep joint-action learning]
course: [AGAI]
tags: [multi-agent, marl, deep-rl, exam-topic]
status: complete
---

# Agent Modelling

## Definition

> [!definition] Agent modelling
> An agent **explicitly models the policies of the other agents**, so that it can condition its own action selection on their predicted behaviour and adapt when that behaviour changes. In deep MARL, agent $i$'s model of agent $j$ is a network $\hat\pi_j^i(a_j \mid h_i;\phi_j^i)$:
> - **input:** agent $i$'s own observation history $h_i$
> - **output:** a probability distribution over agent $j$'s actions
>
> It is trained on the actions agent $i$ has observed $j$ take.

**Opponent shaping** goes one step further: since the other agents are learning, an agent can shape their learning to its own advantage. The well-known method is **LOLA** (Learning with Opponent-Learning Awareness).

## Intuition

> [!intuition] Predicting the others instead of absorbing them
> An independent learner experiences the other agents only as noise in its rewards and transitions, averaged over whatever they did in the past. An agent with models asks directly "what is agent $j$ likely to do now, given what I see?" and picks the action that is best against that prediction. Opponent shaping asks a further question: "how will my move change what $j$ learns next?"

## Mathematical Formulation

**Model loss** (negative log-likelihood of the observed action):

$$\mathcal{L}(\phi_j^i) = -\log\hat\pi_j^i\big(a_j^t \mid h_i^t;\phi_j^i\big)$$

**Deep joint-action learning.** Agent $i$ learns a value over joint actions $Q(h_i, \langle a_i, a_{-i}\rangle;\theta_i)$ and averages it over the models' predictions:

$$AV(h_i, a_i;\theta_i) = \sum_{a_{-i}}Q\big(h_i, \langle a_i, a_{-i}\rangle;\theta_i\big)\prod_{j \neq i}\hat\pi_j^i\big(a_j \mid h_i;\phi_j^i\big)$$

- action selection: ε-greedy on $AV(h_i, \cdot)$
- target: $y_i = r_i + \gamma\max_{a_i'}AV(h_i', a_i';\bar\theta_i)$ ($r_i$ at terminal states)
- critic loss: $(y_i - Q(h_i, \langle a_i, a_{-i}\rangle;\theta_i))^2$ on the observed joint action
- with many agents the sum over $a_{-i}$ can be approximated by sampling the others' actions from the models

**LOLA** (two agents $i$, $j$; $J_i$, $J_j$ their expected returns as functions of both parameter vectors):

$$\theta_j^{t+1}(\theta_i^t) = \theta_j^t + \beta\nabla_{\theta_j}J_j(\theta_i^t, \theta_j^t), \qquad \theta_i^{t+1} = \theta_i^t + \alpha\nabla_{\theta_i}J_i\big(\theta_i^t, \theta_j^{t+1}(\theta_i^t)\big)$$

A naive learner uses $\nabla_{\theta_i}J_i(\theta_i^t, \theta_j^t)$. LOLA differentiates through the opponent's anticipated update, adding a shaping term: how $\theta_i$ moves the opponent's learning step, times what that move is worth to agent $i$.

## Key Properties / Variants

- Counts as decentralised training (agents learn from their own observations), but each agent must observe the others' actions, and needs one model per other agent: $n - 1$ extra networks per agent.
- Models predict behaviour; they do not copy the other agent's true parameters.
- LOLA needs the opponent's parameters and learning rule, or an estimate of them. (Outside the AGAI lecture: in the iterated Prisoner's Dilemma, LOLA agents learn reciprocal, tit-for-tat-like cooperation where naive learners defect.)
- Contrast with CTDE methods ([[Centralised Training with Decentralised Execution|centralised critics]], [[Value Decomposition Networks (VDN)|value decomposition]]), which infer the others' behaviour implicitly from training data.

## Connections

- Setting: [[Stochastic Game]], [[Multi-Agent System]]
- Alternatives: [[Independent Learning]], [[Centralised Training with Decentralised Execution]]
- Base algorithms: [[Deep Q-Network (DQN)]], [[Policy Gradient Methods]]
- Related: [[Best Response]], [[Nash Equilibrium]]

## Appears In

- [[AGAI-L05 - Deep MARL, Part II]]
