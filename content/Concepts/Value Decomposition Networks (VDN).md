---
type: concept
aliases: [VDN, Value Decomposition Networks, value decomposition networks, linear value decomposition, value decomposition]
course: [AGAI]
tags: [multi-agent, marl, deep-rl, exam-topic]
status: complete
---

# Value Decomposition Networks (VDN)

## Definition

> [!definition] Value decomposition networks
> A CTDE value-based method for common-reward games that represents the joint action-value function as the **sum of individual utilities**, one per agent, each conditioned only on that agent's history and action:
> $$Q(h, z, a;\theta) = \sum_{i \in I} Q(h_i, a_i;\theta_i)$$
> The utilities are trained jointly so that their sum predicts the team value.

## Intuition

> [!intuition] Learn agent-level values whose sum predicts the team value
> The team gets one reward. VDN never tells an agent what its own share is; it only asks that the agents' utilities add up to the team's value, and lets gradient descent work out the split. Because the joint value is a sum, the best joint action is just every agent's own best action, so each agent can act alone at execution.

## Mathematical Formulation

If the shared reward is a sum of unobserved individual terms, $r^t = \bar r_1^t + \dots + \bar r_n^t$, linearity of expectation gives

$$Q(h^t, z^t, a^t;\theta) = \mathbb{E}\Big[\sum_{\tau=t}^{\infty}\gamma^{\tau-t}\sum_{i}\bar r_i^\tau \,\Big|\, h^t, z^t, a^t\Big] = \sum_i \mathbb{E}\Big[\sum_{\tau=t}^{\infty}\gamma^{\tau-t}\bar r_i^\tau \,\Big|\, h^t, z^t, a^t\Big] \approx \sum_i Q(h_i^t, a_i^t;\theta_i)$$

the last step approximating each agent's share by a utility of its own history and action.

Training, from a shared replay buffer of transitions $(h^k, a^k, r^k, h^{k+1})$:

$$y^k = r^k + \gamma\sum_i\max_{a_i'}Q(h_i^{k+1}, a_i';\bar\theta_i), \qquad \mathcal{L}(\theta) = \frac{1}{B}\sum_{k=1}^{B}\Big(y^k - \sum_i Q(h_i^k, a_i^k;\theta_i)\Big)^2$$

- $\bar\theta_i$: target network parameters; $y^k = r^k$ at terminal states
- $B$: mini-batch size; the loss is minimised over all $\theta_i$ together
- action selection: ε-greedy on each agent's own $Q(h_i, \cdot;\theta_i)$

## Key Properties / Variants

- Satisfies the [[Individual-Global-Max Property|IGM property]], so execution is decentralised and greedy, and the max in the target is a sum of individual maxima.
- Needs no centralised information $z$ in the network; centralisation is in the shared loss and buffer.
- Utilities are not unique (shift a constant between agents) and need not equal real returns.
- **Weakness: additivity is restrictive.** Tasks that need specific combinations of actions, such as two agents choosing matching actions, cannot be represented by a sum. In a matching game with (A,A) = (B,B) = 10 and 0 otherwise, the best additive fit values every joint action at 5.
- Generalised by [[QMIX]], which replaces the sum with a learned monotonic mixing function (VDN is the special case with the sum).

## Connections

- Criterion: [[Individual-Global-Max Property]]
- Generalisation: [[QMIX]]
- Base algorithm: [[Deep Q-Network (DQN)]], [[Q-Learning]]
- Training mode: [[Centralised Training with Decentralised Execution]]
- Problem addressed: [[Multi-Agent Credit Assignment]]

## Appears In

- [[AGAI-L05 - Deep MARL, Part II]]
