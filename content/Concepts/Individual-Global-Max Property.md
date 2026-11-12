---
type: concept
aliases: [IGM, IGM property, Individual-Global-Max, individual-global-max property]
course: [AGAI]
tags: [multi-agent, marl, exam-topic]
status: complete
---

# Individual-Global-Max Property

## Definition

> [!definition] Individual-Global-Max (IGM) property
> A value decomposition satisfies IGM if
> $$\arg\max_{a_1, \dots, a_n} Q(s, a_1, \dots, a_n) = \Big(\arg\max_{a_1}Q_1(o_1, a_1),\; \dots,\; \arg\max_{a_n}Q_n(o_n, a_n)\Big)$$
> The **joint action maximising the joint action-value function** equals the **set of individual actions maximising the individual action-value functions** (utilities).

## Intuition

> [!intuition] Selfish greed that adds up
> In a cooperative team with a shared reward, we want each agent to pick its action alone, from what it sees, and still land on the best joint action. IGM is the condition that makes this work: if every agent simply takes the action its own utility rates highest, the team plays the action the joint value rates highest. Nobody has to search the joint action space or talk to anyone at execution time.

## Mathematical Formulation

- $Q(s, a_1, \dots, a_n)$: the joint (team) action-value function, over all agents' actions
- $Q_i(o_i, a_i)$: agent $i$'s individual utility, from its own observation (or history $h_i$) and its own action
- the right-hand side is a tuple: each agent maximises separately

Two sufficient ways to get IGM:

- **Linear:** $Q = \sum_i Q_i$ (VDN). The maximum of a sum of terms in separate variables is the sum of their maxima.
- **Monotonic:** $Q = f_{\text{mix}}(Q_1, \dots, Q_n)$ with $\partial Q / \partial Q_i > 0$ for all $i$ (QMIX). Raising any utility never lowers $Q$, so maximising each utility maximises $Q$.

> [!example] A linear game
> Shared rewards (A,A) = 1, (A,B) = (B,A) = 5, (B,B) = 9. Utilities $Q_1 = (0.88, 4.88)$ and $Q_2 = (0.12, 4.12)$ for (A, B) sum to every cell. Each agent's argmax is B, and (B, B) = 9 is the joint argmax: IGM holds.

## Key Properties / Variants

- **Implication 1, decentralised execution:** each agent follows the greedy policy with respect to its own utility.
- **Implication 2, efficiency:** the greedy joint action is found with $n$ maximisations over $|A_i|$ values instead of one over $\prod_i |A_i|$ joint actions. This also makes the max in Q-learning targets tractable during training.
- IGM is a criterion for judging a decomposition: utilities can be far from actual returns (they are not unique) and still satisfy it.
- Linear and monotonic decompositions are sufficient, not necessary; methods such as QTRAN aim at IGM beyond them.

## Connections

- Decompositions that satisfy it: [[Value Decomposition Networks (VDN)]], [[QMIX]]
- Training mode: [[Centralised Training with Decentralised Execution]]
- Problem it addresses: [[Multi-Agent Credit Assignment]]
- Single-agent basis: [[Q-Learning]], [[Deep Q-Network (DQN)]]

## Appears In

- [[AGAI-L05 - Deep MARL, Part II]]
