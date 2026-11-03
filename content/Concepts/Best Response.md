---
type: concept
aliases: [best response, best responses, best-response policy, best-response]
course: [AGAI]
tags: [game-theory, multi-agent, exam-topic]
status: complete
---

# Best Response

## Definition

> [!definition] Best response
> A **best-response policy** $\pi_i$ maximises the expected return for agent $i$ against a given set of policies for all other agents, $\pi_{-i} = (\pi_1, \dots, \pi_{i-1}, \pi_{i+1}, \dots, \pi_n)$:
> $$BR_i(\pi_{-i}) = \arg\max_{\pi_i} U_i(\langle \pi_i, \pi_{-i} \rangle)$$
> where $\langle \pi_i, \pi_{-i} \rangle$ is the complete joint policy.

## Intuition

> [!intuition] The best I can do, given what you do
> In a game no action is good or bad on its own: its value depends on what the others play. A best response fixes the others' policies and asks what agent $i$ should do against them. In the [[Stag Hunt]], C is the best response to C and D the best response to D; in the [[Prisoner's Dilemma]], D is the best response to everything, which makes D a dominant action.

> [!example] Reading best responses off a matrix
> | | C | D |
> |---|---|---|
> | **C** | 4, 4 | 0, 3 |
> | **D** | 3, 0 | 2, 2 |
>
> For the row agent: against column C compare 4 and 3, best response C; against column D compare 0 and 2, best response D. The column agent's best responses are the same by symmetry. Cells where both agents are best-responding, (C, C) and (D, D), are the pure Nash equilibria.

## Mathematical Formulation

$$BR_i(\pi_{-i}) = \arg\max_{\pi_i} U_i(\langle \pi_i, \pi_{-i} \rangle)$$

where:
- $\pi_{-i}$: the policies of all agents except $i$, held fixed
- $U_i$: agent $i$'s expected return
- $\arg\max$: the set of maximisers; several policies can tie, so $BR_i$ is a set

**Against a mixed policy.** If the partner in a symmetric two-action game plays C with probability $p$, agent $i$'s expected payoffs are $u_C(p) = pR + (1-p)S$ for C and $u_D(p) = pT + (1-p)P$ for D. The best response is C when $u_C > u_D$, D when $u_D > u_C$, and any mix when they are equal. A mixed policy can only be a best response if every action it uses earns the same, which is how mixed equilibria are computed.

## Key Properties / Variants

- **Nash equilibrium = mutual best response:** a joint policy where every agent's policy is a best response to the others'. See [[Nash Equilibrium]].
- **Dominant action:** an action that is a best response to every policy of the others (D in the Prisoner's Dilemma).
- **Relative to fixed opponents:** if the others change their policies, the best response can change, which is why learning agents face a moving target (non-stationarity).

## Connections

- Built on: [[Normal-Form Game]], [[Policy]]
- Leads to: [[Nash Equilibrium]]
- Games: [[Prisoner's Dilemma]], [[Stag Hunt]]
- Setting: [[Multi-Agent System]], [[Stochastic Game]]

## Appears In

- [[AGAI-L02 - Game Theory and Population Dynamics]]
- [[AGAI-Lab01 - Simple Agents, Complex Dynamics]]
