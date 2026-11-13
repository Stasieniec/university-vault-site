---
type: concept
aliases: [population-based training, Population based training, PBT, league training, AlphaStar league]
course: [AGAI]
tags: [multi-agent, marl, deep-rl, exam-topic]
status: complete
---

# Population-Based Training

## Definition

> [!definition] Population-based training
> A multi-agent training approach that tests one policy against a **population** of other policies. Each agent has its own population, and the policies in it can represent agents with **different roles, actions and observations**. General procedure: **initialise** one population per agent (for example with uniform policies), **evaluate** the current policies against the policies in the other populations, **modify** existing policies (random changes, copying the parameters of the best-performing policies), and repeat.

## Intuition

> [!intuition] Train against a crowd
> [[Self-Play|Self-play]] trains one policy against itself, which only works when both players have the same role. A population lets each side have its own kind of policy, and training against many opponents instead of one keeps a policy from overfitting to a single opponent's quirks. The update rule is evolutionary: what does well against the current populations spreads, and random changes add variation, the same logic as the population dynamics of evolutionary game theory.

## Mathematical Formulation

Populations $\Pi_1, \dots, \Pi_n$, one per agent. Evaluating policy $\pi_i \in \Pi_i$ means estimating its expected return $U_i(\pi_i, \pi_{-i})$ against policies $\pi_{-i}$ drawn from the other populations, by simulating episodes. How the opponents are drawn and how the policies are modified defines the method:

- **Evolutionary modification:** replace weak policies by copies of strong ones, then perturb them.
- **[[Policy Space Response Oracles (PSRO)|PSRO]]:** build a meta-game (a normal-form game whose actions are the policies), solve it with a meta-solver such as a Nash equilibrium, and add a best response to the solution trained with RL.

## Key Properties / Variants

- **Handles asymmetric roles**, which self-play cannot.
- **Keeps old strategies around**, which protects against the cycling of self-play in non-transitive games.
- **AlphaStar's league** (StarCraft II, DeepMind 2019) is the course's PSRO-like example: self-play with heterogeneous populations. (From outside the course material: the league kept frozen copies of past agents as opponents, added "exploiter" agents trained to find the weaknesses of the main agents and of the league as a whole, and chose opponents by how often the learner lost to them.)
- Expensive: every policy pairing has to be evaluated by simulation, and each new policy is a training run.

## Connections

- Special case and starting point: [[Self-Play]], [[AlphaGo Zero]]
- Specific algorithm: [[Policy Space Response Oracles (PSRO)]]
- Game theory behind the meta-game: [[Normal-Form Game]], [[Nash Equilibrium]], [[Best Response]]
- Population dynamics: [[Replicator Dynamics]]

## Appears In

- [[AGAI-L06a - From Deep MARL to Reasoning]]
- [[AGAI-L05 - Deep MARL, Part II]] (briefly, with PSRO)
