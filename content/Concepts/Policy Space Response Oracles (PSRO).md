---
type: concept
aliases: [PSRO, Policy Space Response Oracles, policy space response oracles, meta-game, population-based training]
course: [AGAI]
tags: [multi-agent, marl, game-theory, exam-topic]
status: complete
---

# Policy Space Response Oracles (PSRO)

Source: MARL book (Albrecht, Christianos and Schäfer, Ch 9, population-based training) and the AGAI mock midterm key; the AGAI lecture on deep MARL does not cover it.

## Definition

> [!definition] PSRO
> A population-based training method for multi-agent games. Each agent keeps a **population of policies**. PSRO repeatedly builds a **meta-game**, a **normal-form game whose actions are the policies in the current populations** and whose payoffs are the expected returns when those policies play each other; solves it with a **meta-solver** to get a distribution over each population; and adds, for each agent, a new policy trained (by an "oracle", typically deep RL) as a **best response** to the other agents' meta-game distributions.

## Intuition

> [!intuition] Beat the whole history of strategies
> Plain self-play trains only against the current version of oneself. In games with non-transitive strategies (B beats A, C beats B, A beats C, as in rock-paper-scissors) that can cycle and forget how to beat old strategies. PSRO keeps all the strategies found so far, works out which mixture of them is strongest (by solving a small matrix game), and trains a new strategy that beats that mixture.

## Mathematical Formulation

Populations $\Pi_1, \dots, \Pi_n$, one per agent. Each iteration:

1. **Meta-game:** for every combination $(\pi_1, \dots, \pi_n) \in \Pi_1 \times \dots \times \Pi_n$, estimate each agent's expected return $U_i(\pi_1, \dots, \pi_n)$ by simulating episodes. This is a [[Normal-Form Game|normal-form game]] with action sets $\Pi_i$.
2. **Meta-solver:** compute a distribution $\sigma_i$ over $\Pi_i$ for each agent, for example a [[Nash Equilibrium|Nash equilibrium]] of the meta-game.
3. **Oracle:** for each agent $i$, train a new policy $\pi_i'$ that maximises its expected return when the others play policies drawn from $\sigma_{-i}$: an approximate [[Best Response|best response]].
4. Add $\pi_i'$ to $\Pi_i$ and repeat.

## Key Properties / Variants

- The meta-game is small (one action per stored policy) even when the underlying game is huge.
- The choice of meta-solver changes the method: a Nash meta-solver gives the double oracle idea; a uniform distribution over the population is closer to fictitious play (training against the average of all past policies).
- Generalises self-play, which always responds to the latest policy only.
- A population-based league of this kind underlies AlphaStar (StarCraft II).

## Connections

- Game theory: [[Normal-Form Game]], [[Nash Equilibrium]], [[Best Response]], [[Stochastic Game]]
- Self-play in zero-sum games: [[AlphaGo Zero]], [[Monte Carlo Tree Search (MCTS)]]
- Learning the best responses: [[Deep Reinforcement Learning]]

## Appears In

- [[AGAI-L05 - Deep MARL, Part II]] (from the MARL book and the mock exam key)
