---
type: concept
aliases: [PSRO, Policy Space Response Oracles, policy space response oracles, Policy Space Response Oracle, meta-game]
course: [AGAI]
tags: [multi-agent, marl, game-theory, exam-topic]
status: complete
---

# Policy Space Response Oracles (PSRO)

Source: AGAI Lecture 6 (after Lanctot et al., "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning", NeurIPS 2017), the MARL book (Albrecht, Christianos and Schäfer, Ch 9, population-based training, Algorithm 26) and the AGAI mock midterm key.

## Definition

> [!definition] PSRO
> A [[Population-Based Training|population-based training]] method for multi-agent games. Each agent keeps a **population of policies**, started for example with random policies. PSRO repeatedly builds a **meta-game**, a simplified **normal-form game whose actions are the policies in the current populations** and whose payoffs are the average returns when those policies play each other; solves it with a **meta-solver** to get a distribution over each population; and adds, for each agent, a new policy trained (by an "oracle", typically deep RL) as a **best response** to the other agents' meta-game distributions.

## Intuition

> [!intuition] Beat the whole history of strategies
> Plain self-play trains only against the current version of oneself. In games with non-transitive strategies (B beats A, C beats B, A beats C, as in rock-paper-scissors) that can cycle and forget how to beat old strategies. PSRO keeps all the strategies found so far, works out which mixture of them is strongest (by solving a small matrix game), and trains a new strategy that beats that mixture.

## Mathematical Formulation

Populations $\Pi_1, \dots, \Pi_n$, one per agent. Each iteration:

1. **Meta-game:** for every combination $(\pi_1, \dots, \pi_n) \in \Pi_1 \times \dots \times \Pi_n$, estimate each agent's expected return $U_i(\pi_1, \dots, \pi_n)$ by simulating episodes. This is a [[Normal-Form Game|normal-form game]] with action sets $\Pi_i$.
2. **Meta-solver:** compute a distribution $\sigma_i$ over $\Pi_i$ for each agent, for example a [[Nash Equilibrium|Nash equilibrium]] of the meta-game.
3. **Oracle:** for each agent $i$, train a new policy $\pi_i'$ that maximises its expected return when the others play policies drawn from $\sigma_{-i}$: an approximate [[Best Response|best response]].
4. Add $\pi_i'$ to $\Pi_i$ and repeat.

The MARL book's version, as taught in AGAI Lecture 6:

```pseudo
Algorithm: Policy space response oracles (PSRO)
Initialize populations Π_i^1 for all agents i ∈ I (e.g. random policies)
For each generation k = 1, 2, 3, ...:
  Construct meta-game M^k from current populations {Π_i^k}
  Use meta-solver on M^k to obtain distributions {δ_i^k}
  For each agent i:                                  // train best-response policies
    For each episode e = 1, 2, 3, ...:
      Sample policies for the other agents π_{−i} ~ δ_{−i}^k
      Use single-agent RL to train π_i' with respect to π_{−i} in the underlying game G
  Grow populations Π_i^{k+1} ← Π_i^k ∪ {π_i'}
```

Each generation adds one row and column per new policy to the meta-game; their payoffs against the older policies must be simulated before the next meta-solve.

## Key Properties / Variants

- The meta-game is small (one action per stored policy) even when the underlying game is huge.
- The choice of meta-solver changes the method: a Nash meta-solver gives the double oracle idea; a uniform distribution over the population is closer to fictitious play (training against the average of all past policies).
- Generalises self-play, which always responds to the latest policy only.
- AlphaStar (StarCraft II, DeepMind 2019) trains in a PSRO-like league: self-play with heterogeneous populations.
- On rock-paper-scissors starting from {Rock}, PSRO adds Paper, then Scissors, then stops at the uniform Nash equilibrium, where naive self-play would cycle (worked in [[AGAI-L06a - From Deep MARL to Reasoning]]).

## Connections

- Game theory: [[Normal-Form Game]], [[Nash Equilibrium]], [[Best Response]], [[Stochastic Game]]
- Self-play in zero-sum games: [[Self-Play]], [[AlphaGo Zero]], [[Monte Carlo Tree Search (MCTS)]]
- The broader family: [[Population-Based Training]]
- Learning the best responses: [[Deep Reinforcement Learning]]

## Appears In

- [[AGAI-L05 - Deep MARL, Part II]] (from the MARL book and the mock exam key)
- [[AGAI-L06a - From Deep MARL to Reasoning]] (the lecture that teaches PSRO: meta-game, algorithm, AlphaStar)
