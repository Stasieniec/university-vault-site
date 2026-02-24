---
type: concept
aliases: [DP, dynamic programming]
course: [RL]
tags: [tabular-methods, exam-topic]
status: complete
---

# Dynamic Programming

## Definition

> [!definition] Dynamic Programming
> **Dynamic Programming** refers to a collection of algorithms that compute optimal policies given a perfect model of the environment (i.e., the [[Markov Decision Process|MDP]] dynamics $p(s',r|s,a)$). DP uses the [[Bellman Equation]] as an update rule to iteratively improve value estimates.

> [!intuition] Core Idea
> DP turns the Bellman equation into an **assignment** (update rule). Instead of solving a system of equations, it repeatedly applies the Bellman equation as an update until convergence. "Sweep" through all states, update each one, repeat.

## Key Algorithms

### [[Policy Evaluation]] (Prediction)

Compute $v_\pi$ for a given policy $\pi$ by iterative application of the Bellman equation:

> [!formula] Iterative Policy Evaluation Update
> $$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)\left[r + \gamma V_k(s')\right] \quad \forall s \in \mathcal{S}$$
> 
> Repeat until $\max_s |V_{k+1}(s) - V_k(s)| < \theta$ (convergence threshold).

### [[Policy Iteration]]

Alternates between evaluation and improvement:
1. **Policy Evaluation**: Compute $v_\pi$ (iteratively until convergence)
2. **Policy Improvement**: $\pi'(s) = \arg\max_a q_\pi(s,a)$ (greedy w.r.t. current value function)
3. Repeat until policy is stable ($\pi' = \pi$)

Guaranteed to converge to optimal policy in finite number of iterations (for finite MDPs).

### [[Value Iteration]]

Combines evaluation and improvement into a single update:

> [!formula] Value Iteration Update
> $$V_{k+1}(s) = \max_a \sum_{s',r} p(s',r|s,a)\left[r + \gamma V_k(s')\right] \quad \forall s \in \mathcal{S}$$
> 
> Equivalent to: one sweep of policy evaluation + greedy policy improvement. Converges to $v_*$.

### [[Generalized Policy Iteration]] (GPI)

> [!definition] GPI
> Any interaction between policy evaluation and policy improvement, regardless of granularity. Value iteration, policy iteration, and most RL algorithms are instances of GPI.
> 
> ```
> Evaluation ←→ Improvement
>     ↓              ↓
>   v ≈ v_π      π ≈ greedy(v)
>     ↘              ↙
>        v* and π*
> ```

## Limitations

- **Requires full model**: Must know $p(s',r|s,a)$ for all transitions
- **Curse of dimensionality**: Sweeps over all states — infeasible for large/continuous state spaces
- **Full-width backups**: Each update considers all possible next states

> [!tip] Why DP Matters Despite Limitations
> DP provides the **theoretical foundation** for all of RL. MC and TD methods are essentially doing DP-like updates but with samples instead of expectations. Understanding DP is key to understanding everything else.

## Connections

- Solves: [[Bellman Equation]], [[Bellman Optimality Equation]]
- Requires: [[Markov Decision Process]] model
- Generalized by: [[Generalized Policy Iteration]]
- Sample-based alternatives: [[Monte Carlo Methods]], [[Temporal Difference Learning]]
- With approximation: [[Function Approximation]], [[Semi-Gradient Methods]]

## Appears In

- [[RL-L02 - Dynamic Programming]]
- [[RL-Book Ch4 - Dynamic Programming]]
- [[RL-CA01 - Dynamic Programming]]
- [[RL-ES01 - Exercise Set Week 1]]
