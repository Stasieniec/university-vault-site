---
type: concept
aliases: [Bellman equation, Bellman equations]
course: [RL]
tags: [foundations, key-formula, exam-topic]
status: complete
---

# Bellman Equation

## Definition

> [!definition] Bellman Equation
> The **Bellman equation** expresses the value of a state (or state-action pair) as the immediate reward plus the discounted value of the successor state. It captures the recursive structure of the [[Value Function]]: the value of a state depends on the values of its possible successor states.

## Bellman Equation for $v_\pi$ (State-Value)

> [!formula] Bellman Equation for $v_\pi$
> $$v_\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) \left[ r + \gamma \, v_\pi(s') \right]$$
> 
> where:
> - $v_\pi(s)$ — value of state $s$ under [[Policy]] $\pi$
> - $\pi(a|s)$ — probability of taking action $a$ in state $s$
> - $p(s', r | s, a)$ — [[Markov Decision Process|MDP]] dynamics
> - $r$ — immediate reward
> - $\gamma$ — [[Discount Factor]]
> - $v_\pi(s')$ — value of successor state

> [!intuition] Reading the Equation
> "The value of a state = weighted average over all actions I might take (weighted by my policy) of: [the immediate reward I get + discounted value of where I end up]."
> 
> It's an **expectation** over the next step, then recursion handles the rest. This is the key insight — you don't need to look all the way to the end. Just one step ahead + the value of where you land.

**Alternative compact form:**
$$v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma \, v_\pi(S_{t+1}) \mid S_t = s]$$

## Bellman Equation for $q_\pi$ (Action-Value)

> [!formula] Bellman Equation for $q_\pi$
> $$q_\pi(s, a) = \sum_{s', r} p(s', r | s, a) \left[ r + \gamma \sum_{a'} \pi(a'|s') \, q_\pi(s', a') \right]$$
> 
> or equivalently:
> $$q_\pi(s, a) = \mathbb{E}_\pi[R_{t+1} + \gamma \, q_\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a]$$

The relationship between $v_\pi$ and $q_\pi$:
$$v_\pi(s) = \sum_{a} \pi(a|s) \, q_\pi(s, a)$$

## Bellman Optimality Equations

> [!formula] Bellman Optimality Equation for $v_*$
> $$v_*(s) = \max_{a} \sum_{s', r} p(s', r | s, a) \left[ r + \gamma \, v_*(s') \right]$$
> 
> The optimal value of a state is achieved by always picking the **best** action.

> [!formula] Bellman Optimality Equation for $q_*$
> $$q_*(s, a) = \sum_{s', r} p(s', r | s, a) \left[ r + \gamma \max_{a'} q_*(s', a') \right]$$
> 
> The optimal action-value: immediate reward + the best you can do from the next state.

> [!intuition] Optimality = Max Instead of Average
> Compare the regular Bellman equation (average over policy) vs optimality equation (max over actions). Regular: "What do I expect under my current policy?" Optimal: "What's the best I could ever do?"

## Backup Diagrams

**Bellman equation for $v_\pi$:**
```
         (s)          ← state node (open circle)
        / | \
       a₁ a₂ a₃      ← action nodes (solid dots), weighted by π(a|s)
      /|  |  |\
    s' s' s' s' s'    ← next states, weighted by p(s',r|s,a)
```
White circles = state nodes, black dots = action nodes. Each branch represents one possible action and one possible next-state transition.

**Bellman optimality for $v_*$:**
```
         (s)
        / | \
       a₁ a₂ a₃      ← MAX over actions (arc across branches)
      /|  |  |\
    s' s' s' s' s'    ← weighted by p(s',r|s,a)
```
The "max" replaces the weighted average over actions.

## Solving Bellman Equations

| Method | Approach | Requires Model? |
|--------|----------|----------------|
| [[Dynamic Programming]] | Iterative solution of Bellman equations | Yes |
| [[Monte Carlo Methods]] | Sample-based estimation of expectations | No |
| [[Temporal Difference Learning]] | Bootstrapped sample-based estimation | No |

The Bellman equation is a system of $|\mathcal{S}|$ linear equations (for $v_\pi$) — solvable directly for small state spaces, iteratively for large ones.

## Key Properties

- **Linearity**: The Bellman equation for $v_\pi$ is linear in $v_\pi$ (it's $v = r_\pi + \gamma P_\pi v$ in matrix form)
- **Contraction**: The Bellman operator is a $\gamma$-contraction mapping → unique fixed point, iterative methods converge
- **Foundation**: Every RL algorithm is essentially approximating or solving some form of Bellman equation

> [!warning] Common Exam Mistake
> Don't mix up the Bellman equation (for a given policy $\pi$) with the Bellman **optimality** equation (for the optimal policy $\pi_*$). The first uses $\sum_a \pi(a|s)$, the second uses $\max_a$.

## Connections

- Defined on: [[Markov Decision Process]]
- Solved by: [[Dynamic Programming]], [[Policy Iteration]], [[Value Iteration]]
- Approximated by: [[Temporal Difference Learning]], [[Monte Carlo Methods]]
- Extended: [[Bellman Error]], [[Bellman Optimality Equation]]

## Appears In

- [[RL-L01 - Intro, MDPs & Bandits]]
- [[RL-L02 - Dynamic Programming]]
- [[RL-Book Ch3 - Finite MDPs]], [[RL-Book Ch4 - Dynamic Programming]]
- [[RL-ES01 - Exercise Set Week 1]]
