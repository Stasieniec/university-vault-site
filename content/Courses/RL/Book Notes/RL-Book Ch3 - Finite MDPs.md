---
type: book-chapter
course: RL
book: "Reinforcement Learning: An Introduction (2nd ed.)"
chapter: 3
sections: ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8"]
topics:
  - "[[Markov Decision Process]]"
  - "[[Value Function]]"
  - "[[Bellman Equation]]"
  - "[[Bellman Optimality Equation]]"
  - "[[Optimal Policy]]"
  - "[[Return]]"
  - "[[Discount Factor]]"
  - "[[Policy]]"
  - "[[Reward Signal]]"
  - "[[State Space]]"
status: complete
---

# Chapter 3: Finite Markov Decision Processes

## Overview
This chapter formalizes the problem of **sequential decision making** through the framework of **Finite Markov Decision Processes (MDPs)**. Unlike bandit problems, MDPs involve an associative aspect—choosing different actions in different situations—and account for **delayed rewards**, where current actions influence future states and thus future rewards.

## 3.1 The Agent–Environment Interface
The [[Reinforcement Learning]] problem is framed as a continuous interaction between two entities:
- **Agent**: The learner and decision-maker.
- **Environment**: Everything outside the agent.

### The Interaction Loop
At each discrete time step $t=0, 1, 2, \dots$:
1. The agent receives a representation of the environment's **state**, $S_t \in \mathcal{S}$.
2. The agent selects an **action**, $A_t \in \mathcal{A}(s)$.
3. One time step later, the agent receives a numerical **reward**, $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$, and finds itself in a new state $S_{t+1}$.

This creates a trajectory: $S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \dots$

> [!definition] Dynamics Function
> In a **Finite MDP**, the sets $\mathcal{S}, \mathcal{A},$ and $\mathcal{R}$ are finite. The environment's dynamics are completely defined by the probability:
> $$p(s', r | s, a) \doteq \Pr\{S_t = s', R_t = r \mid S_{t-1} = s, A_{t-1} = a\}$$
> for all $s', s \in \mathcal{S}$, $r \in \mathcal{R}$, and $a \in \mathcal{A}(s)$.

> [!formula] Derived Dynamics Functions
> - **State-transition probabilities**: $p(s' | s, a) = \sum_{r \in \mathcal{R}} p(s', r | s, a)$
> - **Expected rewards for state-action pairs**: $r(s, a) = \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r | s, a)$
> - **Expected rewards for state-action-next-state triples**: $r(s, a, s') = \sum_{r \in \mathcal{R}} r \frac{p(s', r | s, a)}{p(s' | s, a)}$

> [!intuition] The Markov Property
> The state $S_t$ must include information about all aspects of the past that matter for the future. If the transition probabilities depend only on the *immediately preceding* state and action, the state is said to have the **Markov Property**.

> [!example] Recycling Robot
> A robot collects empty cans. States: $\mathcal{S} = \{high, low\}$ battery levels. Actions: $\{search, wait, recharge\}$. 
> - Searching retrieves cans (reward) but drains battery. 
> - Waiting retrieves fewer cans but saves battery. 
> - Recharging is only possible when battery is low.
> - **Transition Graph**: State nodes (large circles) and Action nodes (small solid circles) show the probabilities $p(s' | s, a)$ and rewards $r$.

## 3.2 Goals and Rewards
The agent's goal is to maximize the **cumulative reward** in the long run.

> [!warning] The Reward Hypothesis
> All of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward).

**Key Insight**: Rewards should communicate *what* you want the agent to achieve, not *how* you want it achieved. Sub-goals (e.g., in Chess) should not be rewarded directly if they might conflict with the ultimate goal (winning).

## 3.3 Returns and Episodes
The [[Return]], $G_t$, is the specific function of the reward sequence the agent seeks to maximize.

### Episodic Tasks
Tasks that break into finite subsequences ending in a **terminal state**.
$$G_t \doteq R_{t+1} + R_{t+2} + \dots + R_T$$

### Continuing Tasks
Tasks that go on without limit ($T = \infty$). We must use a [[Discount Factor]] $\gamma \in [0, 1]$ to keep the return finite.
$$G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

> [!formula] Recursive Relation of Returns
> $$G_t = R_{t+1} + \gamma G_{t+1}$$

## 3.4 Unified Notation
We unify episodic and continuing tasks by treating episode termination as entering a **special absorbing state** that transitions only to itself and generates zero rewards. This allows using the discounted sum formula for both.

## 3.5 Policies and Value Functions
A [[Policy]], $\pi$, is a mapping from states to probabilities of selecting each possible action: $\pi(a|s)$.

A [[Value Function]] estimates "how good" it is to be in a certain state or perform a certain action.

> [!definition] State-Value Function ($v_\pi$)
> The expected return when starting in state $s$ and following policy $\pi$:
> $$v_\pi(s) \doteq \mathbb{E}_\pi [G_t \mid S_t = s] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t = s \right]$$

> [!definition] Action-Value Function ($q_\pi$)
> The expected return starting from $s$, taking action $a$, and thereafter following policy $\pi$:
> $$q_\pi(s, a) \doteq \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]$$

### The Bellman Equation for $v_\pi$
Values of states satisfy recursive relationships.
> [!formula] Bellman Equation
> $$v_\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

> [!intuition] Backup Diagrams
> Backup diagrams show the flow of value. For $v_\pi$:
> - Root node: State $s$ (open circle).
> - Branches: Actions $a$ chosen via $\pi$.
> - Leaves: Next states $s'$ (open circles) and rewards $r$ determined by $p$.
> The value of $s$ is the average of the discounted values of successor states plus expected immediate rewards.

> [!example] Gridworld
> - **Dynamics**: 5x5 grid. Actions: N, S, E, W. Bumping into the edge results in -1 reward.
> - **Special States**: Transitions from A give +10 and land in A'. Transitions from B give +5 and land in B'.
> - **Value Function**: Under a random policy ($\pi=0.25$ for all actions), states near edges have lower/negative values, while states near A have high values.

## 3.6 Optimal Policies and Optimal Value Functions
An [[Optimal Policy]], $\pi_*$, is better than or equal to all other policies ($\pi_* \geq \pi \iff v_{\pi_*}(s) \geq v_\pi(s)$ for all $s \in \mathcal{S}$).

> [!formula] Optimal Value Functions
> - **Optimal State-Value**: $v_*(s) \doteq \max_\pi v_\pi(s)$
> - **Optimal Action-Value**: $q_*(s, a) \doteq \max_\pi q_\pi(s, a)$
>   - Note: $q_*(s, a) = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a]$

### The Bellman Optimality Equation
> [!formula] Bellman Optimality Equation for $v_*$
> $$v_*(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_*(s')]$$

> [!formula] Bellman Optimality Equation for $q_*$
> $$q_*(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \max_{a'} q_*(s', a')]$$

> [!intuition] Greedy Policies
> Once $v_*$ is known, the optimal policy is **greedy** with respect to it. Because $v_*$ already accounts for all future rewards, a local one-step search is sufficient for global optimality. With $q_*$, even the one-step search is unnecessary; the agent simply picks $\arg \max_a q_*(s, a)$.

## 3.7 Optimality and Approximation
Finding the exact solution to the [[Bellman Optimality Equation]] is often impossible due to:
1. **Model Uncertainty**: We don't know the dynamics $p$.
2. **Computational Constraints**: The [[State Space]] is too large (e.g., Chess, Backgammon).
3. **Memory Constraints**: We cannot store a table for all states.

RL methods focus on **approximating** $v_*$ and $\pi_*$, often prioritizing frequently visited states to make the best use of limited resources.

## Summary
- **MDPs** provide a mathematical framework for goal-directed learning.
- **Value functions** ($v$ and $q$) are central, representing expected future returns.
- **Bellman equations** provide the recursive structure needed to compute these values.
- **Optimality** is an ideal reached through **greedy** behavior relative to optimal value functions.
- RL agents must usually settle for **approximations** due to the curse of dimensionality and unknown dynamics.