---
type: book-chapter
course: RL
book: "Reinforcement Learning: An Introduction (2nd ed.)"
chapter: 10
sections: ["10.1", "10.2", "10.3", "10.4", "10.5", "10.6"]
topics:
  - "[[Episodic Semi-Gradient Control]]"
  - "[[Semi-Gradient Methods]]"
  - "[[Function Approximation]]"
  - "[[SARSA]]"
status: complete
---

# RL-Book Ch10: On-policy Control with Approximation

## Overview
In this chapter, we extend [[Function Approximation]] from state-value prediction to **action-value control**. We transition from estimating $v_\pi(s)$ to $q_\pi(s, a)$. While the extension is straightforward for the episodic case, the continuing case requires a shift from discounting to the **average-reward** formulation.

## 10.1 Episodic Semi-gradient Control
The semi-gradient methods developed in Chapter 9 for prediction are extended to control by approximating the action-value function $q(s, a, \mathbf{w}) \approx q_\pi(s, a)$.

> [!formula] Semi-gradient Action-Value Update
> The general gradient-descent update for action-value prediction is:
> $$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha [U_t - q(S_t, A_t, \mathbf{w}_t)] \nabla q(S_t, A_t, \mathbf{w}_t)$$
> where $U_t$ is the update target (e.g., $G_t$ or a TD return).

### Episodic Semi-gradient Sarsa
For the one-step [[SARSA]] method, the update becomes:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha [R_{t+1} + \gamma q(S_{t+1}, A_{t+1}, \mathbf{w}_t) - q(S_t, A_t, \mathbf{w}_t)] \nabla q(S_t, A_t, \mathbf{w}_t)$$

> [!intuition] Policy Improvement
> To perform control, we use [[Epsilon-Greedy Policy]] with respect to the current approximate action values. For a state $s$, the greedy action is $A^* = \arg\max_a q(s, a, \mathbf{w})$.

### Algorithm: Episodic Semi-gradient Sarsa
```text
Input: a differentiable action-value function parameterization q : S x A x R^d -> R
Algorithm parameters: step size alpha > 0, small epsilon > 0
Initialize value-function weights w arbitrarily

Loop for each episode:
    S, A <- initial state and action of episode (e.g., epsilon-greedy)
    Loop for each step of episode:
        Take action A, observe R, S'
        If S' is terminal:
            w <- w + alpha * [R - q(S, A, w)] * grad_q(S, A, w)
            Go to next episode
        Choose A' as a function of q(S', ., w) (e.g., epsilon-greedy)
        w <- w + alpha * [R + gamma * q(S', A', w) - q(S, A, w)] * grad_q(S, A, w)
        S <- S', A <- A'
```

### Example: Mountain Car
The Mountain Car task involves driving an underpowered car up a steep hill. Because gravity is stronger than the engine, the agent must learn to drive away from the goal first to build momentum.
- **State**: Position and Velocity.
- **Actions**: Forward (+1), Reverse (-1), Zero (0).
- **Reward**: -1 per step until the goal is reached.
- **[[Function Approximation]]**: [[Linear Function Approximation]] with [[Tile Coding]].

**Learning Curves**: Typically show that intermediate step sizes $\alpha$ work best, and as learning progresses, the "cost-to-go" function (negative of the value function) accurately represents the time needed to reach the goal from different states.

## 10.2 n-step Semi-gradient Sarsa
The n-step return generalizes to function approximation by using the approximate action-value of the state-action pair $n$ steps ahead.

> [!formula] n-step Semi-gradient Return
> $$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n q(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1})$$

The update rule is:
$$\mathbf{w}_{t+n} = \mathbf{w}_{t+n-1} + \alpha [G_{t:t+n} - q(S_t, A_t, \mathbf{w}_{t+n-1})] \nabla q(S_t, A_t, \mathbf{w}_{t+n-1})$$

Performance usually improves with an intermediate $n$ (e.g., $n=4$ or $n=8$), balancing bias and variance.

## 10.3 Average Reward: A New Problem Setting
In the continuing setting (non-episodic) with function approximation, discounting loses its theoretical grounding. Instead, we use the **average reward** setting.

> [!definition] Average Reward $r(\pi)$
> The quality of a policy $\pi$ is defined as the average reward per time step:
> $$r(\pi) = \lim_{h \to \infty} \frac{1}{h} \sum_{t=1}^h \mathbb{E}[R_t | S_0, \pi]$$
> Under steady-state distribution $\mu_\pi(s)$:
> $$r(\pi) = \sum_s \mu_\pi(s) \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) r$$

In this setting, values are defined as **differential values** relative to the average reward:
$$q_\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=t+1}^\infty (R_k - r(\pi)) \middle| S_t=s, A_t=a \right]$$

> [!formula] Differential TD Error
> $$\delta_t = R_{t+1} - \bar{R}_t + q(S_{t+1}, A_{t+1}, \mathbf{w}_t) - q(S_t, A_t, \mathbf{w}_t)$$
> where $\bar{R}_t$ is an estimate of the average reward $r(\pi)$.

## 10.4 Deprecating the Discounted Setting
Sutton & Barto argue that discounting is "futile" in the continuing setting with function approximation. 
- If we optimize the discounted value over the on-policy distribution, the resulting policy ordering is identical to the average-reward objective, regardless of $\gamma$.
- The Policy Improvement Theorem no longer holds strictly with function approximation.

## 10.5 Differential Semi-gradient n-step Sarsa
To generalize to n-step bootstrapping in the average reward setting, we use the differential n-step return.

> [!formula] Differential n-step Return
> $$G_{t:t+n} = R_{t+1} - \bar{R}_{t+n-1} + R_{t+2} - \bar{R}_{t+n-1} + \dots + R_{t+n} - \bar{R}_{t+n-1} + q(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1})$$

The update for $\mathbf{w}$ remains the same, but we also update the average reward estimate $\bar{R}$:
$$\bar{R}_{t+1} = \bar{R}_t + \beta \delta_t$$

## Summary
- **Episodic Control**: Straightforward extension of semi-gradient methods to $q(s, a, \mathbf{w})$.
- **Mountain Car**: Demonstrates the effectiveness of [[SARSA]] with [[Tile Coding]] in continuous control.
- **Average Reward**: Necessary for continuing tasks with function approximation; replaces discounting with differential value functions.
- **[[TD Error]]**: Updated to include the average reward term $R_{t+1} - \bar{R}_t$.
