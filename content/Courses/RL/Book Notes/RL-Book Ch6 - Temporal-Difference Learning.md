---
type: book-chapter
course: RL
book: "Reinforcement Learning: An Introduction (2nd ed.)"
chapter: 6
sections: ["6.1", "6.2", "6.3", "6.4", "6.5", "6.6", "6.7", "6.8", "6.9"]
topics:
  - "[[Temporal Difference Learning]]"
  - "[[TD(0)]]"
  - "[[TD Error]]"
  - "[[SARSA]]"
  - "[[Q-Learning]]"
  - "[[Expected SARSA]]"
  - "[[Bootstrapping]]"
  - "[[On-Policy vs Off-Policy]]"
status: complete
---

# Chapter 6: Temporal-Difference Learning

## Overview
Temporal-Difference (TD) learning is the central and novel idea of [[Reinforcement Learning]]. It is a combination of [[Monte Carlo Methods]] and [[Dynamic Programming]] (DP) ideas:
- **Like Monte Carlo**: TD methods can learn directly from raw experience without a model of the environment's dynamics.
- **Like DP**: TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they **[[Bootstrapping|bootstrap]]**).

> [!intuition]
> TD learning updates its "guess" based on another "guess" further down the line, rather than waiting for the final reward at the end of an episode.

---

## 6.1 TD Prediction
The goal is to estimate the [[Value Function]] $v_\pi$ for a given [[Policy]] $\pi$. 

### The TD(0) Update Rule
In constant-$\alpha$ MC, the update is:
$$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$$
where $G_t$ is the actual return. In **[[TD(0)]]** (one-step TD), the update is performed immediately after transitioning to $S_{t+1}$ and receiving $R_{t+1}$:

> [!formula] **TD(0) Update**
> $$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

### TD Error
The quantity in the brackets is the **[[TD Error]]**, denoted by $\delta_t$:
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

### Pseudocode: Tabular TD(0)
```python
# Tabular TD(0) for estimating v_pi
Input: policy pi to be evaluated
Algorithm parameter: step size alpha in (0, 1]
Initialize V(s) for all s in S+, arbitrarily, V(terminal) = 0

Loop for each episode:
    Initialize S
    Loop for each step of episode:
        A = action given by pi for S
        Take action A, observe R, S'
        V(S) <- V(S) + alpha * [R + gamma * V(S') - V(S)]
        S = S'
    until S is terminal
```

### Backup Diagrams
- **TD(0)**: Updates from a single sample transition ($S_t \to R_{t+1}, S_{t+1}$).
- **MC**: Updates from the entire sequence of rewards until the end of the episode.
- **DP**: Updates based on the complete distribution of all possible successors (expected update).

---

## 6.2 Advantages of TD Prediction Methods
1. **Model-Free**: Does not require $P(s', r | s, a)$.
2. **Online/Incremental**: Updates every time step. MC must wait until the end of an episode.
3. **Convergence**: TD(0) converges to $v_\pi$ for a fixed policy $\pi$.
4. **Efficiency**: Empirically, TD methods often converge faster than MC on stochastic tasks (e.g., the **Random Walk** example).

---

## 6.3 Optimality of TD(0)
In **Batch Updating**, where experience is presented repeatedly until convergence:
- **Batch MC** minimizes mean square error on the training set (best fit to observed returns).
- **Batch TD(0)** converges to the **Certainty-Equivalence Estimate** (the value function that would be correct for the maximum-likelihood model of the MDP).

> [!example] **The Predictor's Dilemma**
> If state A always leads to B (reward 0), and B has a 75% chance of return 1, TD(0) says $V(A) = 0.75$ (correct for the underlying Markov process), while MC might say $V(A) = 0$ if the single observed path from A gave return 0.

---

## 6.4 SARSA: On-Policy TD Control
To solve the control problem, we switch to an action-value function $Q(s, a)$.

> [!formula] **SARSA Update**
> $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$
> The name comes from the quintuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$.

### Pseudocode: SARSA
```python
# Sarsa (on-policy TD control) for estimating Q ~ q*
Algorithm parameters: step size alpha in (0, 1], small epsilon > 0
Initialize Q(s, a) for all s, a, Q(terminal, .) = 0

Loop for each episode:
    Initialize S
    Choose A from S using policy derived from Q (e.g., epsilon-greedy)
    Loop for each step of episode:
        Take action A, observe R, S'
        Choose A' from S' using policy derived from Q (e.g., epsilon-greedy)
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        S = S'; A = A'
    until S is terminal
```

---

## 6.5 Q-Learning: Off-Policy TD Control
**[[Q-Learning]]** directly approximates $q^*$, the optimal action-value function, independent of the policy being followed.

> [!formula] **Q-Learning Update**
> $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)]$$

### Pseudocode: Q-Learning
```python
# Q-learning (off-policy TD control) for estimating pi ~ pi*
Algorithm parameters: step size alpha in (0, 1], small epsilon > 0
Initialize Q(s, a) for all s, a, Q(terminal, .) = 0

Loop for each episode:
    Initialize S
    Loop for each step of episode:
        Choose A from S using policy derived from Q (e.g., epsilon-greedy)
        Take action A, observe R, S'
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]
        S = S'
    until S is terminal
```

> [!example] **Cliff Walking**
> - **Q-Learning** learns the optimal path (along the edge) but suffers lower online rewards because $\epsilon$-greedy exploration causes it to fall off the cliff.
> - **SARSA** learns a "safer" roundabout path because it takes the exploration into account (it is [[On-Policy vs Off-Policy|on-policy]]).

---

## 6.6 Expected SARSA
**[[Expected SARSA]]** uses the expected value over next actions instead of the maximum or a single sample:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)]$$
It is more computationally complex but reduces variance compared to SARSA.

---

## 6.7 Maximization Bias and Double Learning
Maximization over estimated values can lead to a positive bias (**Maximization Bias**) because the maximum of estimates $\max(\hat{Q})$ is usually greater than the maximum of true values $\max(q)$.

### Double Q-Learning
To eliminate this, use two independent estimates $Q_1$ and $Q_2$:
- Use $Q_1$ to find the maximizing action: $A^* = \text{argmax}_a Q_1(S', a)$
- Use $Q_2$ to estimate the value: $Q_1(S, A) \leftarrow Q_1(S, A) + \alpha [R + \gamma Q_2(S', A^*) - Q_1(S, A)]$

---

## Summary Comparison
| Feature | TD(0) | SARSA | Q-Learning |
| :--- | :--- | :--- | :--- |
| **Type** | Prediction | Control (On-policy) | Control (Off-policy) |
| **Target** | $R + \gamma V(S')$ | $R + \gamma Q(S', A')$ | $R + \gamma \max_a Q(S', a)$ |
| **Bootstrapping** | Yes | Yes | Yes |
| **Model Req.** | No | No | No |

---
*Created for RL Course - Obsidian University Vault*
