---
type: lecture
course: RL
week: 2
lecture: 4
book_sections: ["Ch 6.1-6.5"]
topics:
  - "[[Temporal Difference Learning]]"
  - "[[TD(0)]]"
  - "[[TD Error]]"
  - "[[SARSA]]"
  - "[[Q-Learning]]"
  - "[[Expected SARSA]]"
  - "[[On-Policy vs Off-Policy]]"
  - "[[Bootstrapping]]"
status: complete
---

# RL Lecture 4: Temporal Difference Learning

Temporal-Difference (TD) learning is a central and novel idea to reinforcement learning. It is a combination of **Monte Carlo (MC)** ideas and **Dynamic Programming (DP)** ideas.

- **Like Monte Carlo:** TD methods can learn directly from raw experience without a model of the environment's dynamics (model-free).
- **Like DP:** TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they **bootstrap**).

---

## 1. TD Prediction (TD(0))

The goal is to solve the prediction problem: estimating the value function $v_\pi$ for a given policy $\pi$.

### TD(0) Update Rule
The simplest TD method, **TD(0)** or one-step TD, makes the following update after transitioning from $S_t$ to $S_{t+1}$ and receiving reward $R_{t+1}$:

$$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

- **Target:** $R_{t+1} + \gamma V(S_{t+1})$ (the TD target)
- **Error:** $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ (the **TD Error**)

### Comparison: MC vs DP vs TD
| Method | Target | Equation | Focus |
| :--- | :--- | :--- | :--- |
| **Monte Carlo** | $G_t$ | $V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$ | Uses actual return (full sample) |
| **Dynamic Programming** | $E_\pi[R_{t+1} + \gamma V(S_{t+1})]$ | $V(S_t) \leftarrow \sum_a \pi(a\|s) \sum_{s', r} p(s', r\|s, a)[r + \gamma V(s')]$ | Uses full model (expectations) |
| **TD(0)** | $R_{t+1} + \gamma V(S_{t+1})$ | $V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$ | Uses sample transitions and bootstraps |

> [!NOTE] Bootstrapping
> TD methods are **bootstrapping** because they base their update on an existing estimate ($V(S_{t+1})$), similar to DP. However, they are **sampling** because they use a single sample transition ($R_{t+1}$), similar to MC.

### Backup Diagrams
- **TD(0):** A single state node $S_t$ leading to a successor state $S_{t+1}$ via a reward $R_{t+1}$. The update looks only one step ahead.
- **Monte Carlo:** A full path from the current state $S_t$ to the terminal state.
- **Dynamic Programming:** A complete tree of all possible transitions from $S_t$ to all possible $S_{t+1}$.

---

## 2. TD Error ($\delta$)

The TD error is the discrepancy between the current estimate $V(S_t)$ and the "better" estimate $R_{t+1} + \gamma V(S_{t+1})$ available one step later.

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

**Significance:**
- It is the error in the estimate made at time $t$, but only available at $t+1$.
- If $V$ does not change during an episode, the total MC error can be written as a sum of TD errors:
  $$G_t - V(S_t) = \sum_{k=t}^{T-1} \gamma^{k-t} \delta_k$$

---

## 3. Advantages of TD Learning

1.  **Learns Online:** Updates are made at every step, whereas MC must wait until the end of an episode.
2.  **Continuous Tasks:** TD works naturally on continuing tasks without episodes, where MC cannot be applied.
3.  **Efficiency:** TD methods usually converge faster than constant-$\alpha$ MC on stochastic tasks (e.g., the **Random Walk** example).
4.  **No Model Required:** Like MC, it does not need $p(s', r | s, a)$.

---

## 4. Batch Updating and Optimality

When experience is limited, we can use **batch updating**—presenting the same finite sequence of experience repeatedly.

- **MC Optimality:** Under batch updating, MC converges to values that minimize the mean square error on the training set.
- **TD(0) Optimality:** Under batch updating, TD(0) converges to the **certainty-equivalence estimate**—the value function that would be correct if the maximum-likelihood model of the Markov process were exactly correct.

> [!TIP] Intuition
> TD takes advantage of the Markov property. It builds a consistent model of the transitions even if it hasn't seen a particular terminal return yet, often leading to better generalizations than MC.

---

## 5. SARSA: On-policy TD Control

SARSA estimates the action-value function $Q(s, a)$ for the current policy.

### Update Rule
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$
The name comes from the quintuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$.

### Pseudocode
```python
Initialize Q(s, a) arbitrarily, Q(terminal, :) = 0
Loop for each episode:
    Initialize S
    Choose A from S using policy derived from Q (e.g., epsilon-greedy)
    Loop for each step of episode:
        Take action A, observe R, S'
        Choose A' from S' using policy derived from Q (e.g., epsilon-greedy)
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        S <- S'; A <- A'
    until S is terminal
```

**Backup Diagram:** From $(S_t, A_t)$, look ahead to $(S_{t+1}, A_{t+1})$ based on the action actually taken by the current policy.

---

## 6. Q-Learning: Off-policy TD Control

Q-learning directly approximates $q^*$, independent of the policy being followed.

### Update Rule
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

### Pseudocode
```python
Initialize Q(s, a) arbitrarily, Q(terminal, :) = 0
Loop for each episode:
    Initialize S
    Loop for each step of episode:
        Choose A from S using policy derived from Q (e.g., epsilon-greedy)
        Take action A, observe R, S'
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]
        S <- S'
    until S is terminal
```

**Backup Diagram:** From $(S_t, A_t)$, look ahead to all possible actions $a$ in $S_{t+1}$ and take the maximum.

---

## 7. Expected SARSA

Expected SARSA uses the expected value of the next state-action pair under the policy, rather than a single sample.

### Update Rule
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)]$$

- **Pros:** Reduces variance due to action selection at $S_{t+1}$. It can set $\alpha=1$ in deterministic environments like Cliff Walking.
- **Relation:** If the target policy is greedy, Expected SARSA is identical to Q-learning.

---

## 8. Cliff Walking Example: SARSA vs Q-Learning

A classic gridworld example where the agent must navigate from Start to Goal.
- **Environment:** A bottom row labeled "The Cliff". Falling in gives $R=-100$ and resets to start.
- **Behavior:**
    - **Q-Learning** (Off-policy) learns the **optimal path** (hugging the cliff), but occasionally falls off during exploration ($\epsilon$-greedy).
    - **SARSA** (On-policy) learns the **safer path** (roundabout) because it accounts for its own exploration and knows it might fall if it gets too close.

---

## 9. Maximization Bias and Double Q-Learning

### Maximization Bias
Algorithms involving a `max` operator (like Q-learning) can develop a positive bias because the maximum of noisy estimates is often greater than the maximum of true values.

### Double Q-Learning
Maintains two independent estimates, $Q_1$ and $Q_2$. One is used to select the maximizing action, and the other is used to estimate its value.

**Update (if $Q_1$ is updated):**
$$Q_1(S, A) \leftarrow Q_1(S, A) + \alpha [R + \gamma Q_2(S', \text{argmax}_a Q_1(S', a)) - Q_1(S, A)]$$

---

## Figure Summaries (Lecture Slides)

### Backup Diagram Comparison
- **MC:** Long trace of state-action-reward pairs until termination. No branching.
- **DP:** Full branch tree from $S_t$ to all possible actions, then to all possible successor states.
- **TD(0):** Short trace from $S_t$ to $S_{t+1}$.
- **SARSA:** Short trace from $(S_t, A_t)$ to $(S_{t+1}, A_{t+1})$.
- **Q-Learning:** Short trace from $(S_t, A_t)$ to $S_{t+1}$, then branching to all actions with an arc signifying the `max`.

### Performance Examples
- **Random Walk:** TD(0) converges much faster than MC, reaching optimal values around 100 episodes compared to MC's slower progression.
- **Windy Gridworld:** Shows the agent learning to reach the goal faster over time (Steps per episode decreasing). The windy grid shifts the next state upward depending on the column.
- **Cliff Walking:** Visualizes Reward per Episode. SARSA has higher average reward because it avoids the cliff, while Q-learning has lower average reward due to falling off during exploration, despite finding the shorter path.

---

## Summary of TD Control
| Algorithm | Type | Target |
| :--- | :--- | :--- |
| **SARSA** | On-policy | $R + \gamma Q(S', A')$ |
| **Q-Learning** | Off-policy | $R + \gamma \max_a Q(S', a)$ |
| **Expected SARSA**| On/Off-policy | $R + \gamma \sum_a \pi(a\|S') Q(S', a)$ |

---
**References:**
- Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 6.1-6.7.
- Lecture Slides RL L04.
