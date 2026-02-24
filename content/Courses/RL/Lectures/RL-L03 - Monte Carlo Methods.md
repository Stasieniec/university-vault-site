---
type: lecture
course: RL
week: 2
lecture: 3
book_sections: ["Ch 5.1-5.7"]
topics:
  - "[[Monte Carlo Methods]]"
  - "[[First-Visit MC]]"
  - "[[Every-Visit MC]]"
  - "[[Monte Carlo Control]]"
  - "[[Exploring Starts]]"
  - "[[Epsilon-Greedy Policy]]"
  - "[[On-Policy vs Off-Policy]]"
  - "[[Importance Sampling]]"
  - "[[Generalized Policy Iteration]]"
status: complete
---

# RL-L03: Monte Carlo Methods

## Overview
Monte Carlo (MC) methods learn value functions and optimal policies from **experience** in the form of sample episodes. 

### Key Characteristics
- **Model-free**: Unlike [[Dynamic Programming]], MC does not require knowledge of MDP dynamics ($p(s', r | s, a)$).
- **Averages Returns**: Estimates are based on averaging sample returns for each state-action pair.
- **Episodic Tasks**: Defined only for episodic tasks as it requires the completion of an episode to calculate the return $G_t$.
- **No Bootstrapping**: MC methods do not update estimates based on other estimates; they use actual sampled returns.

### MC vs. DP Comparison
| Feature | [[Dynamic Programming]] | [[Monte Carlo Methods]] |
|:--- |:--- |:--- |
| **Model** | Needs full $p(s', r \| s, a)$ | Model-free (Sample experience) |
| **Bootstrapping** | Yes (Updates based on next state values) | No (Updates based on returns) |
| **Width** | Full (Expectation over all transitions) | Single (Sample trajectory) |
| **Depth** | 1-step lookahead | Full (Until end of episode) |

---

## 1. Monte Carlo Prediction
The goal of **prediction** is to estimate the state-value function $v_\pi(s)$ under a fixed policy $\pi$.

### The Return
For a trajectory $S_t, A_t, R_{t+1}, S_{t+1}, \dots, S_T$, the return is:
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{T-t-1} R_T$$
By the law of large numbers, the average return converges to the expected value:
$$v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

### First-Visit vs. Every-Visit MC
- **First-Visit MC**: Averages returns only from the *first* time a state $s$ is visited in each episode.
  - Each return is an i.i.d. estimate of $v_\pi(s)$.
  - Convergence is $O(1/\sqrt{n})$.
- **Every-Visit MC**: Averages returns from *all* visits to $s$ in each episode.
  - Estimates are not independent but also converge to $v_\pi(s)$ quadratically.

### First-Visit MC Prediction Algorithm
> [!NOTE] Algorithm: First-visit MC prediction
> **Input**: a policy $\pi$ to be evaluated
> **Initialize**:
> $V(s) \in \mathbb{R}$ arbitrarily, $Returns(s) \leftarrow$ empty list
> **Loop forever** (for each episode):
> 1. Generate an episode following $\pi$: $S_0, A_0, R_1, \dots, S_T$
> 2. $G \leftarrow 0$
> 3. **Loop backwards** $t = T-1, T-2, \dots, 0$:
>    - $G \leftarrow \gamma G + R_{t+1}$
>    - **Unless** $S_t$ appears in $S_0, S_1, \dots, S_{t-1}$:
>      - Append $G$ to $Returns(S_t)$
>      - $V(S_t) \leftarrow \text{average}(Returns(S_t))$

---

## 2. Blackjack Example
Blackjack is a classic episodic MDP used to illustrate MC prediction.

- **Objective**: Maximize card sum $\le 21$.
- **State Space**: 
  - Current sum (12-21)
  - Dealer's showing card (Ace-10)
  - Usable Ace (Yes/No)
  - Total: 200 states.
- **Rewards**: +1 for win, -1 for loss, 0 for draw.
- **Action**: Hit or Stick.
- **Policy Evaluation**: Average returns over thousands of simulated games (episodes).
- **Observation**: States with usable aces are less frequent and thus have higher variance in the value function estimate.

---

## 3. Monte Carlo Control
Control aims to approximate optimal policies using [[Generalized Policy Iteration]] (GPI).

### Action Values ($Q$)
Without a model, state values $V(s)$ are insufficient for control (cannot look ahead). We must estimate action-value functions $Q(s, a)$.
$$q_\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

### The Exploration-Exploitation Dilemma
Many state-action pairs might never be visited if $\pi$ is deterministic. Two solutions:
1. **[[Exploring Starts]]**: Assume every episode starts at a random state-action pair with non-zero probability.
2. **[[On-Policy vs Off-Policy|On-Policy]]**: Use $\epsilon$-greedy policies.
3. **[[On-Policy vs Off-Policy|Off-Policy]]**: Use a separate behavior policy to explore.

### Algorithm: Monte Carlo ES (Exploring Starts)
This algorithm alternates between evaluation and improvement episode-by-episode.

> [!IMPORTANT] Algorithm: Monte Carlo ES
> **Initialize**: $\pi(s)$ arbitrarily, $Q(s, a)$ arbitrarily, $Returns(s, a)$ empty
> **Loop forever**:
> 1. Choose $S_0, A_0$ such that all pairs have probability $>0$ (Exploring Starts)
> 2. Generate episode from $S_0, A_0$ following $\pi$: $S_0, A_0, R_1, \dots, S_T$
> 3. $G \leftarrow 0$
> 4. **Loop backwards** $t = T-1, \dots, 0$:
>    - $G \leftarrow \gamma G + R_{t+1}$
>    - **Unless** $(S_t, A_t)$ appeared earlier in the episode:
>      - Append $G$ to $Returns(S_t, A_t)$
>      - $Q(S_t, A_t) \leftarrow \text{average}(Returns(S_t, A_t))$
>      - $\pi(S_t) \leftarrow \text{argmax}_a Q(S_t, a)$

---

## 4. On-Policy MC Control ($\epsilon$-greedy)
Avoids exploring starts by using a soft policy (e.g., $\epsilon$-greedy).

### $\epsilon$-greedy Improvement
For an $\epsilon$-soft policy $\pi$, an $\epsilon$-greedy policy $\pi'$ wrt $q_\pi$ is an improvement ($\forall s, v_{\pi'}(s) \ge v_\pi(s)$).
$$
\pi(a|s) = \begin{cases} 
1 - \epsilon + \frac{\epsilon}{|\mathcal{A}(s)|} & \text{if } a = \text{argmax} Q(s, a) \\
\frac{\epsilon}{|\mathcal{A}(s)|} & \text{otherwise}
\end{cases}
$$

**Proof Idea (PIT):**
$$q_\pi(s, \pi'(s)) = \sum_a \pi'(a|s) q_\pi(s, a) = \frac{\epsilon}{|\mathcal{A}|} \sum_a q_\pi(s, a) + (1-\epsilon) \max_a q_\pi(s, a) \ge v_\pi(s)$$

---

## 5. Off-Policy Prediction and Control
Learn about a **target policy** $\pi$ while following a **behavior policy** $b$ ($b \neq \pi$).

### Coverage Assumption
The behavior policy $b$ must be able to take any action that $\pi$ might take:
$$\pi(a|s) > 0 \implies b(a|s) > 0$$

### Importance Sampling (IS) Ratio
To transform expectations from $b$ to $\pi$, we weight returns by the probability of the trajectory occurring under $\pi$ vs. $b$:
$$\rho_{t:T-1} = \frac{\prod_{k=t}^{T-1} \pi(A_k | S_k) p(S_{k+1} | S_k, A_k)}{\prod_{k=t}^{T-1} b(A_k | S_k) p(S_{k+1} | S_k, A_k)} = \prod_{k=t}^{T-1} \frac{\pi(A_k | S_k)}{b(A_k | S_k)}$$
*Note: Transition dynamics $p$ cancel out!*

### Types of Importance Sampling
1. **Ordinary IS**: Simple average of scaled returns.
   - Unbiased, but can have **infinite variance**.
   $$V(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T-1} G_t}{|\mathcal{T}(s)|}$$
2. **Weighted IS**: Weighted average of scaled returns.
   - Biased (bias $\to 0$ as $n \to \infty$), but **finite variance**.
   $$V(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T-1} G_t}{\sum_{t \in \mathcal{T}(s)} \rho_{t:T-1}}$$

### Algorithm: Off-policy MC Control
> [!IMPORTANT] Algorithm: Off-policy MC Control
> **Initialize**: $Q(s, a)$ arbitrarily, $C(s, a) \leftarrow 0$, $\pi(s) \leftarrow \text{argmax}_a Q(s, a)$
> **Loop forever**:
> 1. Select soft behavior policy $b$; Generate episode following $b$: $S_0, A_0, \dots, S_T$
> 2. $G \leftarrow 0, W \leftarrow 1$
> 3. **Loop backwards** $t = T-1, \dots, 0$:
>    - $G \leftarrow \gamma G + R_{t+1}$
>    - $C(S_t, A_t) \leftarrow C(S_t, A_t) + W$
>    - $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{W}{C(S_t, A_t)} [G - Q(S_t, A_t)]$
>    - $\pi(S_t) \leftarrow \text{argmax}_a Q(S_t, a)$
>    - **If** $A_t \neq \pi(S_t)$ **then exit inner loop**
>    - $W \leftarrow W \frac{1}{b(A_t | S_t)}$

---

## 6. Incremental Implementation
Weighted IS can be implemented incrementally to avoid storing all returns.
Given $G_1, \dots, G_n$ with weights $W_1, \dots, W_n$:
1. $C_n = C_{n-1} + W_n$
2. $V_{n+1} = V_n + \frac{W_n}{C_n} [G_n - V_n]$

---

## 7. Diagrams

### Backup Diagram: MC Prediction
```
      (S_t)       <-- Root (state to update)
        |
     [A_t, R_{t+1}]
        |
      (S_{t+1})
        |
     [A_{t+1}, R_{t+2}]
        |
       ...
        |
      ((T))       <-- Terminal (end of episode)
```
*Contrast with DP: MC looks at a single, full trajectory.*

---

## Summary Key Points
- **MC** learns from experience, avoiding the need for environment models.
- **Goal**: Average returns to estimate expectations.
- **GPI** applies: use evaluation (averaging) and improvement (greedy/$\epsilon$-greedy).
- **Off-policy** requires [[Importance Sampling]] to account for different behavior.
- **Variance** is the main challenge in MC, especially in Off-policy IS.
