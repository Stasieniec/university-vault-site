---
type: book-chapter
course: RL
book: "Reinforcement Learning: An Introduction (2nd ed.)"
chapter: 5
sections: ["5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8*", "5.9*", "5.10"]
topics:
  - "[[Monte Carlo Methods]]"
  - "[[First-Visit MC]]"
  - "[[Every-Visit MC]]"
  - "[[Monte Carlo Control]]"
  - "[[Exploring Starts]]"
  - "[[Epsilon-Greedy Policy]]"
  - "[[Importance Sampling]]"
  - "[[Off-Policy Learning]]"
status: complete
---

# Chapter 5: Monte Carlo Methods

## Overview
[[Monte Carlo Methods]] are ways of solving the reinforcement learning problem based on **averaging sample returns**. Unlike Dynamic Programming (DP), MC methods do not assume complete knowledge of the environment's dynamics (the $p(s', r | s, a)$ function). They require only *experience*—sample sequences of states, actions, and rewards from actual or simulated interaction.

> [!abstract] Key Characteristics
> - **Experience-based**: Learns from sample trajectories.
> - **No Model Required**: Only requires samples, not transition probabilities.
> - **Episodic Tasks Only**: Returns must be well-defined at the end of an episode.
> - **No [[Bootstrapping]]**: Estimates for one state do not depend on estimates for other states (unlike DP).

---

## 5.1 Monte Carlo Prediction
MC Prediction aims to estimate the [[Value Function]] $v_\pi(s)$ for a fixed policy $\pi$.

### First-Visit vs. Every-Visit MC
- **[[First-Visit MC]]**: Averages the returns following only the *first* time a state $s$ is visited in an episode.
- **[[Every-Visit MC]]**: Averages the returns following *all* visits to state $s$ in an episode.

Both converge to $v_\pi(s)$ as the number of visits approaches infinity.

### Algorithm: First-visit MC Prediction
```python
Initialize:
    pi = policy to be evaluated
    V(s) = arbitrary real numbers, for all s in S
    Returns(s) = empty list, for all s in S

Loop forever (for each episode):
    Generate an episode following pi: S0, A0, R1, ..., ST-1, AT-1, RT
    G = 0
    Loop for each step of episode, t = T-1, T-2, ..., 0:
        G = gamma * G + Rt+1
        Unless St appears in S0, S1, ..., St-1:
            Append G to Returns(St)
            V(St) = average(Returns(St))
```

> [!example] Blackjack
> In Blackjack, the state includes the player's sum (12-21), the dealer's showing card (Ace-10), and whether the player has a "usable ace". 
> - **Goal**: Reach sum $\le 21$ and higher than the dealer.
> - **Rewards**: +1 for win, -1 for loss, 0 for draw.
> - **MC Advantage**: Obtaining the exact probabilities $p(s', r | s, a)$ for dealer card transitions is complex, but simulating games is easy.

### Backup Diagrams
For MC estimation of $v_\pi$:
- **Root**: A state node.
- **Path**: A single entire trajectory ending at a terminal state.
- **Contrast**: DP diagrams show all possible transitions for one step; MC diagrams show one sampled transition for all steps to the end.

---

## 5.2 Monte Carlo Estimation of Action Values
Without a model, estimating $v_\pi(s)$ is insufficient for control. We must estimate [[Value Function|Action-Value Functions]] $q_\pi(s, a)$.

### The Exploration Problem
If $\pi$ is deterministic, many state-action pairs $(s, a)$ may never be visited. To evaluate all actions, we need to maintain exploration.

> [!important] [[Exploring Starts]]
> To ensure all state-action pairs are visited, we specify that episodes begin at a $(s, a)$ pair, and every pair has a non-zero probability of being selected as the start.

---

## 5.3 Monte Carlo Control
Control follows the [[Generalized Policy Iteration]] (GPI) pattern:
1. **Policy Evaluation**: Estimate $q_\pi$.
2. **Policy Improvement**: Make $\pi$ greedy with respect to $q$:
   $$\pi(s) = \arg\max_a q(s, a)$$

### Algorithm: Monte Carlo ES (Exploring Starts)
```python
Initialize:
    pi(s) in A(s) (arbitrarily)
    Q(s, a) in R (arbitrarily)
    Returns(s, a) = empty list

Loop forever (for each episode):
    Choose S0 in S, A0 in A(S0) randomly (Exploring Starts)
    Generate an episode from S0, A0, following pi
    G = 0
    Loop for each step of episode, t = T-1, T-2, ..., 0:
        G = gamma * G + Rt+1
        Unless St, At appears in S0, A0, ..., St-1, At-1:
            Append G to Returns(St, At)
            Q(St, At) = average(Returns(St, At))
            pi(St) = argmax_a Q(St, a)
```

---

## 5.4 On-Policy MC Control
To avoid the unrealistic "Exploring Starts" assumption, we use **on-policy** methods that use [[Epsilon-Greedy Policy]] to ensure continuous exploration.

> [!formula] Epsilon-Greedy Policy
> For a state $s$, selected action $a$:
> $$\pi(a|s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|A(s)|} & \text{if } a = \arg\max_a Q(s, a) \\ \frac{\epsilon}{|A(s)|} & \text{if } a \neq \arg\max_a Q(s, a) \end{cases}$$

### Algorithm: On-policy first-visit MC control
```python
Initialize:
    pi = an arbitrary epsilon-soft policy
    Q(s, a) = arbitrary
    Returns(s, a) = empty list

Repeat forever:
    Generate an episode following pi
    G = 0
    Loop for each step t = T-1, ..., 0:
        G = gamma * G + Rt+1
        Unless St, At appears in earlier steps:
            Append G to Returns(St, At)
            Q(St, At) = average(Returns(St, At))
            A_star = argmax_a Q(St, a)
            For all a in A(St):
                pi(a|St) = (1 - eps + eps/|A|) if a == A_star else (eps/|A|)
```

---

## 5.5 Off-Policy Prediction via Importance Sampling
[[Off-Policy Learning]] evaluates a **target policy** $\pi$ while following a **behavior policy** $b$.

### Assumption of Coverage
Every action taken under $\pi$ must be taken at least occasionally under $b$.
$$\pi(a|s) > 0 \implies b(a|s) > 0$$

### [[Importance Sampling]] Ratio
The relative probability of a trajectory occurring under $\pi$ vs. $b$:
$$\rho_{t:T-1} = \frac{\prod_{k=t}^{T-1} \pi(A_k|S_k) p(S_{k+1}|S_k, A_k)}{\prod_{k=t}^{T-1} b(A_k|S_k) p(S_{k+1}|S_k, A_k)} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

### Two Types of IS
Let $\mathcal{T}(s)$ be the set of time steps in which state $s$ was visited.
1. **Ordinary Importance Sampling**:
   $$V(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{|\mathcal{T}(s)|}$$
   - *Unbiased*, but can have *infinite variance*.
2. **Weighted Importance Sampling**:
   $$V(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}}$$
   - *Biased* (asymptotically zero), but *much lower variance*. Strongly preferred in practice.

---

## 5.6 Incremental Implementation
To avoid storing all returns, we update estimates incrementally. For **Weighted IS**, we maintain a cumulative weight sum $C_n$.

> [!formula] Incremental Update
> $$V_{n+1} = V_n + \frac{W_n}{C_n} [G_n - V_n]$$
> $$C_{n+1} = C_n + W_{n+1}$$

### Algorithm: Off-policy MC Prediction
```python
Initialize: Q(s, a) arbitrarily, C(s, a) = 0
Repeat forever:
    b = any policy with coverage of pi
    Generate episode following b
    G = 0, W = 1
    Loop for each step t = T-1, ..., 0, while W != 0:
        G = gamma * G + Rt+1
        C(St, At) = C(St, At) + W
        Q(St, At) = Q(St, At) + (W / C(St, At)) * [G - Q(St, At)]
        W = W * (pi(At|St) / b(At|St))
```

---

## 5.7 Off-Policy Monte Carlo Control
Separates the behavior policy (exploratory) from the target policy (learning about optimality).

### Algorithm: Off-policy MC Control
```python
Initialize: Q(s, a) arbitrarily, C(s, a) = 0, pi(s) = argmax Q(s, a)
Repeat forever:
    b = any epsilon-soft policy
    Generate episode following b
    G = 0, W = 1
    Loop for each step t = T-1, ..., 0:
        G = gamma * G + Rt+1
        C(St, At) = C(St, At) + W
        Q(St, At) = Q(St, At) + (W / C(St, At)) * [G - Q(St, At)]
        pi(St) = argmax_a Q(St, a)
        If At != pi(St): break Loop
        W = W * (1 / b(At|St))
```

> [!warning] Efficiency Issue
> Off-policy MC only learns from the **tails** of episodes (once behavior matches the greedy policy). This can significantly slow down learning in long episodes.

---

## 5.8 & 5.9 Improving Importance Sampling*
- **Discounting-aware IS**: Uses the structure of discounted returns to associate importance sampling ratios only with relevant rewards, reducing variance when $\gamma < 1$.
- **Per-decision IS**: Even for $\gamma = 1$, helps by observing that $E[\rho_{t:T-1} R_{t+k}] = E[\rho_{t:t+k-1} R_{t+k}]$, removing noise from future actions that don't affect immediate rewards.

---

## Summary: MC vs. DP
| Feature | [[Dynamic Programming]] (DP) | [[Monte Carlo Methods]] (MC) |
| :--- | :--- | :--- |
| **Model** | Requires $p(s', r \| s, a)$ | Requires only samples |
| **[[Bootstrapping]]** | Yes (estimates from estimates) | No (completes episodes) |
| **Independence** | Estimates are interdependent | Estimates for one state are independent |
| **Applicability** | Full state sweeps | Can focus on subsets of states |
| **Markov Requirement** | Strongly sensitive | Less harmed by Markov violations |
