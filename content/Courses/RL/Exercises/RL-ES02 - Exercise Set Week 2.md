---
type: exercise
course: RL
week: 2
source: "Exercise Set 2"
concepts:
  - "[[Monte Carlo Methods]]"
  - "[[First-Visit MC]]"
  - "[[Importance Sampling]]"
  - "[[Monte Carlo Control]]"
  - "[[Temporal Difference Learning]]"
  - "[[SARSA]]"
  - "[[Q-Learning]]"
  - "[[TD Error]]"
status: complete
---

# RL Exercise Set Week 2: Monte Carlo & TD Learning

## 3 Monte Carlo methods

### 3.1 Monte Carlo

> [!question] Exercise 1: MC Estimation
> Consider an MDP with a single state $s_0$ that has a certain probability of transitioning back onto itself with a reward of 0, and will otherwise terminate with a reward of 3. Your agent has interacted with the environment and has gotten the following two sequences of rewards obtained: $[0, 0, 3]$, $[0, 0, 0, 3]$. Use $\gamma = 0.8$.
> 
> (a) Estimate the value of $s_0$ using **first-visit MC**.
> (b) Estimate the value of $s_0$ using **every-visit MC**.

**Concepts Tested:** `[[First-Visit MC]]`, `[[Monte Carlo Methods]]`

**Solution:**

First, let's calculate the returns ($G_t$) for each visit to $s_0$.
The return $G_t$ is defined as $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$.

*   **Sequence 1:** $[0, 0, 3]$ (Transitions: $s_0 \xrightarrow{0} s_0 \xrightarrow{0} s_0 \xrightarrow{3} Terminal$)
    *   Visit 1 (at $t=0$): $G_0 = 0 + 0.8 \cdot 0 + 0.8^2 \cdot 3 = 1.92$
    *   Visit 2 (at $t=1$): $G_1 = 0 + 0.8 \cdot 3 = 2.4$
    *   Visit 3 (at $t=2$): $G_2 = 3.0$
*   **Sequence 2:** $[0, 0, 0, 3]$ (Transitions: $s_0 \xrightarrow{0} s_0 \xrightarrow{0} s_0 \xrightarrow{0} s_0 \xrightarrow{3} Terminal$)
    *   Visit 1: $G_0 = 0 + 0.8 \cdot 0 + 0.8^2 \cdot 0 + 0.8^3 \cdot 3 = 1.536 \approx 1.54$
    *   Visit 2: $G_1 = 0 + 0.8 \cdot 0 + 0.8^2 \cdot 3 = 1.92$
    *   Visit 3: $G_2 = 0 + 0.8 \cdot 3 = 2.4$
    *   Visit 4: $G_3 = 3.0$

**(a) First-visit MC:**
We only take the return from the *first time* $s_0$ is visited in each episode.
*   Episode 1: $G = 1.92$
*   Episode 2: $G = 1.54$
$$V(s_0) \approx \frac{1.92 + 1.54}{2} = 1.73$$

**(b) Every-visit MC:**
We average the returns from *every* visit to $s_0$ across both episodes.
*   Total visits = $3 + 4 = 7$
*   Sum of returns = $(1.92 + 2.4 + 3.0) + (1.54 + 1.92 + 2.4 + 3.0) = 16.18$
$$V(s_0) \approx \frac{16.18}{7} \approx 2.31$$

---

### 3.2 Bias of $v_\pi$ Monte Carlo estimators

> [!question] Exercise 1: Importance Sampling
> Comment on the bias of **weighted importance sampling** compared to **ordinary importance sampling**. Why might we nevertheless use weighted importance sampling?

**Concepts Tested:** `[[Importance Sampling]]`

**Solution:**
Ordinary importance sampling is **unbiased**, while weighted importance sampling is **biased** (though the bias converges to zero as the number of samples increases). However, weighted importance sampling is preferred because it significantly **reduces variance**. In ordinary importance sampling, the variance can be unbounded if the importance ratios are large (e.g., a rare action in the behavior policy is common in the target policy).

> [!question] Exercise 2: Unbiasedness of Single Episode MC
> Consider one episode following $\pi$: $(S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T)$, where $S_0 = s$. Determine and provide intuition on the biasedness of the following estimator for $v_\pi(s)$:
> $$\sum_{i=1}^T \gamma^{i-1} R_i$$

**Concepts Tested:** `[[Monte Carlo Methods]]`

**Solution:**
This estimator is **unbiased**.
By definition: $v_\pi(s) = \mathbb{E}_\pi [G_0 \mid S_0 = s]$.
The sum $\sum_{i=1}^T \gamma^{i-1} R_i$ is exactly the return $G_0$ for an episode starting in state $s$ at $t=0$. Since the episode follows policy $\pi$, its expectation is exactly the value function $v_\pi(s)$.

> [!question] Exercise 3: Every-visit MC Bias
> Determine the biasedness of:
> $$\frac{1}{|J|} \sum_{j \in J} \sum_{i=1}^{T-j} \gamma^{i-1} R_{j+i}$$
> where $J$ contains all indices $j$ such that $S_j = s$.

**Concepts Tested:** `[[Monte Carlo Methods]]`

**Solution:**
This is the **every-visit MC estimator**. It is **biased**.
**Intuition:** For any visit after the first, the corresponding return $G_j$ is a sample from the distribution of returns *conditioned* on the fact that the state $s$ has already been visited earlier in the trajectory. This conditioning restricts the sample space and induces a bias relative to the true value function $v_\pi(s)$, which is the unconditional expectation of returns from state $s$.

> [!question] Exercise 4: Latest-visit MC Bias
> Determine the biasedness of:
> $$\sum_{i=1}^{T-t_s} \gamma^{i-1} R_{t_s+i}$$
> where $t_s$ is the **latest** time step such that $S_{t_s} = s$. How does this compare to the first-visit MC estimator?

**Solution:**
This estimator is **biased** for the same reasons as every-visit MC. If $t_s$ is not the first visit, it is a conditional sample. It only becomes unbiased in the specific case where $s$ is visited exactly once in the episode (making it identical to a first-visit sample).

---

### 3.3 * Exam question: Monte Carlo for control

> [!tip] Exam-Style Question
> The following questions refer to the pseudo-code for Off-policy MC Control (Figure 1).
> 
> 1. Part of the algorithm is covered by a black square (the inner loop range). What is the missing information?
> 2. What is stored in $C(S_t, A_t)$?
> 3. Why is the inner loop stopped when $A_t \neq \pi(S_t)$?

**Concepts Tested:** `[[Monte Carlo Control]]`, `[[Importance Sampling]]`

![[rl_p4.png]]

> [!example] ASCII Reproduction of Figure 1: Off-policy MC Control
> ```pseudo
> Initialize, for all s ∈ S, a ∈ A(s):
>   Q(s,a) ∈ R (arbitrarily)
>   C(s,a) ← 0
>   π(s) ← argmax_a Q(s,a)    (with ties broken consistently)
> 
> Loop forever (for each episode):
>   b ← any soft policy
>   Generate an episode using b: S0, A0, R1, ..., ST-1, AT-1, RT
>   G ← 0
>   W ← 1
>   Loop for each step of episode, t = T-1, T-2, ... 0:  <-- [BLACK SQUARE]
>     G ← G + Rt+1
>     C(St, At) ← C(St, At) + W
>     Q(St, At) ← Q(St, At) + (W / C(St, At)) [ G − Q(St, At) ]
>     π(St) ← argmax_a Q(St, a)    (with ties broken consistently)
>     If At ≠ π(St) then exit inner loop
>     W ← W * 1 / b(At | St)
> ```

**Solution:**

1.  The missing range is **$t = T-1, T-2, \dots, 0$** (working backwards from the end of the episode).
2.  $C(S_t, A_t)$ stores the **cumulative importance weights** of all visits to that state-action pair across all episodes. It acts as the denominator for the weighted average.
3.  The inner loop stops because we are evaluating and improving a **greedy policy** $\pi$. If an action $A_t$ taken by the behavior policy $b$ is *not* the action that the greedy target policy would have taken, the probability of that action under $\pi$ is 0. Since the importance sampling weight involves the ratio $\frac{\pi(A_t|S_t)}{b(A_t|S_t)}$, subsequent weights for the rest of the episode would become 0.

---

## 4 Temporal Difference Learning

### 4.1 Temporal Difference Learning (application)

> [!question] Exercise 1: TD, SARSA, Q-Learning Trace
> Consider an undiscounted MDP with states $A, B$ and terminal state $T$ ($V(T)=0$). 
> Observed episode: $A \xrightarrow[r=-3]{a=1} B \xrightarrow[r=4]{a=1} A \xrightarrow[r=-4]{a=2} A \xrightarrow[r=-3]{a=1} T$
> 
> Parameters: $\gamma = 1, \alpha = 0.1$, initial values = 0.
> 
> Calculate final estimates for:
> (a) **TD(0)**
> (b) **SARSA**
> (c) **Q-learning**

**Concepts Tested:** `[[Temporal Difference Learning]]`, `[[SARSA]]`, `[[Q-Learning]]`, `[[TD Error]]`

**Solution:**

**(a) TD(0):**
Update rule: $V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$

1.  $A \to B$ ($r=-3$): $V(A) = 0 + 0.1[-3 + 0 - 0] = -0.3$
2.  $B \to A$ ($r=4$): $V(B) = 0 + 0.1[4 + (-0.3) - 0] = 0.37$
3.  $A \to A$ ($r=-4$): $V(A) = -0.3 + 0.1[-4 + (-0.3) - (-0.3)] = -0.7$
4.  $A \to T$ ($r=-3$): $V(A) = -0.7 + 0.1[-3 + 0 - (-0.7)] = -0.93$

**Final:** $V(A) = -0.93, V(B) = 0.37$

**(b) SARSA:**
Update rule: $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$

1.  $(A,1) \to (B,1)$ ($r=-3$): $Q(A,1) = 0 + 0.1[-3 + 0 - 0] = -0.3$
2.  $(B,1) \to (A,2)$ ($r=4$): $Q(B,1) = 0 + 0.1[4 + 0 - 0] = 0.4$
3.  $(A,2) \to (A,1)$ ($r=-4$): $Q(A,2) = 0 + 0.1[-4 + (-0.3) - 0] = -0.43$
    *   *Note: uses $Q(A,1)=-0.3$ for the next state-action $A_{t+1}$.*
4.  $(A,1) \to T$ ($r=-3$): $Q(A,1) = -0.3 + 0.1[-3 + 0 - (-0.3)] = -0.57$

**Final:** $Q(A,1) = -0.57, Q(A,2) = -0.43, Q(B,1) = 0.4, Q(B,2) = 0$

**(c) Q-Learning:**
Update rule: $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$

1.  $(A,1) \to B$ ($r=-3$): $Q(A,1) = 0 + 0.1[-3 + \max(0,0) - 0] = -0.3$
2.  $(B,1) \to A$ ($r=4$): $Q(B,1) = 0 + 0.1[4 + \max(0,0) - 0] = 0.4$
3.  $(A,2) \to A$ ($r=-4$): $Q(A,2) = 0 + 0.1[-4 + \max(-0.3, 0) - 0] = -0.4$
4.  $(A,1) \to T$ ($r=-3$): $Q(A,1) = -0.3 + 0.1[-3 + 0 - (-0.3)] = -0.57$

**Final:** $Q(A,1) = -0.57, Q(A,2) = -0.4, Q(B,1) = 0.4, Q(B,2) = 0$

---

### 4.2 Temporal Difference Learning (Theory)

> [!question] Exercise 1: Incremental MC Update
> Show that the average return $V_M(S) = \frac{1}{M} \sum_{n=1}^M G_n(S)$ can be written in the incremental update form:
> $V_M(S) = V_{M-1}(S) + \alpha_M [G_M(S) - V_{M-1}(S)]$
> Identify the learning rate $\alpha_M$.

**Concepts Tested:** `[[Monte Carlo Methods]]`

**Solution:**
$$V_M(S) = \frac{1}{M} \sum_{n=1}^M G_n(S)$$
$$V_M(S) = \frac{1}{M} \left[ G_M(S) + \sum_{n=1}^{M-1} G_n(S) \right]$$
Since $V_{M-1}(S) = \frac{1}{M-1} \sum_{n=1}^{M-1} G_n(S)$, we have $\sum_{n=1}^{M-1} G_n(S) = (M-1)V_{M-1}(S)$.
$$V_M(S) = \frac{1}{M} \left[ G_M(S) + (M-1)V_{M-1}(S) \right]$$
$$V_M(S) = \frac{1}{M} G_M(S) + \frac{M-1}{M} V_{M-1}(S)$$
$$V_M(S) = \frac{1}{M} G_M(S) + \left( 1 - \frac{1}{M} \right) V_{M-1}(S)$$
$$V_M(S) = V_{M-1}(S) + \frac{1}{M} [G_M(S) - V_{M-1}(S)]$$
The learning rate is **$\alpha_M = \frac{1}{M}$**.

> [!question] Exercise 2: Expected TD Error
> Consider the TD-error $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$.
> 
> (a) What is $\mathbb{E}[\delta_t \mid S_t = s]$ if $\delta_t$ uses the true state-value function $v_\pi$?
> (b) What is $\mathbb{E}[\delta_t \mid S_t = s, A_t = a]$ if $\delta_t$ uses the true state-value function $v_\pi$?

**Concepts Tested:** `[[TD Error]]`

**Solution:**

**(a) Given $S_t=s$:**
$$\mathbb{E}[\delta_t \mid S_t = s] = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) - v_\pi(S_t) \mid S_t = s]$$
$$\mathbb{E}[\delta_t \mid S_t = s] = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] - v_\pi(s)$$
By the Bellman Equation, $\mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] = v_\pi(s)$.
$$\mathbb{E}[\delta_t \mid S_t = s] = v_\pi(s) - v_\pi(s) = 0$$

**(b) Given $S_t=s, A_t=a$:**
$$\mathbb{E}[\delta_t \mid S_t = s, A_t = a] = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = a] - v_\pi(s)$$
The first term is the definition of the action-value function $q_\pi(s, a)$.
$$\mathbb{E}[\delta_t \mid S_t = s, A_t = a] = q_\pi(s, a) - v_\pi(s)$$
This result is known as the **Advantage Function** $A(s, a)$.
