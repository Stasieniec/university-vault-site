---
type: exercise
course: RL
week: 2
source: "Homework 2 (Coding Assignment 2)"
concepts:
  - "[[Monte Carlo Methods]]"
  - "[[Importance Sampling]]"
  - "[[First-Visit MC]]"
  - "[[Every-Visit MC]]"
  - "[[On-Policy vs Off-Policy]]"
  - "[[SARSA]]"
  - "[[Q-Learning]]"
  - "[[Epsilon-Greedy Policy]]"
status: complete
---

# RL-HW02: Homework 2 — Monte Carlo & TD

> [!tip] Exam Relevance
> Questions on IS derivations (1a), first-visit vs every-visit calculations (2a), and SARSA vs Q-learning convergence analysis (3a-3c) are classic exam material.

---

## Part 1: Importance Sampling Derivations

### Q1a: Incremental Update Rule for Ordinary IS (1.0p)

**Q:** Given $V_n = \frac{\sum_{k=1}^n W_k G_k}{n}$, derive the incremental update rule of the form $V_n = V_{n-1} + a \cdot (b - V_{n-1})$.

### Solution

Starting from:
$$V_n = \frac{\sum_{k=1}^n W_k G_k}{n} = \frac{(n-1) V_{n-1} + W_n G_n}{n}$$

> [!formula] Derivation
> $$V_n = V_{n-1} + \frac{1}{n}(W_n G_n - V_{n-1})$$
> 
> So: $a = \frac{1}{n}$ and $b = W_n G_n$.

> [!warning] Contrast with Weighted IS
> For **weighted** IS (equations 5.7-5.8 in book), the step size is $\frac{W_n}{C_n}$ where $C_n = \sum_{k=1}^n W_k$. For **ordinary** IS, the step size is simply $\frac{1}{n}$ — just a standard running average, but the target is $W_n G_n$ (the importance-weighted return).

### Q1b: Advantages of MC over DP (1.5p)

**Q:** Two advantages of MC over DP? When to use each?

### Solution

1. **Model-free**: MC doesn't need $p(s',r|s,a)$ — learns directly from episodes of experience
2. **Computational focus**: MC can estimate value for a single state without computing all states (estimates are independent)

**Use DP when**: You have a complete model and a manageable state space.
**Use MC when**: No model available, or state space is too large for full sweeps.

---

## Part 2: On-policy and Off-policy MC

### Q2a: First-Visit vs Every-Visit Calculation (1.0p)

**Q:** Using policy $\pi$, trajectory $(s_0, a_0, 1, s_0, a_0, 1, s_0, a_1, 1, T)$. Calculate $v^\pi(s_0)$ for first-visit and every-visit MC. Use $\gamma = 1$.

### Solution

Trajectory visits $s_0$ at $t=0, 2, 4$ with rewards $R_1=1, R_2=1, R_3=1$.

Returns from each visit:
- $t=0$: $G_0 = 1 + 1 + 1 = 3$
- $t=2$: $G_2 = 1 + 1 = 2$ (note: $t=2$ means after the second transition, state visited again)
- $t=4$: $G_4 = 1$

Wait — let me reparse the trajectory. Format: $(S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, T)$

So: $S_0 = s_0, A_0 = a_0, R_1 = 1, S_1 = s_0, A_1 = a_0, R_2 = 1, S_2 = s_0, A_2 = a_1, R_3 = 1, T$

State $s_0$ visited at $t = 0, 1, 2$.

Returns ($\gamma = 1$):
- $G_0 = R_1 + R_2 + R_3 = 1 + 1 + 1 = 3$
- $G_1 = R_2 + R_3 = 1 + 1 = 2$
- $G_2 = R_3 = 1$

> [!formula] First-Visit MC
> Only use $G_0 = 3$ (first visit to $s_0$):
> $$v^\pi(s_0) = 3$$

> [!formula] Every-Visit MC
> Average all visits:
> $$v^\pi(s_0) = \frac{G_0 + G_1 + G_2}{3} = \frac{3 + 2 + 1}{3} = 2$$

### Q2b: Off-Policy Advantages (1.0p)

**Solution:**
- **Advantage of off-policy**: Can learn the optimal (greedy) policy while following an exploratory behavior policy. Separates exploration from the policy being learned.
- **Why soft policy from $\pi$ as behavior**: Ensures [[Importance Sampling#Coverage Requirement|coverage]] — every action that $\pi$ might take has non-zero probability under $b$. A soft (e.g., ε-greedy) version of $\pi$ guarantees this while staying close to $\pi$'s behavior.

### Q2c: Weighted IS Calculation (1.0p)

**Q:** Behavior policy $b$ sampled trajectory $(s_0, a_0, 1, s_0, a_0, 1, s_0, a_1, 1, T)$. Compute $v^b(s_0)$ and $v^\pi(s_0)$ using first-visit MC with weighted IS.

### Solution

**$v^b(s_0)$:** Same as first-visit MC under $b$. $G_0 = 3$, so $v^b(s_0) = 3$.

**$v^\pi(s_0)$** using weighted IS:
The [[Importance Sampling]] ratio for the full trajectory from $t=0$:
$$\rho_{0:2} = \frac{\pi(a_0|s_0)}{b(a_0|s_0)} \cdot \frac{\pi(a_0|s_0)}{b(a_0|s_0)} \cdot \frac{\pi(a_1|s_0)}{b(a_1|s_0)}$$

With only one episode, weighted IS gives:
$$v^\pi(s_0) = \frac{\rho_{0:2} \cdot G_0}{\rho_{0:2}} = G_0 = 3$$

> [!warning] Single-Episode Weighted IS
> With only one episode, weighted IS always returns $G_0$ regardless of the importance ratio (the ratio cancels). This is a known property — weighted IS has this bias for small sample sizes but lower variance overall.

### Q2d: Ordinary vs Weighted IS from Graph (0.5p)

**Q:** Line A shows ordinary IS, Line B shows weighted IS. How to distinguish?

**Solution:** **Ordinary IS** (Line A) has **higher variance** — larger fluctuations, potentially spiking high or low. **Weighted IS** (Line B) is **smoother** with lower variance. Ordinary IS is unbiased but can have extreme values; weighted IS is slightly biased but much more stable.

---

## Part 3: SARSA and Q-Learning Analysis

### Q3a: SARSA Convergence with Parameter n (1.5p)

**Q:** MDP with states $A$ (start), $B$, $C$, $T$ (terminal). ε-greedy with $\varepsilon = 0.2$, $\gamma = 1$. $n \in (-\infty, 2)$. Find converged Q-values under SARSA and determine policy preference.

### Solution

> [!tip] Key Insight
> [[SARSA]] learns the value of the **ε-greedy policy** (on-policy). The converged Q-values reflect the expected return including random exploration actions.

The actual Q-values depend on the specific transition structure of the MDP (which includes rewards on edges). Since SARSA is on-policy, the Q-values account for the $\varepsilon = 0.2$ probability of taking random actions.

The policy in state $A$ prefers $a_1$ when $Q(A, a_1) > Q(A, a_0)$, which depends on $n$ (the reward parameter on action $a_1$ in state $C$). The exact threshold depends on the MDP structure.

> [!warning] Without the Exact MDP Diagram
> This problem requires the specific MDP diagram (Figure 2 from the homework PDF) to compute exact values. The key principle: SARSA's converged Q-values include the effect of ε-greedy exploration, so the threshold for preferring $a_1$ vs $a_0$ will be different from Q-learning.

### Q3b: Q-Learning Convergence (1.5p)

**Q:** Same MDP but $n \in (-\infty, \infty)$ with Q-learning.

### Solution

[[Q-Learning]] converges to $q_*$ (optimal action-values) regardless of the behavior policy. Q-values reflect the **greedy** (optimal) policy, not the ε-greedy behavior.

The threshold for preferring $a_1$ will differ from SARSA because Q-learning doesn't penalize for exploration randomness.

### Q3c: Adding Action a₂ (1.0p)

**Q:** With $n = 1$, add action $a_2$ in $A$ (goes to $B$, reward 1). Would final performance differ for SARSA and Q-learning across the two MDPs?

### Solution

- **Q-Learning**: The final performance under the target policy $\pi$ (greedy w.r.t. $q_*$) **could change** — the new action $a_2$ might be optimal if its Q-value is highest. Q-learning would discover this.
- **SARSA**: The final performance **could also change** — SARSA learns the ε-greedy policy's value, and adding a new action changes the ε-greedy exploration probabilities (now $\varepsilon/3$ per random action instead of $\varepsilon/2$), affecting the learned values.

> [!intuition] The Key Point
> Q-learning's greedy policy will pick the best of the (now three) actions. SARSA's policy values change because the exploration distribution changes. Both algorithms are affected, but for different reasons.
