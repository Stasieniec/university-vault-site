---
type: exercise
course: RL
week: 3
source: "Homework 3 (Coding Assignment 3)"
concepts:
  - "[[SARSA]]"
  - "[[Q-Learning]]"
  - "[[On-Policy vs Off-Policy]]"
  - "[[Function Approximation]]"
  - "[[Mean Squared Value Error]]"
  - "[[Linear Function Approximation]]"
  - "[[Semi-Gradient Methods]]"
  - "[[Stochastic Gradient Descent]]"
  - "[[Value Iteration]]"
status: complete
---

# RL-HW03: Homework 3 — TD Learning & Function Approximation

> [!tip] Exam Relevance
> Q3 (minimal value error) and Q4 (semi-gradient derivation) are *extremely* exam-relevant. Understanding the VE objective, weighted least squares, and the "semi" in semi-gradient is core material.

---

## Part 2: Coding Assignment Questions (SARSA vs Q-Learning)

### Q2a: Average Returns Comparison (0.25p)

**Q:** Which algorithm achieves higher average return of the behavior policy during training? Same phenomenon as Cliff Walking (Example 6.6)?

### Solution

In the Windy Gridworld:
- **[[SARSA]]** typically achieves **higher average return** during training
- This is the **same phenomenon** as Cliff Walking: SARSA (on-policy) learns to avoid risky areas that the ε-greedy exploration might stumble into, leading to better average performance of the *behavior* policy
- Q-learning finds the optimal path but the ε-greedy behavior policy occasionally deviates from it, leading to lower average returns during training

### Q2b: Return Variance (0.25p)

**Q:** Which algorithm achieves smaller return variance?

### Solution

**SARSA** achieves smaller variance. Because it learns a policy that accounts for exploration, the returns are more consistent. Q-learning's greedy policy walks near dangerous regions (optimal but risky under ε-greedy), causing occasional large negative returns.

### Q2c: When Are They the Same? (0.25p)

**Q:** Under which condition do SARSA and Q-learning behave the same?

### Solution

> [!formula] SARSA = Q-Learning when $\varepsilon = 0$
> When the behavior policy is **greedy** ($\varepsilon = 0$), the action $A_{t+1}$ chosen by SARSA equals $\arg\max_a Q(S_{t+1}, a)$, which is exactly the $\max_a Q(S_{t+1}, a)$ used by Q-learning. The updates become identical.

### Q2d: Which Is Off-Policy? (0.25p)

**Q:** Which is off-policy and why?

### Solution

**[[Q-Learning]]** is off-policy. Its update target uses $\max_a Q(S_{t+1}, a)$ — the value of the **greedy** (optimal) policy — regardless of what action the behavior policy actually takes next. The behavior policy (ε-greedy) ≠ the target policy (greedy).

[[SARSA]] is on-policy: its target uses $Q(S_{t+1}, A_{t+1})$ where $A_{t+1}$ is the action actually taken by the current policy.

---

## Part 3: Minimal Value Error

### Q3a: Find $v_*$ Using Value Iteration (1.0p)

**Q:** For the 4-state MDP in Figure 4 (deterministic actions, rewards on transitions), find $v_*$.

### Solution

> [!warning] Requires MDP Diagram
> The specific MDP diagram (Figure 4) shows 4 states with deterministic actions and rewards. Apply [[Value Iteration]]:
> $$V_{k+1}(s) = \max_a [r(s,a) + \gamma V_k(s')]$$
> 
> With $\gamma$ from the problem, iterate until convergence. Since actions are deterministic, this reduces to a simple evaluation of each action's reward + discounted next-state value.

### Q3b: On-Policy Distribution $\mu(s)$ (1.0p)

**Q:** Starting in each state with equal probability, following optimal policy, what is $\mu(s)$?

### Solution

> [!definition] On-Policy Distribution
> $\mu(s)$ is the fraction of time spent in each state under the given policy and starting state distribution.

With uniform starting distribution and deterministic optimal policy:
- Trace the trajectories from each starting state under $\pi_*$
- Count the fraction of total time steps spent in each state
- $\mu(s) = \frac{\text{time steps in } s}{\text{total time steps across all trajectories}}$

### Q3c: Minimize VE with Linear FA (1.5p)

**Q:** Given features $\phi(s_1) = \begin{bmatrix} 0 \\ 4 \end{bmatrix}$, $\phi(s_2) = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$, $\phi(s_3) = \begin{bmatrix} 3 \\ 0 \end{bmatrix}$, $\phi(s_4) = \begin{bmatrix} 2 \\ 0 \end{bmatrix}$, $\phi(T) = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$. Find weights $\mathbf{w}$ minimizing VE.

### Solution

> [!formula] Weighted Least Squares
> The [[Mean Squared Value Error]]:
> $$\overline{VE}(\mathbf{w}) = \sum_s \mu(s)[v_*(s) - \mathbf{w}^\top \phi(s)]^2$$
> 
> This is a weighted least squares problem. In matrix form:
> $$\mathbf{w}^* = (\Phi^\top D \Phi)^{-1} \Phi^\top D \mathbf{v}_*$$
> 
> where:
> - $\Phi$ is the feature matrix (rows = feature vectors for each state)
> - $D = \text{diag}(\mu(s_1), \mu(s_2), \mu(s_3), \mu(s_4))$ is the diagonal weight matrix
> - $\mathbf{v}_*$ is the vector of optimal values from Q3a

> [!tip] Hint from the problem
> Use weighted least squares directly. Set up $\Phi$, $D$, and $\mathbf{v}_*$, then solve. Note: the terminal state has $\phi(T) = \mathbf{0}$, so its value is automatically 0 regardless of $\mathbf{w}$.

### Q3d: Relationship to Gradient MC (1.0p)

**Q:** What values does $\tilde{v}$ assign to each state? Relationship to gradient MC?

### Solution

$\tilde{v}(s) = \mathbf{w}^{*\top} \phi(s)$ for each state.

> [!intuition] Connection to Gradient MC
> **Gradient MC converges to the same minimum** — the weights that minimize $\overline{VE}$. The analytical solution (weighted least squares) gives us the exact answer that gradient MC would converge to with infinite data and proper step-size schedule. Gradient MC performs stochastic gradient descent on the VE objective.

---

## Part 4: (Semi-) Gradient Descent Methods

### Q4a: Unbiased Estimators Analysis (2.5p)

**Q:** For the update $\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha_t[U_t - \hat{v}(S_t, \mathbf{w}_t)] \nabla \hat{v}(S_t, \mathbf{w}_t)$, determine which targets are unbiased estimators of $v_\pi(s)$:
1. $U_t = G_t$
2. $U_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t)$
3. $U_t = \sum_{a,s',r} \pi(a|S_t) p(s',r|S_t,a)[r + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t)]$

### Solution

> [!formula] Bias Analysis

**1. $U_t = G_t$ — UNBIASED ✅**
By definition: $\mathbb{E}_\pi[G_t | S_t = s] = v_\pi(s)$. The actual return is an unbiased estimate.
→ This is **Gradient Monte Carlo**

**2. $U_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t)$ — BIASED ❌**
$\mathbb{E}[U_t | S_t = s] = \mathbb{E}[R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) | S_t = s]$
This equals $v_\pi(s)$ only if $\hat{v} = v_\pi$, which is generally not true during learning. The bootstrapped estimate introduces bias.
→ This is **Semi-Gradient TD(0)**

**3. $U_t = \sum_{a,s',r} \pi(a|S_t) p(s',r|S_t,a)[r + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t)]$ — BIASED ❌**
Same issue: uses $\hat{v}$ instead of $v_\pi$. This is the expected update (no sampling), but still biased.
→ This is the **Expected (DP-like) update**

**Which guarantees convergence to local optimum of VE?**
Only $U_t = G_t$ (Gradient MC), because it's the only unbiased estimator, making the update a true stochastic gradient of $\overline{VE}$.

**Example step size**: $\alpha_t = 1/t$ satisfies $\sum \alpha_t = \infty$, $\sum \alpha_t^2 < \infty$.

**Why use biased estimators?**
- Lower variance → faster learning in practice
- Can learn online (step-by-step) without waiting for episode end
- Example: continuing tasks (no episodes → MC impossible); or environments with very long episodes where MC has extreme variance

### Q4b: Derive Mean Squared TD Error Minimization (2.0p)

**Q:** Derive weight update that minimizes the mean squared TD error. Compare to Semi-Gradient TD.

### Solution

> [!formula] Mean Squared TD Error
> $$\text{MSTDE}(\mathbf{w}) = \mathbb{E}\left[\delta_t^2\right] = \mathbb{E}\left[\left(R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})\right)^2\right]$$

Taking the gradient:
$$\nabla_\mathbf{w} \text{MSTDE} = 2\mathbb{E}\left[\delta_t \cdot \nabla_\mathbf{w}\delta_t\right]$$

$$\nabla_\mathbf{w}\delta_t = \nabla_\mathbf{w}\left[R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})\right] = \gamma \nabla\hat{v}(S_{t+1}, \mathbf{w}) - \nabla\hat{v}(S_t, \mathbf{w})$$

> [!formula] Full Gradient Update (True Gradient of MSTDE)
> $$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \delta_t \left[\gamma \nabla\hat{v}(S_{t+1}, \mathbf{w}_t) - \nabla\hat{v}(S_t, \mathbf{w}_t)\right]$$
> $$= \mathbf{w}_t + \alpha \delta_t \nabla\hat{v}(S_t, \mathbf{w}_t) - \alpha \gamma \delta_t \nabla\hat{v}(S_{t+1}, \mathbf{w}_t)$$

> [!warning] Comparison with Semi-Gradient TD
> **Semi-Gradient TD** uses only:
> $$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \delta_t \nabla\hat{v}(S_t, \mathbf{w}_t)$$
> 
> It **drops** the $-\alpha\gamma\delta_t \nabla\hat{v}(S_{t+1}, \mathbf{w}_t)$ term — the gradient through the bootstrapped target. This is why it's called "**semi**-gradient": it only takes half the gradient (the part through the prediction, not through the target).
> 
> The missing term is exactly the correction that [[Gradient-TD Methods]] (TDC) add back.
