---
type: lecture
course: RL
week: 3
lecture: 5
book_sections: ["Ch 9.1-9.3"]
topics:
  - "[[Function Approximation]]"
  - "[[Value Function Approximation]]"
  - "[[Stochastic Gradient Descent]]"
  - "[[Semi-Gradient Methods]]"
  - "[[Mean Squared Value Error]]"
  - "[[On-Policy Distribution]]"
status: complete
---

# RL Lecture 5: From Tabular Learning to Approximation

## 0. The Big Picture: Where we are
So far, we have covered model-free methods that learn directly from experience:
*   **Monte Carlo (MC)**: Uses full episode returns $G_t$.
*   **Temporal Difference (TD)**: Uses one-step bootstrapping targets $R + \gamma \hat{V}(S')$.
*   **Q-learning/SARSA**: Control methods for finding optimal policies.

> [!important] The Taxonomy
> *   **Model-based (DP)**: Requires $p(s',r|s,a)$. Uses "Full Backup" (lookahead tree).
> *   **Model-free (RL)**: Only requires data.
>     *   **Monte Carlo**: Sample episodes (Horizontal chain).
>     *   **Temporal Difference**: Sample steps (1-step arrow).

### Why no Importance Sampling in Q-learning?
In off-policy Monte Carlo, we need importance weights $\rho$ to correct the return $G_t$ from the behavior policy $b$ to the target policy $\pi$.
In **Q-learning**, we do not need importance weights because the target is computed *using* the target policy (by taking the `max` over actions), rather than using a sampled return from the behavior policy.

### The RL Methodology Space (Slide 15)
> [!info] The 2D Space of RL
> Reinforcement learning methods can be mapped along two axes:
> 1.  **Sampling vs. Exhaustive (Horizontal)**: Does the method use samples (MC/TD) or exhaustive backups over all possible next states (DP)?
> 2.  **Width vs. Depth (Vertical)**: Does the method use partial bootstrapping (TD) or full-depth returns (MC)?
>
> VFA (Function Approximation) sits atop this space, allowing us to apply these concepts to larger, continuous domains where neither a full table nor exhaustive backups are possible.

---

## 1. Why Function Approximation?
Tabular methods, where each state $s$ has a dedicated entry $V(s)$ in a lookup table, fail in most real-world applications due to:
*   **Curse of Dimensionality**: The number of states grows exponentially with the state variables (e.g., in Backgammon $\approx 10^{20}$, in Go $\approx 10^{170}$).
*   **Generalization**: In large/continuous spaces, we almost never see the exact same state twice. We need a way to generalize from limited experience to "similar" unseen states.

> [!abstract] Key Shift
> Instead of a table, we use a **parameterized functional form** $\hat{v}(s, \mathbf{w}) \approx v_\pi(s)$, where $\mathbf{w} \in \mathbb{R}^d$ is a weight vector with significantly fewer parameters than states ($d \ll |\mathcal{S}|$).

---

## 2. Value Function Approximation (VFA) Setup
The approximate value function is represented as a differentiable function of a weight vector $\mathbf{w}$.
*   **Linear Methods**: $\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s) = \sum_{i=1}^d w_i x_i(s)$
    *   $\mathbf{x}(s)$ is a **feature vector** representing state $s$.
*   **Non-linear Methods**: e.g., Neural Networks where $\mathbf{w}$ represents connection weights.

### Tabular as a Special Case
Tabular learning is a special case of linear function approximation where:
*   $d = |\mathcal{S}|$
*   $\mathbf{x}(s)$ is a one-hot (indicator) vector: $x_i(s) = 1$ if $i=s$, else $0$.
*   Updating one state does **not** affect others (zero generalization).

---

## 3. The Prediction Objective: Mean Squared Value Error ($\overline{\text{VE}}$)
In approximation, we cannot match the true value $v_\pi(s)$ exactly for all states. We must decide which states matter more using a **state distribution** $\mu(s)$, where $\sum_s \mu(s) = 1$.

The **Mean Squared Value Error ($\overline{\text{VE}}$)** is defined as:
$$\overline{\text{VE}}(\mathbf{w}) \doteq \sum_{s \in \mathcal{S}} \mu(s) \left[ v_\pi(s) - \hat{v}(s, \mathbf{w}) \right]^2$$

*   **$\mu(s)$ (On-policy distribution)**: Usually the fraction of time spent in state $s$ under policy $\pi$.
    *   In continuing tasks: The stationary distribution.
    *   In episodic tasks: Depends on the start state distribution $h(s)$ and transition probability.

---

## 4. Stochastic Gradient Descent (SGD)
To minimize $\overline{\text{VE}}$, we adjust weights in the direction of the negative gradient.

### Ideal Gradient Update
If the true value $v_\pi(S_t)$ was known:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ v_\pi(S_t) - \hat{v}(S_t, \mathbf{w}_t) \right] \nabla \hat{v}(S_t, \mathbf{w}_t)$$

Where $\nabla \hat{v}(s, \mathbf{w})$ is the gradient vector of partial derivatives with respect to $\mathbf{w}$:
$$\nabla \hat{v}(s, \mathbf{w}) \doteq \left[ \frac{\partial \hat{v}(s, \mathbf{w})}{\partial w_1}, \frac{\partial \hat{v}(s, \mathbf{w})}{\partial w_2}, \dots, \frac{\partial \hat{v}(s, \mathbf{w})}{\partial w_d} \right]^\top$$

### General SGD with Targets
Since $v_\pi(S_t)$ is unknown, we use a target $U_t$:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ U_t - \hat{v}(S_t, \mathbf{w}_t) \right] \nabla \hat{v}(S_t, \mathbf{w}_t)$$

*   **Monte Carlo Target**: $U_t = G_t$ (unbiased, converges to local optimum).
*   **TD Target**: $U_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t)$ (biased, bootstrapping).

---

## 5. Semi-Gradient TD(0)
When using bootstrapping targets like in TD, the target $U_t$ *depends* on the current weights $\mathbf{w}_t$. A true gradient would need to take the derivative of both the prediction AND the target.

**Semi-gradient methods** ignore the dependence of the target on $\mathbf{w}$. They only take the gradient of the prediction $\hat{v}(S_t, \mathbf{w}_t)$.

### Why "Semi"?
*   **True Gradient**: $\nabla \left( \mathbb{E}[R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w})] - \hat{v}(S_t, \mathbf{w}) \right)^2$ would include $\nabla \hat{v}(S_{t+1}, \mathbf{w})$.
*   **Semi-Gradient**: Treats $R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w})$ as a constant during the update.

### Pseudocode: Semi-gradient TD(0)
> [!note] Algorithm: Semi-gradient TD(0) for estimating $\hat{v} \approx v_\pi$
> **Input:** Policy $\pi$, differentiable function $\hat{v}(s, \mathbf{w})$
> **Parameters:** Step size $\alpha > 0$
> **Initialize:** $\mathbf{w}$ arbitrarily (e.g., $\mathbf{w} = \mathbf{0}$)
> 
> Loop for each episode:
>   Initialize $S$
>   Loop for each step of episode:
>     Choose $A \sim \pi(\cdot|S)$
>     Take action $A$, observe $R, S'$
>     $\mathbf{w} \leftarrow \mathbf{w} + \alpha [R + \gamma \hat{v}(S', \mathbf{w}) - \hat{v}(S, \mathbf{w})] \nabla \hat{v}(S, \mathbf{w})$
>     $S \leftarrow S'$
>   until $S$ is terminal

### Convergence Properties
*   **Gradient MC**: Guaranteed to converge to a local optimum (global optimum for linear cases).
*   **Semi-Gradient TD**: Converges to a **TD Fixed Point** $\mathbf{w}_{TD}$, which is near the global optimum.
*   **Error Bound**: $\overline{\text{VE}}(\mathbf{w}_{TD}) \leq \frac{1}{1-\gamma} \min_{\mathbf{w}} \overline{\text{VE}}(\mathbf{w})$.
    *   As $\gamma \to 1$, the bound becomes loose.

---

## 6. Visualizing Approximation (1000-state Random Walk)
Based on the lecture slides and Chapter 9 figures:

### State Aggregation (Slide/Book Fig 9.1)
This is a form of function approximation where states are grouped.
*   **States 1-1000**: Divided into 10 groups of 100.
*   **Staircase effect**: The learned value function is constant within each group.
*   **Distribution bias**: Because states in the center (near 500) are visited more often ($\mu(s)$ is higher), the approximation is more accurate there.

### Learning Targets Comparison (Slide/Book Fig 9.2)
*   **Monte Carlo**: Asymptotic error is lower (can reach global optimum).
*   **n-step TD**: Faster initial learning but potentially higher asymptotic error.
*   **Linearity**: In the linear case, TD(0) convergence is stable on-policy but can diverge off-policy (the "Deadly Triad").

---

## 7. Worked Example Summary
*Trajectory: 500 $\xrightarrow{R=0}$ 501 $\xrightarrow{R=0}$ 502 ...*
With initial $\mathbf{w}=\mathbf{0}$, $\alpha=0.1$, and linear features (bins):
1.  **State 500 (Bin 5)**: $\hat{v}(500, \mathbf{w}) = 0$.
2.  **Observed Reward $R=0$, Next State 501 (Bin 6)**.
3.  **Target**: $0 + 1.0 \times \hat{v}(501, \mathbf{w}) = 0$.
4.  **Error**: $0 - 0 = 0$. Weight stays $\mathbf{0}$.
5.  *Later in episode, when reaching terminal state with $R=1$*:
    The weights for the bins leading to the end will increase based on the backpropagated rewards (bootstrapping for TD, or full return for MC).

---
## Summary Table: Tabular vs. Function Approx
| Feature | Tabular | Function Approximation |
| :--- | :--- | :--- |
| **Resolution** | Single state level | Grouped/Functional level |
| **Generalization** | None | High (through shared $\mathbf{w}$) |
| **State Space** | Small/Discrete | Large/Continuous |
| **Memory** | $\mathcal{O}(|\mathcal{S}|)$ | $\mathcal{O}(d)$ |
| **Update Impact**| Local to state $S_t$ | Global (affects many states) |
