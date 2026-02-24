---
type: lecture
course: RL
week: 3
lecture: 6
book_sections: ["Ch 9.3-9.4", "Ch 9.7-9.8"]
topics:
  - "[[Linear Function Approximation]]"
  - "[[Feature Construction]]"
  - "[[Tile Coding]]"
  - "[[Semi-Gradient Methods]]"
  - "[[LSTD]]"
  - "[[Neural Network Function Approximation]]"
  - "[[TD Fixed Point]]"
status: complete
---

# RL Lecture 6: On-Policy TD Learning with Approximation

## Overview
This lecture explores how to extend [[Temporal Difference Learning]] to large or continuous state spaces using [[Function Approximation]]. We focus on **on-policy prediction**, where the goal is to estimate the value function $v_\pi$ for a fixed policy $\pi$ using parameterized functional forms instead of tables.

---

## 1. Value Function Approximation
In large state spaces, we cannot store a value for every state. Instead, we represent the value function with a weight vector $\mathbf{w} \in \mathbb{R}^d$:
$$\hat{v}(s, \mathbf{w}) \approx v_\pi(s)$$
Typically, $d \ll |S|$, meaning changing one weight affects many states (generalization).

> [!definition] Mean Squared Value Error (VE)
> To evaluate the approximation, we use the weighted mean squared error over the state distribution $\mu(s)$:
> $$\overline{VE}(\mathbf{w}) \doteq \sum_{s \in \mathcal{S}} \mu(s) [v_\pi(s) - \hat{v}(s, \mathbf{w})]^2$$
> Where $\mu(s)$ is usually the on-policy distribution (fraction of time spent in state $s$ under $\pi$).

---

## 2. Linear Function Approximation
A common and tractable case is [[Linear Function Approximation]], where the estimate is a linear combination of features:

> [!formula] Linear Value Function
> $$\hat{v}(s, \mathbf{w}) \doteq \mathbf{w}^\top \mathbf{x}(s) = \sum_{i=1}^d w_i x_i(s)$$
> where $\mathbf{x}(s)$ is a **feature vector** representing state $s$.

### Gradient Descent Updates
For linear methods, the gradient with respect to $\mathbf{w}$ is simply the feature vector:
$$\nabla \hat{v}(s, \mathbf{w}) = \mathbf{x}(s)$$

The general [[Stochastic Gradient Descent]] (SGD) update rule is:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha [U_t - \hat{v}(S_t, \mathbf{w}_t)] \nabla \hat{v}(S_t, \mathbf{w}_t)$$
For linear methods, this simplifies to:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha [U_t - \hat{v}(S_t, \mathbf{w}_t)] \mathbf{x}(S_t)$$

---

## 3. Semi-Gradient TD(0)
When the target $U_t$ depends on the current weights $\mathbf{w}_t$ (e.g., in [[Bootstrapping]]), the update does not follow the true gradient of the error. We call these **[[Semi-Gradient Methods]]**.

> [!formula] Semi-Gradient TD(0) Update
> $$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha [R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)] \nabla \hat{v}(S_t, \mathbf{w}_t)$$

### The TD Fixed Point
In the linear case, TD(0) converges to the **[[TD Fixed Point]]** $\mathbf{w}_{TD}$, which satisfies the following system of linear equations:
$$\mathbf{A}\mathbf{w}_{TD} = \mathbf{b}$$
Where:
- $\mathbf{A} \doteq \mathbb{E}[\mathbf{x}_t (\mathbf{x}_t - \gamma \mathbf{x}_{t+1})^\top]$
- $\mathbf{b} \doteq \mathbb{E}[R_{t+1} \mathbf{x}_t]$

> [!intuition] Convergence Bound
> While [[Monte Carlo Methods]] converge to the global minimum of the VE, linear TD(0) converges to a point $\mathbf{w}_{TD}$ whose error is bounded relative to the best possible error:
> $$\overline{VE}(\mathbf{w}_{TD}) \le \frac{1}{1-\gamma} \min_{\mathbf{w}} \overline{VE}(\mathbf{w})$$
> This expansion factor can be large if $\gamma \approx 1$.

### Pseudocode: Linear Semi-Gradient TD(0)
```pseudo
Algorithm: Linear Semi-Gradient TD(0) for estimating v̂ ≈ v_π
──────────────────────────────────────────────────────────────
Input: policy π, step-size α > 0
Input: differentiable v̂(s,w) = w^T · x(s)
Initialize: w arbitrarily (e.g., w = 0)

Loop for each episode:
  Initialize S
  Loop for each step of episode:
    Choose A ~ π(·|S)
    Take action A, observe R, S'
    If S' is terminal:
      w ← w + α[R - v̂(S,w)] · x(S)
      Go to next episode
    w ← w + α[R + γ·v̂(S',w) - v̂(S,w)] · x(S)
    S ← S'
```

---

## 4. Feature Construction
The performance of linear methods depends entirely on the choice of [[Feature Construction]].

### 4.1 Polynomials
States are represented as powers and products of state variables.
- Example for 2D state $(s_1, s_2)$: $\mathbf{x}(s) = (1, s_1, s_2, s_1 s_2, s_1^2, s_2^2, \dots)$
- Allows modeling interactions but doesn't scale well to high dimensions

### 4.2 Fourier Basis
Uses cosine functions of different frequencies:
$$x_i(s) = \cos(\pi \mathbf{s}^\top \mathbf{c}^i)$$
where $\mathbf{c}^i$ is an integer vector specifying frequencies along each dimension.

> [!tip] Step-Size Scaling for Fourier
> Konidaris et al. (2011) suggest per-feature step sizes: $\alpha_i = \alpha / \sqrt{(c_{i1})^2 + \cdots + (c_{ik})^2}$ (except when all $c_{ij} = 0$, use $\alpha_i = \alpha$).

### 4.3 Coarse Coding
Binary features representing overlapping "receptive fields" (e.g., circles in 2D). A state activates a feature if it falls inside the corresponding region.

- **Large receptive fields** → broad generalization, low resolution
- **Small receptive fields** → narrow generalization, high resolution
- **More features** → finer discrimination but slower learning

### 4.4 Radial Basis Functions (RBF)
Continuous version of coarse coding. Feature value depends on distance to center $c_i$:

> [!formula] RBF Feature
> $$x_i(s) = \exp\left(-\frac{\|s - c_i\|^2}{2\sigma_i^2}\right)$$

Provides smooth, differentiable approximation. Continuous-valued features (unlike binary coarse coding).

---

## 5. Tile Coding
[[Tile Coding]] is the most practically important feature construction method for RL.

> [!definition] Tilings and Tiles
> The state space is partitioned into a grid called a **tiling**. Each grid cell is a **tile** (a binary feature). Multiple overlapping tilings, each **offset** from the others, are used to achieve both generalization and fine resolution.

### How It Works
1. Define $n$ tilings over the state space, each a regular grid
2. Each tiling is offset by a fraction of the tile width
3. For a given state $s$: exactly **one tile per tiling** is active → $n$ active features total
4. $\hat{v}(s, \mathbf{w}) = \sum_{\text{active tiles } i} w_i$ (sum of active tile weights)

### Key Properties
- **Binary features**: Updates are just additions to active tile weights
- **Fixed cost**: Always exactly $n$ active features, regardless of state space size
- **Step-size scaling**: Use $\alpha = \alpha_0 / n$ to account for $n$ tilings contributing
- **Hashing**: Map large tile spaces to smaller arrays using hash function — handles curse of dimensionality

> [!warning] Displacement Vectors
> Uniform offsets (equal in all dimensions) create diagonal artifacts. **Asymmetric offsets** using displacement vectors like $(1, 3, 5, \dots)$ times the fundamental unit produce better, more isotropic generalization.

---

## 6. Least-Squares TD (LSTD)
Instead of iterative updates, [[LSTD]] estimates the $\mathbf{A}$ matrix and $\mathbf{b}$ vector directly from data to solve $\mathbf{w} = \mathbf{A}^{-1} \mathbf{b}$.

### The Algorithm
- $\hat{\mathbf{A}}_t \doteq \sum_{k=0}^{t-1} \mathbf{x}_k (\mathbf{x}_k - \gamma \mathbf{x}_{k+1})^\top + \epsilon \mathbf{I}$
- $\hat{\mathbf{b}}_t \doteq \sum_{k=0}^{t-1} R_{k+1} \mathbf{x}_k$

> [!formula] Sherman-Morrison Update
> To avoid $O(d^3)$ matrix inversion every step, update $\mathbf{A}^{-1}$ directly in $O(d^2)$:
> $$\hat{\mathbf{A}}_t^{-1} = \hat{\mathbf{A}}_{t-1}^{-1} - \frac{\hat{\mathbf{A}}_{t-1}^{-1} \mathbf{x}_{t-1} (\mathbf{x}_{t-1} - \gamma \mathbf{x}_t)^\top \hat{\mathbf{A}}_{t-1}^{-1}}{1 + (\mathbf{x}_{t-1} - \gamma \mathbf{x}_t)^\top \hat{\mathbf{A}}_{t-1}^{-1} \mathbf{x}_{t-1}}$$

### LSTD Trade-offs

| | LSTD | Semi-Gradient TD |
|---|---|---|
| Step-size $\alpha$? | No (direct solution) | Yes (sensitive to tuning) |
| Data efficiency | Higher (no data wasted) | Lower (iterative) |
| Per-step computation | $O(d^2)$ | $O(d)$ |
| Memory | $O(d^2)$ (stores $\mathbf{A}^{-1}$) | $O(d)$ |

> [!tip] LSTD "Never Forgets"
> LSTD uses all past transitions equally — the TD fixed point depends on all data ever seen. This is sample efficient but problematic if the policy or environment changes (non-stationarity).

---

## 7. Neural Network Function Approximation
[[Neural Network Function Approximation]] allows for nonlinear value functions: $\hat{v}(s, \mathbf{w}) = \text{NN}_\mathbf{w}(s)$.

### Architecture
A feedforward network maps state features through hidden layers:
$$\hat{v}(s, \mathbf{W}^{(1)}, \mathbf{W}^{(2)}) = \sum_{m=0}^{M} w_m^{(2)} \, h\left(\sum_{d=0}^{D} w_{md}^{(1)} s_d\right)$$
where $h(\cdot)$ is a non-linear activation function (ReLU, sigmoid, etc.).

### Semi-Gradient Update with Neural Nets
Same update rule as linear case, but gradient computed via backpropagation:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \, \delta_t \, \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w}_t)$$

> [!warning] Challenges of Neural Networks in RL
> 1. **Non-stationarity**: Targets change as the network learns (bootstrapping moves the goal)
> 2. **Correlated data**: Sequential RL data violates i.i.d. assumption of SGD
> 3. **Catastrophic forgetting**: Learning new states can degrade performance on previously learned states
> 4. **No convergence guarantees**: Unlike linear semi-gradient TD, non-linear methods have no guaranteed convergence

These challenges motivate the stabilization techniques in [[Deep Q-Network (DQN)]]: [[Experience Replay]] and [[Target Network]].

---

## 8. Summary: Method Comparison

| Feature Type | Representation | Key Property |
|:---|:---|:---|
| **State Aggregation** | One-hot over partitions | Simplest; piecewise constant |
| **Polynomials** | Powers of state variables | Global; poor scaling |
| **Fourier Basis** | Cosine functions | Good for smooth functions |
| **Coarse Coding** | Binary overlapping regions | Local generalization |
| **[[Tile Coding]]** | Multiple offset grids | Efficient; tunable; practical |
| **RBF** | Gaussian bumps | Smooth; computationally expensive |
| **Neural Networks** | Learned non-linear features | Most expressive; least stable |
