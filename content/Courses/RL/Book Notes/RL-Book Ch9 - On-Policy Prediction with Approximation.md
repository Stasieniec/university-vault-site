---
type: book-chapter
course: RL
book: "Reinforcement Learning: An Introduction (2nd ed.)"
chapter: 9
sections: ["9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7", "9.8", "9.9", "9.10", "9.11", "9.12"]
topics:
  - "[[Function Approximation]]"
  - "[[Stochastic Gradient Descent]]"
  - "[[Semi-Gradient Methods]]"
  - "[[Linear Function Approximation]]"
  - "[[Tile Coding]]"
  - "[[Feature Construction]]"
  - "[[Neural Network Function Approximation]]"
  - "[[LSTD]]"
  - "[[Mean Squared Value Error]]"
status: complete
---

# Chapter 9: On-policy Prediction with Approximation

## Overview
This chapter explores how to estimate the state-value function $v_\pi$ from on-policy data when the state space is too large for tabular representations. We transition from tables to **parameterized functional forms** with a weight vector $\mathbf{w} \in \mathbb{R}^d$, where $d \ll |\mathcal{S}|$.

> [!intuition]
> In function approximation, an update to one state affects many others through **generalization**. This makes learning more powerful but also more complex to manage, as we cannot get the values of all states exactly correct.

---

## 9.1 Value-function Approximation
All prediction methods are viewed as updates $s \mapsto u$, where $s$ is the state and $u$ is the target.
- **Monte Carlo**: $S_t \mapsto G_t$
- **TD(0)**: $S_t \mapsto R_{t+1} + \hat{v}(S_{t+1}, \mathbf{w}_t)$
- **n-step TD**: $S_t \mapsto G_{t:t+n}$

This is essentially a **supervised learning** task where we provide $(s, u)$ training examples to a [[Function Approximation]] method. However, RL requires methods that can handle:
1. **Online Learning**: Learning while interacting with the environment.
2. **Nonstationary Targets**: The target $v_\pi$ changes as the policy improves (control) or because of [[Bootstrapping]] (TD/DP).

---

## 9.2 The Prediction Objective ($\overline{VE}$)
In the tabular case, we could reach $v_\pi(s)$ exactly for all $s$. With approximation, we must decide which states to prioritize using a state distribution $\mu(s)$.

> [!formula] Mean Squared Value Error (VE)
> $$\overline{VE}(\mathbf{w}) \doteq \sum_{s \in \mathcal{S}} \mu(s) [v_\pi(s) - \hat{v}(s, \mathbf{w})]^2$$
> Where $\mu(s)$ is typically the **on-policy distribution** (fraction of time spent in $s$ under policy $\pi$).

Under on-policy training, $\mu(s)$ for a continuing task is the stationary distribution. For episodic tasks:
$$\eta(s) = h(s) + \sum_{\bar{s} \in \mathcal{S}} \eta(\bar{s}) \sum_a \pi(a|\bar{s}) p(s|\bar{s}, a)$$
$$\mu(s) = \frac{\eta(s)}{\sum_{s'} \eta(s')}$$

---

## 9.3 Stochastic Gradient and Semi-gradient Methods
[[Stochastic Gradient Descent]] (SGD) is ideal for online RL. We adjust weights to reduce the error on the current example:

> [!formula] SGD Update Rule
> $$\mathbf{w}_{t+1} \doteq \mathbf{w}_t - \frac{1}{2} \alpha \nabla [v_\pi(S_t) - \hat{v}(S_t, \mathbf{w}_t)]^2 = \mathbf{w}_t + \alpha [v_\pi(S_t) - \hat{v}(S_t, \mathbf{w}_t)] \nabla \hat{v}(S_t, \mathbf{w}_t)$$
> where $\nabla \hat{v}(S_t, \mathbf{w}_t)$ is the **gradient** vector of partial derivatives with respect to $\mathbf{w}$.

### Gradient Monte Carlo
Since $G_t$ is an unbiased estimate of $v_\pi(S_t)$, MC using SGD converges to a local optimum.

```python
# Gradient Monte Carlo Algorithm
Initialize w arbitrarily
Loop for each episode:
    Generate episode S_0, A_0, R_1, ..., S_T using pi
    Loop for each step t = 0, ..., T-1:
        w <- w + alpha * [G_t - v_hat(S_t, w)] * grad_v_hat(S_t, w)
```

### Semi-Gradient Methods
When we use [[Bootstrapping]] (TD), the target depends on the weights $\mathbf{w}_t$, meaning the update is not a true gradient. We call these [[Semi-Gradient Methods]].

> [!formula] Semi-gradient TD(0) Update
> $$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha [R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)] \nabla \hat{v}(S_t, \mathbf{w}_t)$$

```python
# Semi-gradient TD(0) Algorithm
Initialize w arbitrarily
Loop for each episode:
    Initialize S
    Loop for each step of episode:
        Choose A ~ pi(.|S)
        Take action A, observe R, S'
        w <- w + alpha * [R + gamma*v_hat(S', w) - v_hat(S, w)] * grad_v_hat(S, w)
        S <- S'
    Until S is terminal
```

> [!warning] 
> Semi-gradient methods do not converge as robustly as gradient methods but often learn faster and enable online learning.

---

## 9.4 Linear Methods
In [[Linear Function Approximation]], the estimate is the inner product of weights and a **feature vector** $\mathbf{x}(s)$:
$$\hat{v}(s, \mathbf{w}) \doteq \mathbf{w}^\top \mathbf{x}(s) = \sum_{i=1}^d w_i x_i(s)$$
Here, the gradient is simply the feature vector: $\nabla \hat{v}(s, \mathbf{w}) = \mathbf{x}(s)$.

### The TD Fixed Point
Linear semi-gradient TD(0) converges to the **TD fixed point**:
$$\mathbf{w}_{TD} = \mathbf{A}^{-1} \mathbf{b}$$
Where:
- $\mathbf{A} \doteq \mathbb{E}[\mathbf{x}_t (\mathbf{x}_t - \gamma \mathbf{x}_{t+1})^\top]$
- $\mathbf{b} \doteq \mathbb{E}[R_{t+1} \mathbf{x}_t]$

At this point, the error is bounded: $\overline{VE}(\mathbf{w}_{TD}) \leq \frac{1}{1-\gamma} \min_\mathbf{w} \overline{VE}(\mathbf{w})$.

---

## 9.5 Feature Construction
The choice of features determines how the agent generalizes.

### Polynomials and Fourier Basis
- **Polynomials**: Features are combinations of state dimensions (e.g., $1, s_1, s_2, s_1 s_2$). They have difficulty with high dimensionality.
- **Fourier Basis**: Uses cosine functions of different frequencies: $x_i(s) = \cos(\pi \mathbf{s}^\top \mathbf{c}_i)$. Often performs better than polynomials in RL.

### Coarse Coding
States are represented by overlapping binary features (e.g., circles in a 2D space).
- **Narrow features**: Fine discrimination, slow generalization.
- **Broad features**: Broad generalization, coarse initial approximation.

### Tile Coding
A computationally efficient form of coarse coding using shifted grids (tilings).
- Each state falls into exactly one **tile** per **tiling**.
- Total number of active features = number of tilings.
- **Hashing** can be used to reduce memory requirements.

> [!example] Tile Coding Illustration
> If you have 8 tilings shifted asymmetrically, a single point in state space activates 1 feature in each tiling. Generalization occurs to any state that shares one or more tiles.

### Radial Basis Functions (RBFs)
Continuous version of coarse coding. Features have a Gaussian response:
$$x_i(s) = \exp\left(-\frac{\|\mathbf{s} - \mathbf{c}_i\|^2}{2\sigma_i^2}\right)$$

---

## 9.7 Neural Network Function Approximation
[[Neural Network Function Approximation]] uses multi-layer ANNs to learn non-linear approximations.
- **Hidden layers**: Automatically create features.
- **Backpropagation**: Computes gradients of the loss with respect to weights.
- **Deep RL**: Successes in Go and Atari rely on deep convolutional networks that extract hierarchical spatial features.

---

## 9.8 Least-Squares TD (LSTD)
Instead of iterative updates, LSTD estimates $\mathbf{A}$ and $\mathbf{b}$ directly.
- **Complexity**: $O(d^2)$ per step using the Sherman-Morrison formula for recursive matrix inversion.
- **Data Efficiency**: Most efficient linear TD method; no step-size $\alpha$ needed (though it needs a regularization parameter $\epsilon$).

---

## 9.11 Interest and Emphasis
We can focus approximation on specific "interesting" states using:
1. **Interest** $I_t$: How much we care about the error at time $t$.
2. **Emphasis** $M_t$: A multiplier for the update that maintains stability.
$$M_t = I_t + \gamma^n M_{t-n}$$

---

## Summary
- **Function Approximation** is necessary for large state spaces.
- **SGD** provides the theoretical bedrock for updates.
- **Linear Methods** with [[Tile Coding]] or [[Fourier Basis]] are robust and efficient.
- **Deep Learning** allows for complex, non-linear feature discovery.
- There is a trade-off between **Local Optimum** convergence (Gradient MC) and the faster, biased convergence of [[Semi-Gradient Methods]].
