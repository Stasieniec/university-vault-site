---
type: concept
aliases: [function approximation, value function approximation, FA, Value Function Approximation]
course: [RL]
tags: [approximation, exam-topic]
status: complete
---

# Function Approximation

## Definition

> [!definition] Function Approximation in RL
> **Function approximation** replaces the lookup table used in tabular RL with a parameterized function $\hat{v}(s, \mathbf{w}) \approx v_\pi(s)$ (or $\hat{q}(s, a, \mathbf{w}) \approx q_\pi(s,a)$). Instead of storing one value per state, we learn a weight vector $\mathbf{w} \in \mathbb{R}^d$ where $d \ll |\mathcal{S}|$.

> [!intuition] Why We Need It
> Tabular methods store $V(s)$ for every state. Real problems have millions or billions of states (or continuous state spaces). You can't visit every state, let alone store a value for each. Function approximation lets you **generalize** — update $\mathbf{w}$ from one state, and the values of similar states change too.

## The Prediction Objective

> [!formula] Mean Squared Value Error ($\overline{VE}$)
> $$\overline{VE}(\mathbf{w}) = \sum_{s \in \mathcal{S}} \mu(s) \left[ v_\pi(s) - \hat{v}(s, \mathbf{w}) \right]^2$$
> 
> where:
> - $\mu(s)$ — **on-policy distribution** (how often state $s$ is visited under $\pi$)
> - $v_\pi(s)$ — true value (unknown)
> - $\hat{v}(s, \mathbf{w})$ — our approximation

## Stochastic Gradient Descent

> [!formula] SGD Update for Value Prediction
> $$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ v_\pi(S_t) - \hat{v}(S_t, \mathbf{w}_t) \right] \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w}_t)$$
> 
> Problem: we don't know $v_\pi(S_t)$. Replace it with a target:
> - **MC target**: $G_t$ → true gradient method (converges to local minimum of $\overline{VE}$)
> - **TD target**: $R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w})$ → **[[Semi-Gradient Methods|semi-gradient]]** (not true gradient because target depends on $\mathbf{w}$)

## Types of Function Approximators

### [[Linear Function Approximation]]
$$\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s) = \sum_{i=1}^d w_i x_i(s)$$
- $\mathbf{x}(s)$ is a feature vector
- Simple, well-understood convergence guarantees
- Feature design matters: [[Tile Coding]], polynomials, Fourier basis, RBFs

### [[Neural Network Function Approximation]]
$$\hat{v}(s, \mathbf{w}) = f_\mathbf{w}(s)$$
- Non-linear, can represent complex functions
- Trained with backpropagation
- Fewer convergence guarantees
- Foundation of [[Deep Q-Network (DQN)]] and modern deep RL

## Key Distinction: Tabular as Special Case

> [!tip] Tabular = Function Approximation with One-Hot Features
> A lookup table is actually a special case of linear function approximation where the feature vector $\mathbf{x}(s)$ is a one-hot vector (1 in position $s$, 0 elsewhere). Then $\hat{v}(s, \mathbf{w}) = w_s$ — each state has its own weight. All tabular convergence guarantees follow from the more general FA framework.

## Challenges

- **Generalization**: Updating one state affects nearby states — can be good (efficiency) or bad (interference)
- **The [[Deadly Triad]]**: Function approximation + bootstrapping + off-policy = potential divergence
- **Non-stationarity**: Target values change as $\mathbf{w}$ updates

## Connections

- Extends: [[Value Function]] (from tables to functions)
- Methods: [[Semi-Gradient Methods]], [[LSTD]], [[Linear Function Approximation]]
- Features: [[Tile Coding]], [[Feature Construction]]
- Deep version: [[Deep Q-Network (DQN)]], [[Neural Network Function Approximation]]
- Danger: [[Deadly Triad]]

## Appears In

- [[RL-L05 - Tabular to Approximation]]
- [[RL-L06 - On-Policy TD with Approximation]]
- [[RL-L07 - Off-Policy RL with Approximation]]
- [[RL-Book Ch9 - On-Policy Prediction with Approximation]]
