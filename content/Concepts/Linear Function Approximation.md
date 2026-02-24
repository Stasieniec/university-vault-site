---
type: concept
aliases: [linear FA, linear function approximation, linear methods]
course: [RL]
tags: [approximation, key-formula, exam-topic]
status: complete
---

# Linear Function Approximation

> [!definition] Linear Function Approximation
> The value function is approximated as a **linear combination** of features:
> $$\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s) = \sum_{i=1}^{d} w_i \, x_i(s)$$
> 
> where $\mathbf{x}(s) = (x_1(s), \ldots, x_d(s))^\top$ is a **feature vector** and $\mathbf{w}$ is a weight vector.

## Gradient

The gradient is simply the feature vector:
$$\nabla_\mathbf{w} \hat{v}(s, \mathbf{w}) = \mathbf{x}(s)$$

This makes updates simple:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \delta_t \, \mathbf{x}(S_t)$$

## Convergence Guarantee

> [!tip] Why Linear Is Special
> [[Semi-Gradient Methods|Semi-gradient TD(0)]] with linear FA converges to the **[[TD Fixed Point]]**:
> $$\overline{VE}(\mathbf{w}_{TD}) \leq \frac{1}{1-\gamma} \min_\mathbf{w} \overline{VE}(\mathbf{w})$$
> 
> This guarantee does **not** hold for non-linear (e.g., neural network) approximators.

## Feature Construction

The power of linear FA depends entirely on the feature vector $\mathbf{x}(s)$. See [[Feature Construction]]:
- **[[Tile Coding]]**: Binary features from overlapping tilings
- **Polynomials**: $x_i(s) = s^i$
- **Fourier basis**: Cosine functions at different frequencies
- **Radial Basis Functions**: Gaussian bumps centered at prototypes
- **One-hot (tabular)**: Each state gets its own feature → recovers tabular case

## Connections

- Special case of: [[Function Approximation]]
- Solved exactly by: [[LSTD]]
- Feature design: [[Feature Construction]], [[Tile Coding]]
- Convergence: [[TD Fixed Point]]

## Appears In

- [[RL-L06 - On-Policy TD with Approximation]]
- [[RL-Book Ch9 - On-Policy Prediction with Approximation]]
