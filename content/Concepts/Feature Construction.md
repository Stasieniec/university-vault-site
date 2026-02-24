---
type: concept
aliases: [feature construction, feature engineering, basis functions]
course: [RL]
tags: [approximation]
status: complete
---

# Feature Construction

> [!definition] Feature Construction
> The design of feature vectors $\mathbf{x}(s)$ for [[Linear Function Approximation]]. The choice of features determines what the approximator can represent — linear FA is only as good as its features.

## Methods

| Method | Features | Properties |
|--------|----------|------------|
| **Polynomials** | $x_i(s) = s^i$ | Simple, global, poor scaling |
| **Fourier Basis** | $x_i(s) = \cos(i\pi s)$ | Good for smooth functions, global |
| **Coarse Coding** | Binary: overlapping receptive fields | Local generalization |
| **[[Tile Coding]]** | Binary: multiple offset grids | Fast, local, popular in RL |
| **RBF** | $x_i(s) = \exp(-\|s - c_i\|^2 / 2\sigma_i^2)$ | Smooth, local, continuous-valued |
| **One-hot** | $x_s = 1$, rest 0 | Tabular (no generalization) |

> [!tip] Key Insight
> With linear FA, you can't learn features — you have to design them. The move to [[Neural Network Function Approximation]] automates feature learning, which is one of deep RL's main advantages.

## Appears In

- [[RL-L06 - On-Policy TD with Approximation]], [[RL-Book Ch9 - On-Policy Prediction with Approximation]]
