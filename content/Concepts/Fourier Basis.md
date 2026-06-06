---
type: concept
aliases: [Fourier Features]
course: [RL]
tags: [approximation, key-formula, exam-topic]
status: complete
---

# Fourier Basis

## Definition

> [!definition] Fourier Basis
> The **Fourier basis** is a family of feature functions for [[Linear Function Approximation]] that represents the value function as a weighted sum of **cosines of different frequencies**. For a state $s \in [0,1]^k$ (normalized), each feature is $x_i(s) = \cos(\pi\, \mathbf{s}^\top \mathbf{c}_i)$, where the integer vector $\mathbf{c}_i$ selects the frequency and the interactions among the $k$ state dimensions. It comes from the Fourier series: any sufficiently smooth periodic function can be reconstructed as a linear combination of sinusoids, so the learner only needs to fit the weights $\mathbf{w}$ on a fixed set of cosine features.

## Intuition

A value function over a continuous state is just some smooth curve (or surface). A Fourier series says **any** such function can be built by stacking cosines of increasing frequency: low-frequency terms capture the broad shape, high-frequency terms capture the fine detail. Instead of learning *what* features to use (as a neural net would), you **fix** the set of cosine waves and only learn how much of each to add — that is the weight vector $\mathbf{w}$.

Why cosines and not sines? On a bounded interval we only care about the function on $[0,1]$, not its periodic extension. Using **only cosines** lets us treat the function as if it were *even* (mirrored about $0$), which avoids forcing discontinuities at the boundaries and means we don't have to model the function's behaviour outside the region of interest.

The frequency vector $\mathbf{c}_i$ is the key design knob: a component $c_{i,j}=0$ means feature $i$ ignores state dimension $j$, a large $c_{i,j}$ means feature $i$ oscillates rapidly along dimension $j$, and **multiple** nonzero components let a single feature model **interactions** between dimensions.

## Mathematical Formulation

For a $k$-dimensional normalized state $\mathbf{s} = (s_1, \ldots, s_k)^\top$ with each $s_j \in [0,1]$, the **order-$n$ Fourier cosine basis** has features:

$$x_i(\mathbf{s}) = \cos\!\left(\pi\, \mathbf{s}^\top \mathbf{c}_i\right), \qquad \mathbf{c}_i = (c_{i,1}, \ldots, c_{i,k})^\top, \quad c_{i,j} \in \{0, 1, \ldots, n\}$$

The value estimate is then linear in these features:

$$\hat{v}(\mathbf{s}, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(\mathbf{s}) = \sum_{i} w_i \cos\!\left(\pi\, \mathbf{s}^\top \mathbf{c}_i\right)$$

where:
- $\mathbf{s} \in [0,1]^k$ — the state, with each dimension **normalized** to $[0,1]$
- $\mathbf{c}_i$ — integer frequency vector for feature $i$; its $j$-th entry sets the oscillation rate along dimension $j$
- $n$ — the **order** of the basis; allowing each $c_{i,j} \in \{0,\dots,n\}$ gives $(n+1)^k$ features in total
- $\pi$ — scaling that maps the $[0,1]$ domain onto one half-period of the cosine
- $w_i$ — learned weight on feature $i$ (the only thing trained)

Since the approximator is linear, the gradient is just the feature vector, $\nabla_\mathbf{w} \hat{v}(\mathbf{s},\mathbf{w}) = \mathbf{x}(\mathbf{s})$, so the standard linear update applies:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\, \delta_t\, \mathbf{x}(S_t)$$

where $\delta_t$ is the [[TD Error]].

## Key Properties / Variants

- **Global features**: each cosine spans the whole state space, so the Fourier basis generalizes *globally* — unlike [[Tile Coding]] and coarse/RBF coding, which generalize *locally*. One update changes the estimate everywhere.
- **Order vs. dimensionality**: an order-$n$ basis over $k$ dimensions needs $(n+1)^k$ features — it grows **exponentially** in the number of state dimensions, so it is practical only for low-dimensional states.
- **Per-feature step sizes**: high-frequency features oscillate faster, so they benefit from *smaller* learning rates. A common trick is to scale each feature's step size by $\alpha_i = \alpha / \lVert \mathbf{c}_i \rVert$ (with $\alpha_0 = \alpha$ for the constant feature) to keep learning stable across frequencies.
- **Outperforms polynomials**: empirically the Fourier basis tends to learn faster and more accurately than the polynomial basis ($x_i(s)=s^i$) on RL prediction tasks, while being just as simple to set up.
- **Cosine-only**: using only cosines (not sines) suits aperiodic functions on a bounded interval by implicitly assuming an even extension, avoiding boundary discontinuities.
- **Coupling vs. decoupling**: setting only one nonzero entry in each $\mathbf{c}_i$ (a "decoupled" basis) ignores dimension interactions and keeps the feature count linear in $k$; allowing multiple nonzero entries (the "full" basis) models interactions at exponential cost.

Sketch of constructing the feature vector:

```pseudo
Build order-n Fourier feature vector x(s) for state s in [0,1]^k
──────────────────────────────────────────────────────────────
Precompute frequency set C = { c : c in {0,...,n}^k }   # (n+1)^k vectors
  (decoupled variant: keep only c with at most one nonzero entry)

function FEATURES(s):
    for each c_i in C:
        x_i ← cos( π · (s · c_i) )      # dot product s·c_i, then cosine
    return vector x = (x_0, x_1, ..., x_{d-1})

# Use x(s) inside any linear semi-gradient method:
#   v̂(s,w) = wᵀ x(s)
#   w ← w + α · δ · x(s)        # δ = TD error
```

## Connections

- Type of: [[Feature Construction]] for [[Linear Function Approximation]]
- Alternative features: [[Tile Coding]] (local, binary), polynomials, radial basis functions
- Used inside: [[Semi-Gradient Methods]] / semi-gradient TD, with step size driven by the [[TD Error]]
- Contrast: [[Neural Network Function Approximation]] *learns* features instead of fixing them
- Objective minimized: [[Mean Squared Value Error (MSVE)]]

## Appears In

- [[RL-Book Ch9 - On-Policy Prediction with Approximation]]
- [[RL-L06 - On-Policy TD with Approximation]]
