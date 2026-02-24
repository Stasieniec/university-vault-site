---
type: concept
aliases: [tile coding, tilings]
course: [RL]
tags: [approximation, exam-topic]
status: complete
---

# Tile Coding

> [!definition] Tile Coding
> **Tile coding** is a [[Feature Construction]] method for [[Linear Function Approximation]]. It partitions the state space into **tiles** using multiple overlapping grids (**tilings**), each offset from the others. Each tiling produces a binary feature: 1 for the tile containing the current state, 0 for all others.

## How It Works

1. Define $n$ tilings over the state space, each a grid of tiles
2. Each tiling is offset from the others by a fraction of the tile width
3. For a given state $s$, exactly **one tile per tiling** is active → $n$ active features out of total
4. Feature vector $\mathbf{x}(s)$ is binary with exactly $n$ ones

```
Tiling 1:  |  A  |  B  |  C  |  D  |
Tiling 2:    |  E  |  F  |  G  |  H  |    (offset by half tile width)
State s: --------*----
Active tiles: B (tiling 1), F (tiling 2)
→ x(s) = [0,1,0,0, 0,1,0,0]
```

## Value Approximation

$$\hat{v}(s, \mathbf{w}) = \sum_{\text{active tiles } i} w_i$$

Since features are binary, the value is just the **sum of weights of active tiles**. Update:
$$w_i \leftarrow w_i + \frac{\alpha}{n} \delta_t \quad \text{for each active tile } i$$

(The $1/n$ factor accounts for $n$ tilings contributing.)

> [!intuition] Why Multiple Tilings?
> A single coarse tiling gives poor resolution. Multiple offset tilings create **receptive fields** that overlap differently, providing finer discrimination. More tilings = smoother approximation (at the cost of more features).

## Properties

- ✅ Simple, fast (binary features → sum of weights)
- ✅ Local generalization (nearby states share tiles)
- ✅ Controllable resolution (tile size, number of tilings)
- ❌ Curse of dimensionality: tiles scale exponentially with state dimensions
- Hashing can compress the feature space when tiles are sparse

## Appears In

- [[RL-L06 - On-Policy TD with Approximation]]
- [[RL-Book Ch9 - On-Policy Prediction with Approximation]] (§9.5.4)
