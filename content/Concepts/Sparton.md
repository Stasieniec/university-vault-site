---
type: concept
aliases: [Sparton, sparse triton, sparton kernel]
course: [IR]
tags: [gpu, efficiency]
status: complete
---

# Sparton

> [!definition] Sparton
> **Sparton** (Sparse Triton) is a highly optimized implementation of the [[SPLADE]] LM head using [[Triton]] kernels. It achieves ~5x speedup and ~4x memory reduction by combining **operator reordering**, **online reduction**, **tiled matrix multiplication**, and **sparse gradient computation** to eliminate the massive $B \times S \times |V|$ intermediate tensor.

> [!intuition] Never Build What You Don't Need
> The key insight is that [[SPLADE]] computes a huge intermediate tensor only to immediately reduce it with Max. Sparton reorders operations so the Max happens *during* computation, tile by tile, never materializing the full tensor. It's like computing a running maximum while streaming through data instead of storing everything first.

## The Problem: SPLADE Memory Bottleneck

> [!formula] Standard SPLADE LM Head
> $$Y = \text{Max}_{i \in S}\left(\log(1 + \text{ReLU}(\text{Mask}(H E^T + b)))\right)$$
>
> where:
> - $H \in \mathbb{R}^{B \times S \times D}$ — hidden states
> - $E \in \mathbb{R}^{|V| \times D}$ — vocabulary embeddings
> - The intermediate tensor $HE^T \in \mathbb{R}^{B \times S \times |V|}$ is **massive**

**Memory explosion example**:
- $B = 32$, $S = 256$, $|V| = 30522$
- Intermediate size: $32 \times 256 \times 30522 \times 4$ bytes $\approx$ **1 GB**

## Innovation 1: Operator Reordering

> [!formula] Sparton Reordered LM Head
> $$Y = \log\left(1 + \text{ReLU}\left(\text{Max}_{i \in S}(\text{Mask}(H E^T + b))\right)\right)$$
>
> The Max moves **inside** the monotonic functions (ReLU, Log1p), reducing tensor size from $B \times S \times |V|$ to $B \times |V|$ **before** applying elementwise operations.

> [!intuition] Why This Works
> For monotonically non-decreasing functions $f$:
> $$\text{Max}(f(x)) = f(\text{Max}(x))$$
> Since ReLU and $\log(1+x)$ are monotonic for $x \geq 0$, we can swap the order.

```
Standard:  MatMul → Mask → ReLU → Log1p → Max
           [B×S×|V|]      [B×S×|V|]      [B×|V|]
                   ↑ All stored in memory

Sparton:   MatMul → Mask → Max → ReLU → Log1p
           [B×S×|V|]     [B×|V|]  (small!)
                   ↑ Never fully materialized
```

## Innovation 2: Online Reduction / Tiled MatMul

Instead of computing the full $B \times S \times |V|$ tensor, Sparton processes vocabulary **tiles** with a running maximum:

```
for each vocabulary tile (size T):
    1. Compute partial: P = H × E^T[tile]  → [B × S × T]
    2. Apply mask
    3. Update running max: max_acc = max(max_acc, P.max(dim=S))
    4. Discard P (don't store!)

Result: Only store max_acc [B × |V|], never full [B × S × |V|]
```

> [!definition] Online Reduction
> An **online algorithm** processes data incrementally without storing all intermediate results. For max: maintain a running maximum, updating it as each tile is processed.

## Innovation 3: Sparse Gradient Computation

During backpropagation, gradients are non-zero only where the max was achieved:

> [!formula] Sparse Gradient
> $$\frac{\partial Y_j}{\partial (HE^T)_{i,j}} = \begin{cases}
> \frac{1}{1 + Y_j} & \text{if } i = \text{argmax}_k (HE^T)_{k,j} \text{ and } Y_j > 0 \\
> 0 & \text{otherwise}
> \end{cases}$$
>
> Only $B \times |V|$ positions have non-zero gradients (vs. $B \times S \times |V|$ naively).

The forward pass stores:
- **argmax indices**: $[B \times |V|]$ — which sequence position achieved the max
- **max values**: $[B \times |V|]$ — the actual maximum values

## Innovation 4: Fused Triton Kernel

All operations combined into a single kernel:

```python
@triton.jit
def sparton_forward(...):
    # Load H tile into SRAM
    h_tile = tl.load(H_ptr + offsets)

    # Initialize accumulators
    max_val = -INF
    max_idx = 0

    # Loop over vocabulary tiles
    for v_start in range(0, V, V_TILE):
        e_tile = tl.load(E_ptr + v_offsets)
        partial = tl.dot(h_tile, e_tile) + bias
        partial = tl.where(mask, partial, -INF)

        # Online max reduction
        new_max = tl.max(partial, axis=0)
        update_mask = new_max > max_val
        max_val = tl.where(update_mask, new_max, max_val)
        max_idx = tl.where(update_mask, seq_idx, max_idx)

    # Apply ReLU and Log1p ONCE
    output = tl.log(1 + tl.maximum(max_val, 0))
    tl.store(out_ptr, output)
    tl.store(idx_ptr, max_idx)
```

## Performance Results (SPLADE V3, B=320, S=512, |V|=30522)

| Phase | Component | Eager Time (ms) | Eager Mem (MiB) |
|-------|-----------|-----------------|-----------------|
| Fwd | Backbone + LM Head | 162.1 | 28885.1 |
| Fwd | Backbone + **Sparton** | **113.7** | **2955.4** |
| Fwd+Bwd | Backbone + LM Head | 498.1 | 88875.0 |
| Fwd+Bwd | Backbone + **Sparton** | **330.1** | **51651.2** |

Sparton almost completely removes the memory overhead of the LM Head. Micro-benchmarks show up to **4.8x faster** and **10x+ peak memory reduction**.

## Key Properties

- **Memory Efficient**: $O(B \times |V|)$ instead of $O(B \times S \times |V|)$
- **Fast**: ~5x speedup from [[Kernel Fusion]] and reduced memory traffic
- **Mathematically Equivalent**: Produces identical outputs to standard SPLADE
- **Backward Compatible**: Drop-in replacement for SPLADE training

## Connections

- [[SPLADE]] — The model Sparton accelerates
- [[Learned Sparse Retrieval]] — The broader family of efficient neural retrievers
- [[Triton]] — The GPU programming framework used
- [[Kernel Fusion]] — Core optimization technique
- [[GPU Architecture]] — Understanding of memory hierarchy enables these optimizations

## Appears In

- [[IR-L14 - Triton and Sparton]]
