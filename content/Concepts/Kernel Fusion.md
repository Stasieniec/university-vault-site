---
type: concept
aliases: [kernel fusion, operator fusion, fused kernels]
course: [IR]
tags: [gpu, efficiency]
status: complete
---

# Kernel Fusion

> [!definition] Kernel Fusion
> **Kernel Fusion** is a GPU optimization technique that combines multiple sequential operations into a single kernel, eliminating intermediate tensor materialization and reducing memory I/O. Instead of writing results to slow HBM memory between operations, fused kernels keep intermediate data in fast SRAM.

> [!intuition] Avoiding the Memory Round-Trip
> Imagine computing `y = log(relu(x + b))` naively: each operation reads from and writes to slow GPU memory. With fusion, we load `x` and `b` once, compute everything in fast cache, and write only `y`. The speedup comes from avoiding the "memory tax" on each operation.

## The Problem: Intermediate Tensor Materialization

```
┌─────────────────────────────────────────────────────────────┐
│                 Naive (Unfused) Execution                    │
│                                                              │
│   Operation 1: z = x + b                                     │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐               │
│   │  HBM    │ ──► │ Compute │ ──► │  HBM    │  (write z)    │
│   │ (read x,b)    │  x + b  │     │ (store z)│               │
│   └─────────┘     └─────────┘     └─────────┘               │
│                                        │                     │
│   Operation 2: a = relu(z)             ▼                     │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐               │
│   │  HBM    │ ──► │ Compute │ ──► │  HBM    │  (write a)    │
│   │ (read z)│     │ relu(z) │     │ (store a)│               │
│   └─────────┘     └─────────┘     └─────────┘               │
│                                        │                     │
│   Operation 3: y = log(a)              ▼                     │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐               │
│   │  HBM    │ ──► │ Compute │ ──► │  HBM    │  (write y)    │
│   │ (read a)│     │ log(a)  │     │ (store y)│               │
│   └─────────┘     └─────────┘     └─────────┘               │
│                                                              │
│   Total HBM transfers: 6 (read x, b, z, a; write z, a, y)   │
└─────────────────────────────────────────────────────────────┘
```

## The Solution: Fused Kernel

```
┌─────────────────────────────────────────────────────────────┐
│                   Fused Execution                            │
│                                                              │
│   ┌─────────┐     ┌─────────────────────────┐  ┌─────────┐ │
│   │  HBM    │ ──► │        SRAM             │  │  HBM    │ │
│   │(read x,b)     │  z = x + b              │  │(write y)│ │
│   └─────────┘     │  a = relu(z) [in SRAM]  │  └─────────┘ │
│                   │  y = log(a)  [in SRAM]  │──►           │
│                   └─────────────────────────┘               │
│                                                              │
│   Total HBM transfers: 3 (read x, b; write y)               │
│   Speedup: ~2x fewer memory operations!                      │
└─────────────────────────────────────────────────────────────┘
```

## Mathematical Formulation

> [!formula] Memory Savings from Fusion
> For a sequence of $n$ elementwise operations on tensor of size $M$:
>
> **Unfused**: $2nM$ bytes transferred (read + write per operation)
>
> **Fused**: $2M$ bytes transferred (read input + write output)
>
> $$\text{Reduction Factor} = \frac{2nM}{2M} = n$$

## Key Properties

- **Eliminates Intermediate Storage**: Tensors between operations never hit slow HBM
- **Reduces Kernel Launch Overhead**: One launch instead of many
- **Improves Cache Utilization**: Data stays in fast SRAM/registers
- **Enables Online Algorithms**: Reductions can be computed incrementally

## Common Fusion Patterns

| Pattern | Operations | Example |
|---------|------------|---------|
| **Activation Fusion** | Linear + Activation | `relu(Wx + b)` |
| **Normalization Fusion** | Stats + Normalize | LayerNorm, BatchNorm |
| **Attention Fusion** | QK^T + Softmax + V | FlashAttention |
| **Reduction Fusion** | Matmul + Max/Sum | [[Sparton]] |

## Variants

### Manual Fusion (Triton/CUDA)
Write custom kernels that combine operations explicitly. Maximum control but requires GPU programming expertise.

### Compiler Fusion (XLA, TorchScript)
Automatic fusion by ML compilers. Less effort but may miss optimization opportunities.

### Framework-Level Fusion
Libraries like FlashAttention provide pre-fused implementations of common patterns.

## Example: SPLADE LM Head

> [!formula] Naive vs. Fused SPLADE
> **Naive**: Separate kernels for each operation
> $$\text{MatMul} \rightarrow \text{Mask} \rightarrow \text{ReLU} \rightarrow \text{Log1p} \rightarrow \text{Max}$$
> Memory: $O(B \times S \times |V|)$ intermediates
>
> **Fused ([[Sparton]])**: Single kernel with online reduction
> $$\text{Fused}(\text{MatMul}, \text{Mask}, \text{Max}) \rightarrow \text{ReLU} \rightarrow \text{Log1p}$$
> Memory: $O(B \times |V|)$ — no large intermediates

## Connections

- [[GPU Architecture]] — Understanding memory hierarchy motivates fusion
- [[Triton]] — Tool for writing custom fused kernels
- [[Sparton]] — Application of fusion to [[Learned Sparse Retrieval]]
- [[SPLADE]] — Model that benefits from kernel fusion

## Appears In

- [[IR-L14 - Triton and Sparton]]
