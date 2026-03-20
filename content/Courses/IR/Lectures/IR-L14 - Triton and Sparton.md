---
type: lecture
course: IR
week: 6
lecture: 14
topics:
  - "[[GPU Architecture]]"
  - "[[Triton]]"
  - "[[Learned Sparse Retrieval]]"
  - "[[SPLADE]]"
  - "[[Kernel Fusion]]"
  - "[[Sparton]]"
status: complete
---

# IR-L14 - Triton and Sparton

## Overview

This lecture bridges **GPU programming** with **efficient neural information retrieval**. We first explore modern GPU architecture and OpenAI's Triton language for writing high-performance kernels, then apply these concepts to **Sparton**—a system that dramatically accelerates [[Learned Sparse Retrieval]] training by exploiting operator reordering and kernel fusion.

Key motivations:
- **Memory bottleneck**: Standard [[SPLADE]] implementations create massive intermediate tensors ($B \times S \times |V|$) that don't fit in GPU memory.
- **Compute inefficiency**: Naive implementations waste bandwidth moving data between HBM and compute units.
- **Solution**: Fuse operations into custom Triton kernels that keep data in fast SRAM and eliminate intermediate materialization.

---

## 1. GPU Architecture Fundamentals

Understanding GPU hardware is essential for writing efficient neural retrieval code.

### 1.1 Processing Elements

Modern NVIDIA GPUs (e.g., A100, H100) contain thousands of parallel processors organized hierarchically:

```
┌─────────────────────────────────────────────────────────────────┐
│                           GPU                                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │     SM 0     │ │     SM 1     │ │    SM ...    │  ...        │
│  │  ┌────────┐  │ │  ┌────────┐  │ │  ┌────────┐  │             │
│  │  │CUDA    │  │ │  │CUDA    │  │ │  │CUDA    │  │             │
│  │  │Cores   │  │ │  │Cores   │  │ │  │Cores   │  │             │
│  │  │(64-128)│  │ │  │(64-128)│  │ │  │(64-128)│  │             │
│  │  └────────┘  │ │  └────────┘  │ │  └────────┘  │             │
│  │  ┌────────┐  │ │  ┌────────┐  │ │  ┌────────┐  │             │
│  │  │Tensor  │  │ │  │Tensor  │  │ │  │Tensor  │  │             │
│  │  │Cores   │  │ │  │Cores   │  │ │  │Cores   │  │             │
│  │  │(4-8)   │  │ │  │(4-8)   │  │ │  │(4-8)   │  │             │
│  │  └────────┘  │ │  └────────┘  │ │  └────────┘  │             │
│  │  ┌────────┐  │ │  ┌────────┐  │ │  ┌────────┐  │             │
│  │  │Shared  │  │ │  │Shared  │  │ │  │Shared  │  │             │
│  │  │Memory  │  │ │  │Memory  │  │ │  │Memory  │  │             │
│  │  │(SRAM)  │  │ │  │(SRAM)  │  │ │  │(SRAM)  │  │             │
│  │  └────────┘  │ │  └────────┘  │ │  └────────┘  │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    HBM (Global Memory)                      │ │
│  │                      40-80 GB                               │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

> [!definition] Streaming Multiprocessor (SM)
> An SM is the fundamental compute unit on NVIDIA GPUs. Each SM contains:
> - **CUDA Cores**: General-purpose ALUs for scalar/vector operations
> - **Tensor Cores**: Specialized matrix-multiply units (e.g., 4x4 FP16 matmul)
> - **Shared Memory (SRAM)**: Fast, programmer-managed cache (~192KB per SM)
> - **Register File**: Per-thread ultra-fast storage

### 1.2 Memory Hierarchy

The critical insight for optimization is the massive speed difference between memory levels:

```
┌───────────────────────────────────────────────────────────────┐
│                    GPU Memory Hierarchy                        │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│   Registers     ████████████████████████  ~20 TB/s            │
│   (per thread)                            (fastest)           │
│                                                                │
│   Shared Mem    ████████████████████     ~15-19 TB/s          │
│   (SRAM)        (192KB per SM)           (very fast)          │
│                                                                │
│   L2 Cache      █████████████            ~4-5 TB/s            │
│                 (40MB)                                         │
│                                                                │
│   HBM           ████████                 ~1.5-2 TB/s          │
│   (Global)      (40-80GB)                (slowest)            │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

> [!intuition] Memory Bandwidth is the Bottleneck
> For many neural network operations, the GPU spends more time waiting for data from HBM than actually computing. The key to optimization is **data reuse**—load data once into SRAM, perform many operations, then write back.

### 1.3 Kernel Optimization: Three Bottleneck Types

| Bottleneck Type | Symptom | Solution |
|-----------------|---------|----------|
| **Compute-bound** | High SM utilization, low memory traffic | Use Tensor Cores, better algorithms |
| **Memory-bound** | Low SM utilization, saturated HBM bandwidth | Kernel fusion, tiling, data reuse |
| **Overhead-bound** | Many small kernels, CPU-GPU sync | Fuse kernels, reduce launch overhead |

> [!formula] Arithmetic Intensity
> $$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes transferred}}$$
>
> Operations with low arithmetic intensity (e.g., elementwise ops, reductions) are **memory-bound**. Matrix multiplication has high arithmetic intensity and is typically **compute-bound**.

---

## 2. Triton Programming Model

[[Triton]] is OpenAI's Python-like language for writing GPU kernels without low-level CUDA complexity.

### 2.1 Why Triton?

| Aspect | CUDA | Triton |
|--------|------|--------|
| **Abstraction** | Thread-level | Block-level |
| **Memory management** | Manual shared mem | Automatic tiling |
| **Learning curve** | Steep | Moderate |
| **Performance** | Maximum control | Near-CUDA with less effort |

### 2.2 Core Concepts

> [!definition] Triton Kernel
> A Triton kernel is a function decorated with `@triton.jit` that operates on **blocks** of data. Each kernel instance (called a "program") processes a tile of the input.

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles a contiguous block of elements
    pid = tl.program_id(axis=0)  # Which block am I?

    # Compute which elements this program handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask out-of-bounds elements
    mask = offsets < n_elements

    # Load, compute, store
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

### 2.3 Program IDs and Grid

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Tensor                             │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────┐  │
│  │ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ ... │  │
│  │ pid=0   │ pid=1   │ pid=2   │ pid=3   │ pid=4   │     │  │
│  └─────────┴─────────┴─────────┴─────────┴─────────┴─────┘  │
│                                                              │
│  Each program instance (pid) processes one block in parallel │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Launcher Function

The launcher configures the grid and calls the kernel:

```python
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()

    # Configure grid: how many programs to launch
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

---

## 3. Learned Sparse Retrieval: The Memory Problem

[[Learned Sparse Retrieval]] models like [[SPLADE]] face severe memory constraints during training.

### 3.1 LM Encoder Architecture

The [[SPLADE]] encoder produces term weights over the entire vocabulary:

```
┌─────────────────────────────────────────────────────────────┐
│                    SPLADE Architecture                       │
│                                                              │
│   Input Text: "information retrieval lecture"                │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Transformer Encoder                     │   │
│   │              (e.g., BERT, DistilBERT)                │   │
│   └─────────────────────────────────────────────────────┘   │
│        │                                                     │
│        ▼ H ∈ R^{B×S×D}                                       │
│        │ (Hidden states)                                     │
│        ▼                                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   LM Head                            │   │
│   │           (Linear: D → |V|)                          │   │
│   │                                                      │   │
│   │     Y = Max(Log(1 + ReLU(Mask(H·E^T + b))))         │   │
│   └─────────────────────────────────────────────────────┘   │
│        │                                                     │
│        ▼ y ∈ R^{B×|V|}                                       │
│        │ (Sparse term weights)                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 The LM Head Formulation

> [!formula] SPLADE LM Head (Standard)
> $$Y = \text{Max}_{i \in S}\left(\log(1 + \text{ReLU}(\text{Mask}(H E^T + b)))\right)$$
>
> where:
> - $H \in \mathbb{R}^{B \times S \times D}$ — hidden states (batch × sequence × hidden dim)
> - $E \in \mathbb{R}^{|V| \times D}$ — vocabulary embeddings
> - $b \in \mathbb{R}^{|V|}$ — bias vector
> - $\text{Mask}$ — zeros out special tokens
> - $\text{Max}$ — aggregates across sequence dimension

### 3.3 The Memory Bottleneck

The intermediate tensor after the linear projection is **massive**:

> [!warning] Memory Explosion
> **Intermediate tensor size**: $B \times S \times |V|$
>
> For typical values:
> - $B = 32$ (batch size)
> - $S = 256$ (sequence length)
> - $|V| = 30522$ (BERT vocabulary)
>
> $$\text{Memory} = 32 \times 256 \times 30522 \times 4 \text{ bytes} \approx \textbf{1 GB}$$
>
> This single intermediate tensor consumes most GPU memory, severely limiting batch sizes and throughput.

```
┌─────────────────────────────────────────────────────────────┐
│              Memory Breakdown (Naive SPLADE)                 │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  Intermediate: H·E^T + b                              │  │
│   │  Shape: [B × S × |V|] = [32 × 256 × 30522]           │  │
│   │  Size: ~1 GB (FP32)                                   │  │
│   └──────────────────────────────────────────────────────┘  │
│                      ▼ (kept for backward pass)             │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  ReLU output: same shape                              │  │
│   │  Size: ~1 GB                                          │  │
│   └──────────────────────────────────────────────────────┘  │
│                      ▼ (kept for backward pass)             │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  Log1p output: same shape                             │  │
│   │  Size: ~1 GB                                          │  │
│   └──────────────────────────────────────────────────────┘  │
│                      ▼                                       │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  Max output: [B × |V|]                                │  │
│   │  Size: ~4 MB (much smaller!)                          │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                              │
│   Total naive memory: ~3 GB for intermediates alone!        │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Sparton: Fast and Memory-Efficient LSR

**Sparton** solves the memory and compute bottleneck through four key innovations:

### 4.1 Innovation 1: Operator Reordering

> [!intuition] Key Insight: Max Commutes with Monotonic Functions
> The standard order is: MatMul → Mask → ReLU → Log1p → Max
>
> Since ReLU and Log1p are **monotonically non-decreasing**, we can move Max earlier:
> - $\text{Max}(\log(1 + x)) = \log(1 + \text{Max}(x))$ when $x \geq 0$
> - $\text{Max}(\text{ReLU}(x)) = \text{ReLU}(\text{Max}(x))$

> [!formula] Sparton Reordered LM Head
> $$Y = \log\left(1 + \text{ReLU}\left(\text{Max}_{i \in S}(\text{Mask}(H E^T + b))\right)\right)$$
>
> The Max now operates **inside** the elementwise operations, reducing the tensor from $B \times S \times |V|$ to $B \times |V|$ **before** applying ReLU and Log1p.

```
┌─────────────────────────────────────────────────────────────┐
│           Standard vs. Sparton Operation Order               │
│                                                              │
│   Standard SPLADE:                                           │
│   H·E^T → Mask → ReLU → Log1p → Max                         │
│   [B×S×|V|]     [B×S×|V|] [B×S×|V|]  [B×|V|]                │
│   ▲              ▲         ▲                                 │
│   │              │         │                                 │
│   └──────────────┴─────────┴── All intermediate tensors     │
│                                 must be stored!              │
│                                                              │
│   Sparton (Reordered):                                       │
│   H·E^T → Mask → Max → ReLU → Log1p                         │
│   [B×S×|V|]     [B×|V|] [B×|V|] [B×|V|]                     │
│   ▲              │       │       │                           │
│   │              └───────┴───────┴── Much smaller!          │
│   └── Can be computed tile-by-tile (never fully stored)     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Innovation 2: Online Reduction / Tiled MatMul

Instead of computing the full $B \times S \times |V|$ tensor, Sparton uses **tiled computation** with **online reduction**:

```
┌─────────────────────────────────────────────────────────────┐
│                  Tiled MatMul with Online Max                │
│                                                              │
│   Hidden States H           Embedding Matrix E^T             │
│   [B × S × D]               [D × |V|]                        │
│                                                              │
│   ┌───────────────┐         ┌─────────────────────────────┐ │
│   │               │         │ Tile 0 │ Tile 1 │ Tile 2 │...│ │
│   │    Full H     │    ×    │[D×T]   │[D×T]   │[D×T]   │   │ │
│   │               │         └─────────────────────────────┘ │
│   └───────────────┘                                          │
│                                                              │
│   For each vocabulary tile (size T):                         │
│     1. Compute partial: P = H × E^T[tile]  → [B × S × T]    │
│     2. Apply mask                                            │
│     3. Update running max: max_acc = max(max_acc, P)        │
│     4. Discard P (don't store!)                              │
│                                                              │
│   Result: Only store max_acc [B × |V|], never full [B×S×|V|]│
└─────────────────────────────────────────────────────────────┘
```

> [!definition] Online Reduction
> An **online algorithm** processes data incrementally without storing the full intermediate result. For max reduction:
> ```
> max_accumulator = -inf
> for tile in tiles:
>     partial_result = compute(tile)
>     max_accumulator = max(max_accumulator, partial_result)
> ```

### 4.3 Innovation 3: Sparse Gradient Computation

During backpropagation, we only need gradients for positions where the max was achieved:

> [!formula] Sparse Gradient
> $$\frac{\partial Y_j}{\partial (HE^T)_{i,j}} = \begin{cases}
> \frac{1}{1 + Y_j} & \text{if } i = \text{argmax}_k (HE^T)_{k,j} \text{ and } Y_j > 0 \\
> 0 & \text{otherwise}
> \end{cases}$$
>
> Only ~$B \times |V|$ gradient values are non-zero (vs. $B \times S \times |V|$ in naive implementation).

```
┌─────────────────────────────────────────────────────────────┐
│              Sparse Gradient Computation                     │
│                                                              │
│   Forward pass stores:                                       │
│   - argmax indices: [B × |V|] (which sequence position won) │
│   - max values: [B × |V|]                                    │
│                                                              │
│   Backward pass:                                             │
│   - Only compute gradients at argmax positions               │
│   - Skip all other positions (gradient = 0)                  │
│                                                              │
│   Memory: O(B × |V|) instead of O(B × S × |V|)              │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 Innovation 4: Fully Fused Forward Kernel

All operations are fused into a single Triton kernel:

```
┌─────────────────────────────────────────────────────────────┐
│              Fused Sparton Forward Kernel                    │
│                                                              │
│   @triton.jit                                                │
│   def sparton_forward(...):                                  │
│       # Load tile of H into SRAM                             │
│       h_tile = tl.load(H_ptr + offsets)                     │
│                                                              │
│       # Initialize accumulators in registers                 │
│       max_val = -INF                                         │
│       max_idx = 0                                            │
│                                                              │
│       # Loop over vocabulary tiles                           │
│       for v_start in range(0, V, V_TILE):                   │
│           # Load embedding tile                              │
│           e_tile = tl.load(E_ptr + v_offsets)               │
│                                                              │
│           # Compute partial matmul in SRAM                   │
│           partial = tl.dot(h_tile, e_tile) + bias           │
│                                                              │
│           # Apply mask                                       │
│           partial = tl.where(mask, partial, -INF)           │
│                                                              │
│           # Online max reduction                             │
│           new_max = tl.max(partial, axis=0)                 │
│           update_mask = new_max > max_val                    │
│           max_val = tl.where(update_mask, new_max, max_val) │
│           max_idx = tl.where(update_mask, seq_idx, max_idx) │
│                                                              │
│       # Apply ReLU and Log1p ONCE at the end                │
│       output = tl.log(1 + tl.maximum(max_val, 0))           │
│                                                              │
│       # Store final result                                   │
│       tl.store(out_ptr, output)                              │
│       tl.store(idx_ptr, max_idx)                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

> [!tip] Benefits of Fusion
> 1. **No intermediate tensor**: $B \times S \times |V|$ is never materialized
> 2. **Data reuse**: H stays in SRAM while iterating over vocabulary tiles
> 3. **Reduced kernel launches**: One kernel instead of 5+ separate operations
> 4. **Better memory bandwidth**: Only read H once, only write final output

---

## 5. Experimental Evaluation

### 5.1 Runtime and Memory: LM Head Overhead (SPLADE V3, B=320, S=512, |V|=30522)

| Phase | Component | Eager Time (ms) | Eager Mem (MiB) | Compiled Time (ms) | Compiled Mem (MiB) |
|-------|-----------|-----------------|-----------------|--------------------|--------------------|
| Fwd | Backbone | 84.4 | 2893.7 | 99.7 | 3083.6 |
| Fwd | Backbone + LM Head | 162.1 | 28885.1 | 122.1 | 10126.0 |
| Fwd | Backbone + **Sparton** | **113.7** | **2955.4** | 129.0 | 3146.0 |
| Fwd+Bwd | Backbone | 293.0 | 50942.8 | 387.0 | 50218.1 |
| Fwd+Bwd | Backbone + LM Head | 498.1 | 88875.0 | 473.0 | 70007.2 |
| Fwd+Bwd | Backbone + **Sparton** | **330.1** | **51651.2** | 423.9 | 51349.8 |

> [!tip] Key Observations
> - LM Head alone adds **>40% runtime** and **+30 GB memory**
> - Sparton **almost completely removes the memory overhead** of the LM Head
> - PyTorch compilation does not help (actually slows down execution)

### 5.2 End-to-End Training (English checkpoint, splade-cocondenser, |V|=30k)

| Method | Batch | Steps | Time (h) | Mem (GB) | nDCG@10 |
|--------|-------|-------|----------|----------|---------|
| Splade-V3 | — | — | — | — | 0.422 |
| Compiled LM | 384 | 67528 | 14.24 | 125.78 | 0.421 |
| Sparton | 384 | 67528 | 12.38 | 96.83 | 0.416 |
| **Sparton** | **512** | **50646** | **12.24** | **128.63** | **0.427** |

- **30% larger batch size** (384 → 512)
- **14% faster training**
- nDCG@10 **matches Splade-V3** (accuracy preserved)

### 5.3 Multilingual Training (xlm-roberta-base, |V|=250k)

For multilingual models with large vocabularies, the impact is even more dramatic:
- **26x larger batch size** (16 → 420)
- **2.5x faster training**

### 5.4 Micro-benchmark Summary

From the Sparton paper's scaling evaluation:
- Up to **4.8x faster** than PyTorch
- Up to **10x+ peak memory reduction**
- Impact **increases with input size** (batch, sequence length, vocabulary)

---

## 6. Summary: Optimization Techniques

```
┌─────────────────────────────────────────────────────────────┐
│                 Sparton Optimization Stack                   │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Level 4: Sparse Gradients                           │   │
│   │  Only compute non-zero gradient positions            │   │
│   └─────────────────────────────────────────────────────┘   │
│                          ▲                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Level 3: Operator Reordering                        │   │
│   │  Move Max before ReLU/Log1p (monotonicity)          │   │
│   └─────────────────────────────────────────────────────┘   │
│                          ▲                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Level 2: Online Reduction                           │   │
│   │  Compute max incrementally, never store full tensor │   │
│   └─────────────────────────────────────────────────────┘   │
│                          ▲                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Level 1: Kernel Fusion (Triton)                     │   │
│   │  Single kernel, data stays in SRAM                  │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **GPU Memory Hierarchy**: Understanding the HBM vs. SRAM bandwidth gap (10x) is essential for optimization. Keep data in fast SRAM as long as possible.

2. **Triton for Research**: Triton provides a Python-like interface for writing efficient GPU kernels, making hardware-aware optimization accessible to ML researchers.

3. **LSR Memory Bottleneck**: The $B \times S \times |V|$ intermediate tensor in [[SPLADE]] is the primary memory constraint, limiting batch sizes and throughput.

4. **Operator Reordering**: Moving Max before ReLU and Log1p is mathematically equivalent (monotonicity) but reduces memory from $O(B \times S \times |V|)$ to $O(B \times |V|)$.

5. **Online Algorithms**: Computing reductions incrementally (tiled matmul with online max) avoids materializing large intermediate tensors.

6. **Kernel Fusion**: Combining multiple operations into a single GPU kernel eliminates memory round-trips and launch overhead, providing 4-5x speedups.

7. **Practical Impact**: Sparton makes [[Learned Sparse Retrieval]] training feasible on consumer hardware, accelerating research iteration.

---

**Related Concepts:**
- [[Learned Sparse Retrieval]]
- [[SPLADE]]
- [[Inverted Index]]
- [[Dense Retrieval]]
- [[GPU Architecture]]
- [[Kernel Fusion]]
