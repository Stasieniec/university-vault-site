---
type: concept
aliases: [GPU architecture, CUDA architecture, NVIDIA GPU architecture]
course: [IR]
tags: [gpu, efficiency]
status: complete
---

# GPU Architecture

> [!definition] GPU Architecture
> **GPU Architecture** refers to the hardware design of Graphics Processing Units, particularly NVIDIA GPUs used for deep learning. Modern GPUs feature a hierarchical structure of **Streaming Multiprocessors (SMs)** containing thousands of parallel processing cores, multiple memory levels with vastly different bandwidths, and specialized units for matrix operations.

> [!intuition] Massively Parallel, Memory-Constrained
> GPUs achieve high throughput by running thousands of threads simultaneously. However, the real bottleneck is often **memory bandwidth**—the GPU can compute faster than it can fetch data. Optimization means keeping data in fast memory (SRAM) and minimizing trips to slow memory (HBM).

## Key Components

### Streaming Multiprocessors (SMs)

> [!definition] Streaming Multiprocessor
> An **SM** is the fundamental compute unit on NVIDIA GPUs. Each SM contains:
> - **CUDA Cores**: General-purpose ALUs for scalar/vector operations (64-128 per SM)
> - **Tensor Cores**: Specialized matrix-multiply units for accelerated FP16/BF16 matmul (4-8 per SM)
> - **Shared Memory (SRAM)**: Fast, programmer-managed cache (~192KB per SM)
> - **Register File**: Per-thread ultra-fast storage

### Memory Hierarchy

| Memory Level | Capacity | Bandwidth | Latency |
|--------------|----------|-----------|---------|
| **Registers** | ~256KB per SM | ~20 TB/s | Fastest |
| **Shared Memory (SRAM)** | ~192KB per SM | ~15-19 TB/s | Very fast |
| **L2 Cache** | ~40MB | ~4-5 TB/s | Medium |
| **HBM (Global Memory)** | 40-80GB | ~1.5-2 TB/s | Slowest |

### HBM vs SRAM

> [!formula] Memory Bandwidth Gap
> $$\frac{\text{SRAM Bandwidth}}{\text{HBM Bandwidth}} \approx \frac{19 \text{ TB/s}}{2 \text{ TB/s}} \approx 10\times$$
>
> SRAM is roughly **10x faster** than HBM, making data locality critical for performance.

**HBM (High Bandwidth Memory)**:
- Large capacity (40-80GB)
- Slower access (~2 TB/s)
- Used for storing model weights, activations, gradients

**SRAM (Shared Memory)**:
- Small capacity (~192KB per SM)
- Much faster access (~19 TB/s)
- Programmer-managed, used for tiling and data reuse

## Memory Bandwidth Bottleneck

> [!intuition] The Real Limiting Factor
> For many neural network operations (elementwise ops, reductions, attention), the GPU spends more time **waiting for data** than actually computing. The key optimization strategy is maximizing **data reuse**—load once into SRAM, compute many operations, then write back.

### Arithmetic Intensity

> [!formula] Arithmetic Intensity
> $$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$$
>
> - **Low intensity** (e.g., elementwise ops): Memory-bound → optimize memory access
> - **High intensity** (e.g., large matmuls): Compute-bound → use Tensor Cores

### Bottleneck Types

| Bottleneck | Symptom | Solution |
|------------|---------|----------|
| **Compute-bound** | High SM utilization | Tensor Cores, better algorithms |
| **Memory-bound** | Saturated HBM bandwidth | [[Kernel Fusion]], tiling, data reuse |
| **Overhead-bound** | Many small kernels | Fuse kernels, reduce launches |

## Key Properties

- **Parallelism**: Thousands of threads execute simultaneously across SMs
- **SIMT Execution**: Single Instruction, Multiple Threads—threads in a warp execute the same instruction
- **Coalesced Memory Access**: Adjacent threads should access adjacent memory for efficiency
- **Occupancy**: Ratio of active warps to maximum warps per SM

## Connections

- [[Triton]] — Python-based language for writing efficient GPU kernels
- [[Kernel Fusion]] — Technique to reduce memory I/O by combining operations
- [[Sparton]] — Uses GPU architecture knowledge for efficient LSR training

## Appears In

- [[IR-L14 - Triton and Sparton]]
