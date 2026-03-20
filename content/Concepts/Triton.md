---
type: concept
aliases: [Triton, OpenAI Triton, triton-lang]
course: [IR]
tags: [gpu, efficiency]
status: complete
---

# Triton

> [!definition] Triton
> **Triton** is OpenAI's Python-based domain-specific language for writing high-performance GPU kernels. It operates at the **block level** (rather than thread level like CUDA), providing automatic memory management and tiling while achieving near-CUDA performance with significantly less complexity.

> [!intuition] CUDA Power, Python Simplicity
> Triton abstracts away low-level GPU details (thread synchronization, shared memory allocation) while giving researchers enough control to write efficient custom kernels. Think of it as "PyTorch-level syntax with CUDA-level performance."

## Core Concepts

### Kernel and Launcher

A Triton program consists of two parts:

**1. Kernel** (`@triton.jit`): The GPU function that processes data blocks

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  # Which block am I?
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

**2. Launcher**: Python function that configures the grid and calls the kernel

```python
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### Block-Level Parallelism

> [!definition] Block-Level Programming
> Unlike CUDA where you manage individual threads, Triton operates on **blocks** (tiles) of data. Each kernel instance (called a **program**) processes one block, and Triton handles thread-level details automatically.

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Tensor                             │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┐       │
│  │ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ ...   │
│  │ pid=0   │ pid=1   │ pid=2   │ pid=3   │ pid=4   │       │
│  └─────────┴─────────┴─────────┴─────────┴─────────┘       │
│                                                              │
│  Each program instance processes one block in parallel       │
└─────────────────────────────────────────────────────────────┘
```

### Key Primitives

| Primitive | Description |
|-----------|-------------|
| `tl.program_id(axis)` | Get the program's block index |
| `tl.arange(start, end)` | Create a range of indices |
| `tl.load(ptr, mask)` | Load data from memory |
| `tl.store(ptr, val, mask)` | Store data to memory |
| `tl.dot(a, b)` | Block matrix multiplication |
| `tl.max(x, axis)` | Reduction operation |
| `tl.where(cond, x, y)` | Conditional selection |

## Comparison with CUDA

| Aspect | CUDA | Triton |
|--------|------|--------|
| **Abstraction** | Thread-level | Block-level |
| **Memory management** | Manual shared memory | Automatic tiling |
| **Learning curve** | Steep | Moderate |
| **Performance** | Maximum control | Near-CUDA with less effort |
| **Language** | C++ | Python |

## Key Properties

- **JIT Compilation**: Kernels are compiled at runtime with `@triton.jit`
- **Automatic Tiling**: Triton handles shared memory allocation and data movement
- **Masking**: Built-in support for out-of-bounds access protection
- **Constexpr**: Compile-time constants for block sizes enable optimization
- **PyTorch Integration**: Seamless interoperability with PyTorch tensors

## Use Cases

- **Custom Fused Kernels**: Combine multiple operations to reduce memory I/O
- **Attention Mechanisms**: FlashAttention-style implementations
- **Specialized Layers**: Custom neural network operations
- **Research Prototyping**: Rapid iteration on GPU algorithms

## Connections

- [[GPU Architecture]] — Understanding hardware enables writing efficient Triton code
- [[Kernel Fusion]] — Primary application of Triton in deep learning
- [[Sparton]] — Example of Triton used for efficient [[Learned Sparse Retrieval]]

## Appears In

- [[IR-L14 - Triton and Sparton]]
