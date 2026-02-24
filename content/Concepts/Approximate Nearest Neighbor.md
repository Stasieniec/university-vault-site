---
type: concept
aliases: [ANN, ANN Search]
course: [IR]
tags: [neural-ir, vector-search, algorithms]
status: complete
---

# Approximate Nearest Neighbor

> [!definition] Approximate Nearest Neighbor (ANN)
> **ANN** refers to a class of algorithms designed to find vectors in a high-dimensional space that are "close" to a query vector, sacrificing perfect accuracy for significant gains in speed and memory efficiency.

> [!intuition] Accuracy vs. Latency
> In [[DPR|Dense Retrieval]], searching for the exact nearest neighbor in a collection of 20 million vectors requires a linear scan ($O(N)$), which is too slow. ANN algorithms create data structures that allow searching in $O(\log N)$ or $O(1)$ by only checking a promising subset of the data.

## Key Algorithms & Techniques

### 1. Inverted File Index (IVF)
- **Mechanism**: Clusters the vector space (e.g., using k-means). At search time, only search vectors in the nearest cluster centroids.
- **Trade-off**: High speed, but can miss neighbors at cluster boundaries.

### 2. HNSW (Hierarchical Navigable Small Worlds)
- **Mechanism**: A graph-based approach where vectors are nodes. It build a multi-layered graph where top layers have "long" skip-links for fast navigation and bottom layers have short-range links for precision.
- **Status**: Current gold-standard for speed/accuracy trade-off.

### 3. Product Quantization (PQ)
- **Mechanism**: Compresses vectors by splitting them into sub-vectors and quantizing each part.
- **Benefit**: Massive memory reduction (e.g., 10x-100x), allowing huge indexes to fit in RAM.

## Trade-offs

- **Recall**: The percentage of times the true 1st-nearest neighbor is actually found.
- **Latency**: Time per query.
- **Index Size**: Memory required on disk/RAM.

## Connections

- Tooling: **FAISS** (Facebook AI Similarity Search).
- Usage: Enables first-stage retrieval for [[DPR]].
- Part of: [[Multi-Stage Ranking]] infrastructure.

## Appears In

- [[IR-L06 - Dense Retrieval]]
