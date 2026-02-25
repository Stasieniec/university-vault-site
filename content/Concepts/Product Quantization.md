---
type: concept
aliases: [PQ, product quantization]
course: [IR]
tags: [neural-ir, efficiency]
status: complete
---

# Product Quantization (PQ)

> [!definition] Product Quantization
> An approximate nearest neighbor search technique that compresses high-dimensional vectors by splitting them into sub-vectors and quantizing each sub-vector independently using a learned codebook. Distances are approximated using pre-computed lookup tables.

## Intuition

Storing and comparing millions of dense vectors (e.g., from [[Bi-Encoder]] models) is expensive. PQ provides massive memory savings by representing each vector as a short code (sequence of codebook indices) while still allowing fast approximate distance computation.

## How It Works

1. **Split**: Divide each $d$-dimensional vector into $m$ sub-vectors of dimension $d/m$
2. **Train codebooks**: For each sub-vector space, learn a codebook of $k$ centroids via k-means
3. **Encode**: Replace each sub-vector with the index of its nearest centroid → vector becomes $m$ integers
4. **Search**: Approximate distances using pre-computed lookup tables of distances between query sub-vectors and centroids

## Memory Savings

- Original vector: $d \times 4$ bytes (float32)
- PQ code: $m \times \lceil\log_2 k\rceil$ bits
- Example: $d=768$, $m=96$, $k=256$ → 96 bytes instead of 3072 bytes (**32× compression**)

## Key Properties

- **Asymmetric Distance Computation (ADC)**: query stays uncompressed, only documents are quantized → better accuracy
- **Additive error**: quantization error is bounded and decreases with more sub-vectors
- **Composable**: often combined with [[Approximate Nearest Neighbor|ANN]] methods (e.g., IVF-PQ)

## Connections

- Enables efficient [[Dense Retrieval]] at scale
- Used in FAISS (Facebook AI Similarity Search)
- Alternative to HNSW graph-based [[Approximate Nearest Neighbor]] search
- Relevant to [[Bi-Encoder]] and [[ColBERT]] index compression

## Appears In

- [[IR-L06 - Dense Retrieval]] (§5.2 — compression techniques)
