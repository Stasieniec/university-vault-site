---
type: book-chapter
course: IR
book: "Pretrained Transformers for Text Ranking: BERT and Beyond"
chapter: 5
sections: ["5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7"]
topics:
  - "[[Dense Retrieval]]"
  - "[[Bi-Encoder]]"
  - "[[DPR]]"
  - "[[ColBERT]]"
  - "[[Contrastive Learning]]"
  - "[[Hard Negative Mining]]"
  - "[[Learned Sparse Retrieval]]"
  - "[[SPLADE]]"
  - "[[Approximate Nearest Neighbor]]"
  - "[[Product Quantization]]"
status: complete
---

# IR-PTR Chapter 5: Dense Retrieval and Learned Sparse Retrieval

## 5.1 Overview: The Dense Retrieval Paradigm
This chapter explores the shift from exact-match [[Information Retrieval]] (sparse) to **[[Dense Retrieval]]**, where queries and documents are mapped into a shared low-dimensional continuous vector space.

> [!definition] Dense Retrieval
> A retrieval paradigm where both the query $q$ and document $d$ are represented as dense vectors $\mathbf{h}_q, \mathbf{h}_d \in \mathbb{R}^D$ (typically $D=768$ for BERT-base). Relevance is defined by a similarity function $\phi(\mathbf{h}_q, \mathbf{h}_d)$, usually the inner product or cosine similarity.

### The Bi-Encoder vs. Cross-Encoder Tradeoff
Unlike the [[Cross-Encoder]] (which allows all-to-all attention between query and document terms), the **[[Bi-Encoder]]** (or dual-encoder) processes the query and document independently.
- **Pros**: Document representations can be precomputed and indexed; supports sub-linear time retrieval via [[Approximate Nearest Neighbor]] (ANN) search.
- **Cons**: Loses the fine-grained interaction between $q$ and $d$ terms, typically leading to lower effectiveness (the "interaction gap").

## 5.2 Bi-Encoder Architecture
A standard bi-encoder uses two encoders (often sharing weights, i.e., Siamese network):
- $\mathbf{h}_q = \text{enc}_q(q)$
- $\mathbf{h}_d = \text{enc}_d(d)$

> [!formula] Similarity Score
> The ranking score is computed as:
> $$s(q, d) = \phi(\eta_q(q), \eta_d(d)) = \mathbf{h}_q^\top \mathbf{h}_d$$

Commonly, the `[CLS]` token representation from [[Transformers]] like [[BERT for IR]] is used as the aggregate vector.

## 5.3 Training Strategies for Dense Retrieval

### 5.3.1 Contrastive Learning and Loss Functions
Bi-encoders are trained to maximize the similarity of positive pairs $(q, d^+)$ and minimize the similarity of negative pairs $(q, d^-)$.

> [!formula] InfoNCE / Contrastive Loss
> For a query $q_i$, a positive document $d_i^+$, and a set of negatives $\{d_{i,j}^-\}$, the loss is:
> $$\mathcal{L}_i = - \log \frac{\exp(s(q_i, d_i^+))}{\exp(s(q_i, d_i^+)) + \sum_{j} \exp(s(q_i, d_{i,j}^-))}$$

### 5.3.2 Selecting Negative Examples
Success in dense retrieval is highly dependent on the "quality" of negative samples:
- **In-batch Negatives**: Used in [[DPR]] (Dense Passage Retrieval). For a batch of $B$ queries, the $B-1$ positive documents for other queries in the same batch serve as negatives for the current query.
- **[[Hard Negative Mining]]**: Selecting negatives that the current model evaluates as highly relevant (high score) but are actually non-relevant. 
- **ANCE**: (Approximate Nearest Neighbor Negative Contrastive Estimation) involves iteratively updating the [[Inverted Index]]/ANN index during training to sample the most informative "global" hard negatives.

### 5.3.3 Knowledge Distillation
Distilling a powerful [[Cross-Encoder]] (Teacher) into a [[Bi-Encoder]] (Student) often yields better results than training on labels alone.
- **Margin-MSE**: Distils the score *margins* to preserve the ranking order.
> [!formula] Margin-MSE Loss
> $$\mathcal{L} = \text{MSE}(s_{student}(q, d^+) - s_{student}(q, d^-), s_{teacher}(q, d^+) - s_{teacher}(q, d^-))$$

## 5.4 Late Interaction: ColBERT
**[[ColBERT]]** (Contextualized Late Interaction over BERT) bridges the gap between bi-encoders and cross-encoders. It stores multiple embeddings per document (one per token).

> [!intuition]
> Instead of compressing a document into *one* vector, ColBERT delays the interaction until the very end using a **MaxSim** operator, allowing query terms to match the "best" document term.

> [!formula] MaxSim Operator
> $$s_{q,d} = \sum_{i \in \eta(q)} \max_{j \in \eta(d)} \eta(q)_i \cdot \eta(d)_j$$

- **Pros**: High effectiveness, competitive with Cross-Encoders.
- **Warning**: High storage requirement. Indexing millions of passages can require hundreds of GBs of RAM/Disk to store per-token vectors.

## 5.5 ANN Search: Indexing and Retrieval
To avoid $O(N)$ brute-force search over the corpus, dense retrieval relies on:
1. **[[Approximate Nearest Neighbor]] (ANN)**: Algorithms like **HNSW** (Hierarchical Navigable Small World) or **FAISS**.
2. **[[Product Quantization]] (PQ)**: Compressing vectors into short codes to save memory and speed up distance calculations.
3. **LSH**: (Locality Sensitive Hashing) for grouping similar vectors.

## 5.6 Learned Sparse Retrieval
Methods like **[[SPLADE]]** and **[[uniCOIL]]** use transformer encoders but project the output back into the vocabulary space ($|V| \approx 30,000$).

> [!tip] Hybrid Approach
> These methods produce "sparse" vectors where most dimensions are zero, allowing them to use traditional **[[Inverted Index]]** structures while benefiting from neural term expansion and weighting.

- **[[SPLADE]]**: Uses Log-Saturation effect and Sparsity regularization (FLOPs/$L_1$) to learn which terms to expand.
> [!formula] SPLADE regularization
> $$\mathcal{L} = \mathcal{L}_{ranking} + \lambda_1 ||\mathbf{w}_q||_1 + \lambda_2 ||\mathbf{w}_d||_1$$

## 5.7 Comparison and Summary

| Feature | Sparse ([[BM25]]) | Dense (Bi-Encoder) | Late Interaction ([[ColBERT]]) |
| :--- | :--- | :--- | :--- |
| **Matching** | Exact term match | Semantic/Latent | Token-level semantic |
| **Index** | [[Inverted Index]] | ANN Index | Multi-vector ANN |
| **Efficiency** | Very High | High | Medium |
| **Effectiveness** | Baseline | High | Very High |

### Key Takeaways
1. Bi-encoders provide a massive speedup by decoupling query and document processing.
2. The bottleneck is often the **interaction gap**; [[ColBERT]] is a leading solution.
3. **Training Matters**: The choice of negatives (In-batch vs. ANCE) and distillation (Margin-MSE) is critical.
4. [[Learned Sparse Retrieval]] (SPLADE) offers a middle ground, providing neural expansion within efficient inverted indexes.
