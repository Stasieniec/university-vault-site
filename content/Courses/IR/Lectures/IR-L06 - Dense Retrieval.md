---
type: lecture
course: IR
week: 3
lecture: 6
topics:
  - "[[Dense Retrieval]]"
  - "[[Bi-Encoder]]"
  - "[[DPR]]"
  - "[[ColBERT]]"
  - "[[Approximate Nearest Neighbor]]"
  - "[[Contrastive Learning]]"
  - "[[Hard Negative Mining]]"
status: complete
---

# IR-L06 - Dense Retrieval

## 1. Introduction to Dense Retrieval

Dense retrieval is a paradigm shift from traditional sparse retrieval (like BM25). Instead of matching exact keywords, it represents queries and documents as dense vectors in a continuous embedding space.

> [!definition] Dense Retrieval
> A retrieval method where queries $q$ and documents $d$ are mapped to a high-dimensional vector space $\mathbb{R}^d$ such that relevance is captured by vector similarity (e.g., dot product or cosine similarity).

### 1.1 Why Dense Retrieval?
1. **Semantic Matching**: Handles synonyms and paraphrases (e.g., "film" vs "movie").
2. **Contextual Awareness**: Word meanings depend on surrounding text (unlike term-matching).
3. **End-to-End Learning**: Retrieval can be optimized directly using relevance labels.

---

## 2. Architectures: Bi-Encoders vs. Interaction Models

### 2.1 Representation-focused (Bi-Encoder)
In a bi-encoder setup, the query and document are encoded independently. This is the foundation of **Dense Passage Retrieval (DPR)**.

- **Architecture**: Siamese (shared weights) or dual encoders ($E_Q$ and $E_D$).
- **Similarity**: Typically a simple dot product:
  $$sim(q, d) = E_Q(q)^\top E_D(d)$$
- **Efficiency**: Documents can be pre-computed, indexed, and searched using [[Approximate Nearest Neighbor]] (ANN).

### 2.2 Interaction-focused (Cross-Encoder)
Cross-encoders feed the query and document into the model simultaneously.
- **Complexity**: $O(N)$ transformer passes per query (where $N$ is corpus size).
- **Quality**: Very high due to full self-attention between all query and document tokens.
- **Limitation**: Infeasible for first-stage retrieval over millions of documents.

### 2.3 Late Interaction: [[ColBERT]]
ColBERT (Contextualized Late Interaction over BERT) provides a middle ground.
- **Mechanism**: Encodes $q$ and $d$ into token-level embeddings.
- **MaxSim Operator**: Computes similarity as the sum of maximum inner products for each query token.
  $$S_{q,d} = \sum_{i \in [|E_q|]} \max_{j \in [|E_d|]} E_{q_i} \cdot E_{d_j}^\top$$
- **Benefit**: Captures soft term matching and interaction without full cross-attention at search time.

---

## 3. Training: [[Contrastive Learning]]

Dense retrievers are trained to maximize the similarity between a query and its positive document relative to irrelevant "negative" documents.

### 3.1 Loss Function
The standard is Negative Log-Likelihood (NLL) over a softmax:
$$\mathcal{L} = -\log \frac{e^{sim(q_i, p_i^+)}}{e^{sim(q_i, p_i^+)} + \sum_{j} e^{sim(q_i, p_{i,j}^-)}}$$

where:
- $p_i^+$: Positive passage containing the answer.
- $p_{i,j}^-$: Negative passage (irrelevant).

### 3.2 Negative Sampling Strategies
The choice of negatives is critical for performance:
- **Random Negatives**: Too easy; model doesn't learn fine distinctions.
- **In-batch Negatives**: Efficient; use the positive passages of other queries in the same batch as negatives.
- **BM25 Negatives**: High-lexical overlap passages that do not contain the answer.

---

## 4. [[Hard Negative Mining]]

To bridge the gap between training and inference, the model needs "hard" negatives—passages that the retriever currently ranks highly but are actually irrelevant.

### 4.1 ANCE (Approximate Nearest Neighbor Negative Contrastive Learning)
1. Train an initial retriever.
2. Build a global ANN index of the entire corpus.
3. Retrieve top-$k$ nearest neighbors for queries and use them as negatives.
4. Update the index periodically as the retriever improves.

### 4.2 TAS (Topic Aware Sampling)
Clusters queries into topics and samples negatives from the same cluster to ensure they are semantically related but irrelevant.

> [!intuition] The "Not Too Hard" Rule
> The hardest negatives (highest score but irrelevant) can sometimes be "false negatives" (actually relevant but not labeled). Optimal learning often comes from "semi-hard" negatives.

---

## 5. Indexing and [[Approximate Nearest Neighbor]] (ANN)

Exact search $O(N \cdot d)$ is too slow for millions of vectors. ANN trades a small amount of recall for massive speed gains.

### 5.1 Partition-based: IVF (Inverted File)
- **Concept**: Use K-means to partition the vector space into clusters (voronoi cells).
- **Search**: Query only the nearest few clusters.

### 5.2 Quantization: Product Quantization (PQ)
- **Mechanism**: Splits a vector into sub-vectors and quantizes each into a codebook.
- **Efficiency**: Huge memory savings; distances are approximated using pre-computed lookup tables.

### 5.3 Graph-based: HNSW (Hierarchical Navigable Small World)
- **Concept**: A multi-layered graph where the top layer has long-range edges and the bottom layer contains the full dataset.
- **Search**: Greedy traversal from top to bottom.
- **Pros**: Very high recall and speed.
- **Cons**: High memory usage.

### 5.4 MIPS to L2 reduction
Many ANN algorithms optimize for Euclidean distance ($L_2$), but retrieval needs Maximum Inner Product Search (MIPS).
- We can transform a MIPS problem in $\mathbb{R}^\ell$ into an $L_2$ problem in $\mathbb{R}^{\ell+1}$ by adding a dimension to normalize vector norms.

---

## 6. Challenges and Limitations

### 6.1 Zero-shot Generalization
Dense retrievers often fail when moving from one domain (e.g., MS MARCO / Web) to another (e.g., Biomedical or Legal).
- **Lexical Fallback**: BM25 often outperforms dense models in zero-shot settings because it doesn't rely on domain-specific semantics.
- **The BEIR Benchmark**: Shows that DPR struggles compared to sparse models in wide-distribution tasks.

### 6.2 Implementation Details (DPR)
- **Passage Size**: Fixed length of 100-token chunks.
- **Backbone**: BERT-base.
- **Shared vs. Dual Encoders**: Sharing parameters between query and document encoders (SDE) often improves performance by keeping representations in the same space.

---

## Summary Table: Retrieval Spectrum

| Model | Selection | Interaction | Speed | Memory |
| :--- | :--- | :--- | :--- | :--- |
| **BM25** | Lexical | Exact Match | +++ | + |
| **Bi-Encoder (DPR)** | Semantic | Single-vector dot product | +++ | ++ |
| **ColBERT** | Semantic | Token-level MaxSim | ++ | +++ |
| **Cross-Encoder** | Deep Semantic | Full Self-Attention | + | N/A |

> [!tip] Practical Implementation
> Most industrial RAG systems today use a hybrid approach: **IVF + PQ** for scale or **HNSW** for performance, often combined with a second-stage cross-encoder re-ranker.
