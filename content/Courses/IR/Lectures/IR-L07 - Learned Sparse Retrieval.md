---
type: lecture
course: IR
week: 4
lecture: 7
topics:
  - "[[Learned Sparse Retrieval]]"
  - "[[SPLADE]]"
  - "[[Document Expansion]]"
  - "[[Query Expansion]]"
  - "[[DeepCT]]"
  - "[[DeepImpact]]"
  - "[[uniCOIL]]"
  - "[[Term Weighting]]"
status: complete
---

# IR-L07 - Learned Sparse Retrieval

## Overview
Learned Sparse Retrieval (LSR) represents a hybrid approach in Information Retrieval that bridges the gap between **traditional sparse retrieval** (like BM25) and **modern neural dense retrieval** (like DPR). LSR projects queries and documents into high-dimensional sparse vectors over a fixed vocabulary (usually the one from a Pretrained Language Model like BERT).

Key advantages:
- **Efficiency:** Compatible with existing inverted index infrastructures.
- **Effectiveness:** Utilizes neural architectures (Transformers) to learn importance weights and expand text with semantically relevant terms.
- **Interpretability:** Unlike dense vectors, LSR dimensions often map directly to vocabulary tokens.

---

## 1. Motivation: Why Sparse, why Learned, why Now?

### The Traditional vs. Neural Gap
- **Traditional Sparse (BM25):** Fast and scalable due to inverted indices, but suffers from the **vocabulary mismatch problem** (cannot find relevant documents that use different synonyms).
- **Dense Retrieval (Neural):** High effectiveness by capturing semantics in continuous latent space, but requires specialized vector databases (ANN) and significant computational resources.

### Bridging the Gap
LSR aims to achieve neural-level effectiveness while maintaining the efficiency of sparse indices.

> [!definition] Learned Sparse Retrieval (LSR)
> LSR uses a query encoder $E_Q$ and a document encoder $E_D$ to project queries $q$ and documents $d$ into sparse vectors over a fixed vocabulary $V$.
> 
> The relevance score is computed as a sparse dot product:
> $$score(q, d) = \sum_{j=1}^{|V|} E_Q(q)_j \cdot E_D(d)_j = E_Q(q)^T E_D(d)$$
> 
> where:
> - $E_Q(q)_j$ — learned weight of the $j$-th term in the query.
> - $E_D(d)_j$ — learned weight of the $j$-th term in the document.

---

## 2. A First Attempt: SNRM
**Standalone Neural Ranking Model (SNRM)** (2018) was the first neural retrieval model to learn high-dimensional sparse representations for inverted indexing.

- **Non-grounded:** Dimensions were latent (did not map to actual words).
- **Architecture:** Used a sliding window over token embeddings followed by average pooling.
- **Sparsity:** Enforced through $L_1$ regularization.
- **Legacy:** Proved that neural models could be used for first-stage retrieval without a lexical ranker.

---

## 3. Grounding and Expansion
Modern LSR models are **grounded**, meaning each activated dimension corresponds to a real token in the vocabulary.

### 3.1 Document Expansion (doc2query)
To solve the vocabulary mismatch problem, documents are expanded with terms they don't explicitly contain but are semantically relevant.

- **doc2query / docTTTTTquery:** Uses a sequence-to-sequence model (like T5) to predict potential queries a document could answer. These predicted queries are appended to the document before standard indexing.
- **Effect:** Increases recall by providing "more entry points" for queries.

### 3.2 Term Importance Prediction
Standard TF-IDF assumes importance based on frequency. LSR learns to predict "Term Impact" directly.

- **DeepCT:** Uses BERT to predict term importance (contextual term weighting). It maps BERT outputs to a value that replaces the traditional "Term Frequency" (TF) in the BM25 formula.
- **DeepImpact:** Learns impact scores directly using a neural model, storing them in the inverted index.

### 3.3 uniCOIL
**uniCOIL** combines term weighting and expansion. It uses a BERT-based encoder to weight tokens and relies on doc2query for expansion. It simplifies the scoring to a sum of weighted overlaps:
$$score(q, d) = \sum_{t \in q \cap d} w_{q,t} \cdot w_{d,t}$$

---

## 4. Sparse Representation Learning: SPLADE
**Sparsified Lexical and Expansion (SPLADE)** is a state-of-the-art LSR family.

### 4.1 Architecture
SPLADE leverages the **Masked Language Modeling (MLM)** head of BERT. 
1. For each token $i$ in the input, get the vocabulary-wide distribution from the MLM head.
2. For each term $j$ in the vocabulary $V$, compute the importance $w_{ij}$.
3. Aggregate these across all input tokens (usually using a `max` operation) to get the final vector $s$:
   $$s_j = \max_{i \in \text{input}} \log(1 + \text{relu}(w_{ij}))$$

> [!intuition] Why MLM works for expansion?
> MLM is pre-trained to predict missing tokens based on context. In SPLADE, we don't mask anything; we just use the MLM head to ask: "Given this context, what other words are semantically plausible here?" This naturally leads to expansion terms.

### 4.2 Training Objectives
- **Ranking Loss:** Contrastive loss (e.g., InfoNCE or MarginMSE) to ensure relevant documents score higher than non-relevant ones.
- **Distillation:** Often trained using a Cross-Encoder (Teacher) to guide the LSR model (Student).

### 4.3 Sparsity via FLOPs Regularization
To ensure the vectors are actually sparse (and thus efficient), a regularization term is added to the loss:
$$L = L_{ranking} + \lambda L_{reg}$$

Instead of simple $L_1$, SPLADE often uses **FLOPs regularization**, which minimizes the expected number of operations during retrieval:
$$L_{FLOPs} = \sum_{j \in V} (\bar{a}_j)^2$$
where $\bar{a}_j$ is the average activation of term $j$ in a batch.

---

## 5. "Wacky" Weights and Interpretability
Recent research (Mackenzie et al., 2021) shows that LSR weights can be counter-intuitive:
- **"Wacky" expansion:** Models might give high weights to stopwords (like `the`, `is`) or punctuation (`,`).
- **Reason:** Neural training exploits any signal that correlates with relevance. If definitional documents always contain a comma after the term, the model learns to weight the comma.
- **Takeaway:** LSR is optimized for **effectiveness**, not necessarily for human-readable semantic meaning.

---

## 6. Comparison: Sparse vs. Dense vs. Hybrid

| Feature | BM25 | Dense (e.g. DPR) | LSR (e.g. SPLADE) |
| :--- | :--- | :--- | :--- |
| **Representation** | Term counts | Continuous latent | Learned weights + expansion |
| **Storage** | Inverted Index (Small) | Vector Index (Large) | Inverted Index (Medium) |
| **Latency** | Very Low | High (requires ANN) | Low |
| **Vocabulary Gap** | High | Low | Low |
| **Interpretability** | High | Low | Medium |

> [!tip] Hybrid Retrieval
> Many production systems use **Hybrid Search**, combining BM25 and Dense/LSR scores through **Reciprocal Rank Fusion (RRF)** to get the "best of both worlds."

---

## Key Takeaways
- **LSR** projects text into sparse, vocabulary-aligned neural representations.
- **SPLADE** is the dominant architecture, utilizing BERT's MLM head for both weighting and expansion.
- **Regularization** (like FLOPs) is essential to keep the index efficient.
- **Efficiency** is the main selling point: we get neural performance using the same "search engine" tech (Lucene/Elasticsearch) we've used for decades.

---
**Related Concepts:**
- [[BM25]]
- [[Information Retrieval Overview]]
- [[Dense Retrieval and Bi-Encoders]]
- [[Inverted Indexing]]
