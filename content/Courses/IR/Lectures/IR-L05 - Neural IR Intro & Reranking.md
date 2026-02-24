---
type: lecture
course: IR
week: 3
lecture: 5
topics:
  - "[[Neural Reranking]]"
  - "[[Cross-Encoder]]"
  - "[[Bi-Encoder]]"
  - "[[BERT for IR]]"
  - "[[MonoBERT]]"
  - "[[Multi-Stage Ranking]]"
status: complete
---

# IR-L05 - Neural IR Intro & Reranking

## Overview
This lecture introduces **Neural Information Retrieval (Neural IR)**, focusing on how deep learning models can overcome the fundamental limitations of classical lexical models (like BM25). We explore the shift from symbolic representations to continuous embeddings, the importance of **Multi-Stage Ranking**, and the role of Transformers (specifically BERT) in modern retrieval pipelines.

---

## 1. Motivation: Why Neural IR?

### 1.1 Limitations of Classical IR
Classical IR models (like BM25) rely on **lexical matching** (exact term overlap). This leads to several issues:
1.  **Vocabulary Mismatch:** $q = \text{"car insurance"}$ and $d = \text{"automobile coverage policies"}$ have zero overlap but share semantic meaning.
2.  **Lack of Semantic Generalization:** Models cannot handle synonyms (doctor/physician), polysemy (bank - river vs. finance), or paraphrases.
3.  **Handcrafted Features:** Requires manual tuning of scoring functions and reweighting.
4.  **Limited Learning:** Learning-to-Rank (LTR) can reweight signals but cannot "learn" new text representations.

### 1.2 The Neural Paradigm
Neural IR replaces symbolic representations with **continuous embeddings**:
$$\psi(\cdot) : \text{text} \to \mathbb{R}^k$$

> [!intuition] Distributional Hypothesis
> "A word is characterized by the company it keeps" (Firth, 1957). If words occur in similar contexts, they tend to have similar meanings. Neural models learn these contexts to map similar concepts close together in vector space.

---

## 2. Multi-Stage Ranking Pipeline

Neural models (especially Cross-Encoders) are computationally expensive. Therefore, modern systems use a **cascaded architecture**:

1.  **Retrieval Phase (First Stage):**
    *   **Goal:** Efficient candidate generation from millions of documents.
    *   **Models:** BM25, Inverted Indexes, or Bi-Encoders.
    *   **Output:** Top-K documents (e.g., $K=100$ or $1000$).
2.  **Reranking Phase (Second Stage):**
    *   **Goal:** Precise semantic ranking of candidates.
    *   **Models:** Cross-Encoders (MonoBERT, MonoT5).
    *   **Output:** Final ranked list for the user.

---

## 3. Early Neural IR (Word Embeddings)

### 3.1 Distributional Representations
*   **Static Word Embeddings (Word2Vec, GloVe):** Fixed vector for each word.
*   **Strengths:** Reduces lexical mismatch; enables semantic similarity.
*   **Weaknesses:** Cannot handle **Polysemy** (one vector per word regardless of context) and lacks a mechanism for phrase/sentence composition.

### 3.2 Interaction-focused Models (DRMM)
**DRMM (Deep Relevance Matching Model)** shifted focus from global semantic similarity to term-level interaction.

> [!formula] DRMM Logic
> 1. Compute cosine similarity between each query term $q_i$ and all document terms $d_j$ into a **similarity matrix**.
> 2. Group similarities into **buckets** (histograms) to handle varying document lengths.
> 3. Pass buckets through an MLP.
> 4. Aggregate using query-term weights (e.g., IDF).

**Takeaway:** Interaction-based models are better for ad-hoc retrieval because rare term matches often matter more than overall topic similarity.

---

## 4. BERT for IR: Cross-Encoders vs. Bi-Encoders

### 4.1 Cross-Encoders (e.g., MonoBERT)
Cross-encoders jointly process the query and document.

> [!definition] MonoBERT
> Input: `[CLS] query [SEP] document [SEP]`
> BERT processes all query-document token interactions through self-attention. A classification head on the `[CLS]` token predicts the relevance score $s(q, d)$.

*   **Strengths:** High effectiveness; captures deep token-level interactions.
*   **Weaknesses:** Extremely slow. Must run the full model for every document in the candidate set at query time. Complexity: $O(N \cdot L^2)$ where $N$ is candidate count and $L$ is sequence length.

### 4.2 Bi-Encoders (Siamese Networks)
Queries and documents are encoded independently.
*   **Score:** $s(q, d) = \cos(\psi(q), \psi(d))$ or dot product.
*   **Advantage:** Document embeddings can be **pre-computed** and indexed (e.g., FAISS). Query time is just one forward pass + vector search.

---

## 5. MonoT5: Generative Reranking
MonoT5 reformulates ranking as a **text-to-text** task.
*   **Prompt:** `Query: [Q] Document: [D] Relevant:`
*   **Target tokens:** `true` or `false`.
*   **Scoring:** The relevance score is calculated based on the Softmax probability of the token `true`.

---

## 6. Training Neural Rankers

### 6.1 Contrastive Learning
Models are often trained using a **pairwise** or **pointwise** approach.

> [!formula] Contrastive Loss (InfoNCE-style)
> $$L = -\log \frac{\exp(\langle \psi(q), \psi(d^+) \rangle)}{\exp(\langle \psi(q), \psi(d^+) \rangle) + \sum_{j} \exp(\langle \psi(q), \psi(d^-_j) \rangle)}$$
>
> where:
> - $d^+$: Positive (relevant) document.
> - $d^-_j$: Negative (non-relevant) document.

### 6.2 Sampling Negatives
Training effectiveness depends heavily on **Negative Mining**:
*   **In-batch negatives:** Using other positive documents in the same batch as negatives.
*   **Hard negatives:** Documents that are lexically similar (high BM25) but irrelevant.

---

## 7. Handling Long Documents
Transformers have a limited context window (usually 512 tokens).

| Method | Description |
| :--- | :--- |
| **FirstP** | Use only the first $n$ tokens of the document. |
| **MaxP** | Split doc into passages; use the score of the maximum scoring passage: $s(q,D) = \max_{p \in D} s(q,p)$. |
| **SumP** | Sum scores across passages. |
| **PARADE** | Encode query-passage pairs and use a learnable **aggregation** layer (e.g., Transformer, CNN) to combine passage embeddings into a document-level score. |

---

## 8. Summary Table: Efficiency vs. Effectiveness

| Model Type | Interaction | Latency | Scale | Effectiveness |
| :--- | :--- | :--- | :--- | :--- |
| **BM25** | Lexical | Ultra-low | Millions | Baseline |
| **Bi-Encoder** | Late (Cosine) | Low | Millions | High |
| **Cross-Encoder** | All-to-all | High | ~100-1000 | **State-of-the-art** |

> [!tip] Exam Hint
> Understand the **Multi-stage pipeline**. We use BM25 to get candidates because Cross-Encoders are too slow for the whole collection ($10^6$ docs), but we use Cross-Encoders to rerank the top-100 because they catch semantic nuances BM25 misses.
