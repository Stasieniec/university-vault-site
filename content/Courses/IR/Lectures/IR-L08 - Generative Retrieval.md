---
type: lecture
course: IR
week: 4
lecture: 8
topics:
  - "[[Generative Retrieval]]"
  - "[[Differentiable Search Index]]"
  - "[[Document Identifiers]]"
  - "[[Autoregressive Retrieval]]"
  - "[[GENRE]]"
  - "[[DSI]]"
status: complete
---

# IR Lecture 8: Generative Retrieval

## Overview
Generative Information Retrieval (GenIR) represents a paradigm shift from the traditional "retrieve-then-rank" pipeline. Instead of using an external index (inverted or vector-based) to look up documents, GenIR encodes the entire corpus into the **parameters of a single sequence-to-sequence model**. Retrieval is framed as an **autoregressive decoding** task, where the model directly generates the identifier (DocID) of relevant documents given a query.

## 1. The Paradigms of Retrieval

| Dimension | Classical (Retrieve-then-Rank) | Generative IR |
| :--- | :--- | :--- |
| **Index** | Inverted / Vector Index (External) | Model Parameters (Internal) |
| **Retrieval Step** | Lookup + Scoring (Deterministic) | Autoregressive Decoding |
| **Differentiable** | Partial (Dense only) | Fully End-to-End |
| **Corpus Update** | Re-index (Fast) | Fine-tune (Slow) |
| **Interpretability** | Document-level ranking | Token-level generation |
| **Knowledge** | Up-to-date (External) | Parametric memory (Static/Stale) |
| **Scalability** | Sub-linear (ANN index) | Linear in document count (Model capacity) |

> [!intuition] Simplification
> GenIR replaces the large external index with an internal one. The model "memorizes" documents by associating their content with specific identifiers.

---

## 2. Core Operations in Generative Retrieval

### 2.1 Indexing (Memorization Phase)
The model learns to map document content to its corresponding document identifier (docid).

> [!formula] Indexing Loss
> Given a corpus $D$ and docid set $ID$, the goal is to maximize the likelihood of the docid given the document $d$:
> $$\mathcal{L}_{\text{Indexing}}(D, ID; \theta) = - \sum_{d \in D} \log P(id \mid d; \theta)$$
> where $\theta$ are the model parameters.

### 2.2 Retrieval (Inference Phase)
Given a query $q$, the model generates the most relevant docids.

> [!formula] Retrieval Loss
> Given a query set $Q$ and relevant docids $I_Q$:
> $$\mathcal{L}_{\text{Retrieval}}(Q, I_Q; \theta) = - \sum_{q \in Q} \sum_{id_q \in I_Q} \log P(id_q \mid q; \theta)$$

### 2.3 Unified Training
The model is optimized end-to-end using a global objective that combines both indexing and retrieval:
$$\mathcal{L}_{\text{Global}} = \mathcal{L}_{\text{Indexing}} + \mathcal{L}_{\text{Retrieval}}$$

### 2.4 Inference (Decoding)
To retrieve the top-k documents, the model uses **constrained beam search** to generate docid strings $w$ token-by-token:
$$w_t = GR_\theta(q, w_0, w_1, \dots, w_{t-1})$$
Generation stops when the `<EOS>` token is produced. Candidates are ranked by their joint probability.

---

## 3. Key Architectures & Models

### 3.1 Differentiable Search Index (DSI)
Introduced by [Tay et al., 2022], DSI is the seminal GenIR model.
- **Phase 1 Indexing:** Maps document text/prefix to DocID.
- **Phase 2 Retrieval:** Maps query to DocID.
- **Inference:** Uses a trie-based constrained beam search to ensure only valid DocIDs are generated.

### 3.2 Neural Corpus Indexer (NCI)
Three major improvements over DSI:
1. **Prefix-aware weight-adaptive decoder:** Uses different heads for different levels of the identifier hierarchy.
2. **Query Augmentation:** Generates synthetic queries for documents and trains on (synthetic query, docid) pairs.
3. **Consistency Training:** Ensures similar queries produce the same DocID.

### 3.3 GENRE (Generative Entity Retrieval)
Focuses on entity linking by generating human-readable Wikipedia titles as identifiers.
- Uses **pre-trained language knowledge** (BART).
- Wikipedia titles act as a natural, structured ID space.

---

## 4. Document Identifier (DocID) Design

Choosing how to represent a document as a string is critical.

| ID Type | Example | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Naive/Atomic** | `1024`, `doc_1` | Simple | No semantic meaning; hard to learn |
| **Semantic String** | `Title`, `URL` | leverages LLM pre-training | Can be long, ambiguous, or rare tokens |
| **Hierarchical (Clusters)** | `1.2.5.4` | Efficient decoding via tries | Sensitive to clustering quality |
| **Semantic Numeric** | Product of k-means | Structured, fixed length | Requires separate clustering phase |

> [!tip] Finding
> **Text-based docids** (like titles) generalize better to unseen documents in dynamic corpora because they align with the language model's pre-training distribution.

---

## 5. Robustness & Recent Progress

Generative Retrieval is currently facing several research challenges categorized under "Robustness":

### 5.1 Explainability
Mechanistic research shows the decoder passes through three stages:
1. **Priming:** No query-specific info used.
2. **Bridging:** Cross-attention transfers info from encoder to decoder.
3. **Interaction:** MLPs process info to predict docids in final layers.

### 5.2 Accuracy & Relevance Alignment
A major issue is aligning **token-level generation** with **document-level relevance**.
- Traditional beam search might prune false negatives.
- **DRO (Direct Relevance Optimization):** A proposal to optimize document relevance directly via pairwise ranking, eliminating the need for reinforcement learning.

### 5.3 Reliability (Autoregressive Limitations)
- Autoregressive models can achieve perfect Top-1 precision but suffer in **Top-k Recall** due to local greedy pruning in beam search.
- There is a lower bound on error related to the KL divergence between ground-truth and predicted step-wise marginal distributions.

### 5.4 Dynamic Corpora (Repeatability)
Most GenIR models are static. Handling additions/deletions/modifications is difficult:
- **Numeric IDs** tend to stick to IDs seen during training.
- **Model Editing:** Techniques like "Model Editing" are being explored to integrate new documents without full re-training.

### 5.5 Safety & Machine Unlearning
How do we delete a document from a model's parameters (e.g., for GDPR "right to be forgotten")?
- Requires specialized **Machine Unlearning** algorithms to remove training data traces without full retraining.

---

## 6. Summary: The Retrieval Evolution

| Stage | Mechanism | Space |
| :--- | :--- | :--- |
| **Sparse** | Exact Lexical Overlap | Vocabulary Terms |
| **Dense** | Semantic Similarity | Latent Embeddings |
| **Generative** | Conditional Generation | Model Parameters |

> [!example] The GenIR Workflow
> 1. **Index:** Feed "How to roast pumpkin seeds" $\rightarrow$ Model learns to output `DocID_567`.
> 2. **Query:** User asks "Pumpkin seed storage" $\rightarrow$ Model decodes `DocID_567` autoregressively using beam search.
