---
type: book-chapter
course: IR
book: "Pretrained Transformers for Text Ranking: BERT and Beyond"
chapter: 3
sections: ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6"]
topics:
  - "[[BERT for IR]]"
  - "[[MonoBERT]]"
  - "[[monoT5]]"
  - "[[duoT5]]"
  - "[[Cross-Encoder]]"
  - "[[Neural Reranking]]"
  - "[[Multi-Stage Ranking]]"
  - "[[Transformers]]"
status: complete
---

# IR-PTR Ch3 - Multi-Stage Architectures for Reranking

## Overview
The core formulation of text ranking in the transformer era is **relevance classification**. This involves training a classifier to estimate the probability that a text belongs to the "relevant" class and sorting documents by these probabilities at inference time. This is a direct realization of the **Probability Ranking Principle**.

> [!definition] Relevance Classification
> Sorting texts based on the estimated probability $P(\text{Relevant}=1 | d, q)$ where $d$ is a document and $q$ is a query.

### Retrieve-and-Rerank Architecture
To handle scalability, most systems use a multi-stage approach:
1.  **Candidate Generation (First-stage):** Uses an inverted index with efficient scoring like [[BM25]] to retrieve $k$ candidates (e.g., $k=1000$).
2.  **Reranking (Second-stage):** Uses a complex model like [[MonoBERT]] to rerank the candidates.

## 3.1 BERT Basics for IR
[[Transformers]]-based contextual embeddings (like BERT) capture syntax, semantics, and polysemy better than static embeddings (word2vec, GloVe).

### Architecture & Components
*   **Input Template:** `[[CLS], tokens..., [SEP], tokens..., [SEP]]`
*   **Embeddings:** Token + Segment (A/B) + Position embeddings (summed element-wise).
*   **Contextual Output:** $T_{[CLS]}$ is typically used as an aggregate representation for classification.

### Pretraining Objectives
1.  **Masked Language Model (MLM):** Predicting "masked" tokens using bidirectional context.
2.  **Next Sentence Prediction (NSP):** Predicting if segment B follows segment A.

> [!tip] Pretrain-then-Fine-tune
> The standard recipe: pretrain on massive corpora using self-supervision (MLM), then fine-tune on task-specific labeled data (e.g., MS MARCO).

---

## 3.2 [[MonoBERT]]: The Baseline Reranker
The first adaptation of BERT for ranking treats query $q$ and document $d$ as two segments in a **[[Cross-Encoder]]** setup.

> [!formula] monoBERT Scoring
> The relevance score $s_i$ is computed using a fully-connected layer over the $T_{[CLS]}$ token:
> $$s_i = \text{softmax}(T_{[CLS]} W + b)_1$$
> Where $W \in \mathbb{R}^{D \times 2}$ and $b \in \mathbb{R}^2$.

### Training
Trained using pointwise **Cross-Entropy Loss**:
$$\mathcal{L} = - \sum_{j \in J_{pos}} \log(s_j) - \sum_{j \in J_{neg}} \log(1 - s_j)$$

### Key Findings
*   **Data Hunger:** monoBERT requires significant data (thousands of pairs) to beat [[BM25]].
*   **Position matters:** Removing position embeddings significantly degrades performance.
*   **No "help" needed:** Interpolating with BM25 scores often doesn't help monoBERT on MS MARCO once $k=1000$ candidates are selected.

---

## 3.3 Passage to Document Ranking (Handling Long Texts)
BERT's 512-token limit creates challenges for "full-length" documents (news articles, papers).

### 3.3.1 Birch (Sentence-level)
*   **Approach:** Rerank individual sentences and aggregate scores.
*   **Aggregation:** Documents are scored based on the top $n$ scoring sentences combined with the original document score:
    $$s_f = \alpha \cdot s_d + (1 - \alpha) \cdot \sum_{i=1}^n w_i \cdot s_i$$
*   **Intuition:** The highest-scoring sentence is a good proxy for document relevance. Supports **zero-shot cross-domain transfer** (e.g., training on tweets, testing on news).

### 3.3.2 BERT-MaxP (Passage-level)
*   **Approach:** Segment documents into overlapping passages (e.g., 150 words, stride 75).
*   **Aggregation:** $s_d = \max s_i$ (MaxP). FirstP and SumP are alternatives.
*   **Query Representation:** Sentence-long natural language "descriptions" outperform keyword "titles" because BERT exploits non-content words (prepositions, etc.) for deeper context.

### 3.3.3 CEDR (Contextualized Embeddings)
*   **Approach:** Uses all contextual embeddings $T_1 \dots T_n$, not just $T_{[CLS]}$.
*   **Design:** Feeds BERT embeddings into pre-BERT interaction models like KNRM or PACRR to build similarity matrices.
*   **Aggregation:** Representation aggregation across document chunks.

### 3.3.4 PARADE (Representation Aggregation)
*   **Approach:** Aggregates **representations** ($p_{cls}$ vectors) from passages using a hierarchical transformer or CNN.
*   **Model:**
    $$d_{cls} = \text{Transformer}([CLS], p_{cls1}, \dots, p_{clsn})$$
*   **Advantage:** End-to-end differentiable, unifies training/inference, more effective than simple score aggregation.

---

## 3.4 Multi-Stage Reranking & Pipelines

### 3.4.1 Pairwise Reranking ([[duoT5]] / duoBERT)
Instead of pointwise probability, estimate $P(d_i \succ d_j | d_i, d_j, q)$.
*   **Cost:** Complexity $O(k^2)$, requiring a multi-stage approach where monoBERT filters $k$ to a smaller set (e.g., 50) before duoBERT processes it.
*   **Loss:** Pairwise hinge/logistic loss.

### 3.4.2 Cascade Transformers
Treats transformer layers as a pipeline with **early exits**.
*   **Intuition:** Discard clear non-relevant candidates after a few layers (e.g., 4 or 6) rather than running all 12 layers on 1000 documents.

---

## 3.5 Beyond BERT

### 3.5.1 Knowledge Distillation
Distilling a large "Teacher" into a "Student" (e.g., TinyBERT, DistilBERT).
*   **Objective:** Minimize MSE between teacher and student logits or hidden states.
*   **Efficiency:** Can achieve up to 9x speedup with minimal loss in effectiveness.

### 3.5.2 Local Architectures (TK, TKL)
*   **[[Transformer Kernel (TK)]]:** Small, from-scratch transformers with local attention to avoid the $O(L^2)$ cost and use interaction kernels (KNRM).
*   **Conformer Kernel (CK):** Mixes convolutions and attention for higher efficiency.

### 3.5.3 Ranking with Sequence-to-Sequence ([[monoT5]])
*   **Concept:** "Text-to-text" paradigm for everything.
*   **Encoding:** `Query: q Document: d Relevant:`
*   **Decoding:** Model generates the token `"true"` or `"false"`.
*   **Score Calculation:** $P(\text{Relevant}) = \frac{e^{\text{logit}(\text{true})}}{e^{\text{logit}(\text{true})} + e^{\text{logit}(\text{false})}}$
*   **Benefit:** Highly effective and extremely data-efficient (great for few-shot).

### 3.5.4 Generative Query Likelihood
Ranking by $P(q|d)$ using models like BART or GPT-2. The "reverse" of standard classification.

---

## Summary Takeaways
1.  **Relevance Classification** is the dominant paradigm (Cross-Encoders).
2.  **Aggregation Matters:** Moving from passage *scores* to passage *representations* (PARADE) improves document ranking.
3.  **Efficiency Tradeoffs:** Pairwise models ([[duoT5]]) add quality but require a second stage to manage latency.
4.  **Generative Future:** [[monoT5]] and seq2seq models are currently state-of-the-art for many tasks.
