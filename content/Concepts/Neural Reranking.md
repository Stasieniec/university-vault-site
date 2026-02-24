---
type: concept
aliases: [Reranking, Neural Ranking]
course: [IR]
tags: [neural-ir, reranking, deep-learning]
status: complete
---

# Neural Reranking

> [!definition] Neural Reranking
> **Neural Reranking** is the process of applying deep learning models (typically Transformers like BERT) to re-evaluate and re-order the top-$k$ results (e.g., $k=100$) retrieved by a first-stage model (e.g., [[BM25]]).

> [!intuition] Why Rerank?
> First-stage retrieval (e.g., [[BM25]]) is fast but relies on exact keyword matching (lexical mismatch problem). Neural models are powerful but too slow to score millions of documents. Reranking combines the best of both: fast initial filtering followed by expensive but precise semantic scoring of the most promising candidates.

## Architectures: Bi-Encoders vs. Cross-Encoders

Neural rerankers typically follow one of two paradigms, balancing performance and efficiency:

### 1. Bi-Encoders (Dual Encoders)
- **Mechanism**: Query and document are encoded independently into vectors $\mathbf{q}$ and $\mathbf{d}$. Scoring is a simple dot product or cosine similarity.
- **Trade-off**: Lower precision (no interaction between query and document tokens) but extremely fast.
- **Used in**: First-stage [[DPR|Dense Retrieval]].

### 2. Cross-Encoders
- **Mechanism**: The query and document are concatenated as a single input $[CLS]\,q\,[SEP]\,d\,[SEP]$ and passed through the model.
- **Trade-off**: High precision (all-to-all attention between q and d tokens) but very slow (latency scales with $k$).
- **Examples**: [[MonoBERT]], [[monoT5]].

> [!tip] Performance vs. Latency
> Cross-encoders are the state-of-the-art for ranking quality, but their computational cost forces us to keep the reranking list ($k$) small.

## Connections

- Part of: [[Multi-Stage Ranking]] pipelines.
- Implemented by: [[MonoBERT]], [[monoT5]].
- Comparison: Improves upon sparse models like [[BM25]] by capturing semantics.
- Opposite of: [[Dense Retrieval]] (which encodes separately).

## Appears In

- [[IR-L05 - Neural IR]]
