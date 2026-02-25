---
type: concept
aliases: [duoT5, duo-T5]
course: [IR]
tags: [neural-ir, reranking]
status: complete
---

# duoT5

> [!definition] duoT5
> A pairwise [[Neural Reranking|reranking]] model based on T5 that takes a query and **two** candidate documents as input, predicting which document is more relevant. It extends [[monoT5]] from pointwise to pairwise ranking.

## How It Works

1. **Input format**: `"Query: {q} Document0: {d₁} Document1: {d₂} Relevant:"`
2. **Output**: T5 generates a token — the probability of generating "true" for Document0 being more relevant than Document1
3. **Aggregation**: Pairwise preferences are aggregated across all document pairs to produce a final ranking (e.g., via a sorting-based approach)

## Architecture

- Built on T5 (Text-to-Text Transfer Transformer) — a sequence-to-sequence model
- Fine-tuned on MS MARCO or similar relevance datasets
- Typically used as a **second-stage reranker** after [[monoT5]] or [[MonoBERT|monoBERT]] narrows down candidates

## Multi-Stage Pipeline

```
Query → BM25 (1000 docs) → monoT5 (rerank to top-50) → duoT5 (pairwise rerank top-50)
```

> [!intuition] Why Pairwise?
> Comparing two documents side-by-side lets the model make finer-grained relevance distinctions than scoring each document independently (pointwise). The downside: $O(n^2)$ comparisons, so only feasible on small candidate sets.

## Key Properties

- **Higher effectiveness** than pointwise monoT5 on small candidate sets
- **Computationally expensive**: $\binom{n}{2}$ inference calls for $n$ documents
- **Practical only as final-stage reranker** on top-$k$ (small $k$)

## Connections

- Extends [[monoT5]] from pointwise to pairwise
- Part of the [[Multi-Stage Ranking]] pipeline
- Related to [[Cross-Encoder]] (both are expensive, high-quality rerankers)
- Pairwise approach connects to [[Learning to Rank]] (pairwise LTR losses like RankNet)

## Appears In

- [[IR-L05 - Neural IR Intro & Reranking]]
- PTR (Lin et al.) §3.4.1
