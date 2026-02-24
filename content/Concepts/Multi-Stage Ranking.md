---
type: concept
aliases: [Multi-Stage Ranking, Ranking Pipeline]
course: [IR]
tags: [neural-ir, architecture, efficiency]
status: complete
---

# Multi-Stage Ranking

> [!definition] Multi-Stage Ranking
> **Multi-Stage Ranking** is a retrieval architecture that pipes results through progressively more complex and expensive models. It balances the "Efficiency vs. Effectiveness" trade-off.

> [!intuition] The Funnel Analogy
> Searching millions of documents is like finding a needle in a haystack. You can't use a microscope (expensive model) on every straw.
> 1. **First Stage (Retrieval)**: Use a leaf-blower ([[BM25]] or [[DPR]]) to quickly grab the top 1000 candidates.
> 2. **Second Stage (Reranking)**: Use a magnifying glass ([[Neural Reranking]] / [[MonoBERT]]) to find the best 100 from those.
> 3. **Final Stage (Optional)**: Use a microscope (Heavy LLMs) to pick the perfect top 10.

## Standard Pipeline Structure

| Stage | Model Type | Documents Handled | Speed | Quality |
|-------|------------|-------------------|-------|---------|
| **Retrieval** | [[BM25]], [[DPR]] | Millions $\to$ 1000 | Sub-millisecond | Medium |
| **Reranking** | [[MonoBERT]], [[Cross-Encoder]] | 1000 $\to$ 100 | Deciseconds | High |
| **Fine Reranking**| [[monoT5]], LLMs | 100 $\to$ 10 | Seconds | Maximum |

## Trade-offs: Efficiency vs. Effectiveness

- **Latency**: If the reranker is slow, we must retrieve fewer documents in the first stage.
- **Recall**: If the first stage misses the relevant document, the reranker can never find it.
- **Cost**: Running Transformers on 1000 documents per query is computationally expensive.

## Connections

- Components: [[Neural Reranking]], [[BM25]], [[Dense Retrieval]].
- Models used: [[MonoBERT]], [[DPR]].
- Solves: The problem that complex models are too slow for full-collection search.

## Appears In

- [[IR-L05 - Neural IR]]
