---
type: concept
aliases: [Learning to Rank, L2R, LTR]
course: [IR]
tags: [neural-ir]
status: complete
---

# Learning to Rank

> [!definition] Learning to Rank (LTR)
> **Learning to Rank** is an application of Machine Learning to the problem of ranking. Instead of using a fixed heuristic like [[BM25]], LTR models are trained on historical relevance data (e.g., click logs or human judgments) to automatically learn an optimal ranking function that combines hundreds of different features.

> [!formula] The Three Approaches
> 1. **Pointwise**: Treats ranking as a regression/classification problem. Predicts a score $s$ for a single $(q, d)$ pair.
>    - Loss: $\mathcal{L}(y, f(q, d))$ (e.g., MSE)
> 2. **Pairwise**: Focuses on the relative order of pairs of documents. Given $d_i$ and $d_j$, the model learns if $d_i \succ d_j$.
>    - Loss: $\mathcal{L}(f(q, d_i), f(q, d_j), y_{ij})$ (e.g., RankNet)
> 3. **Listwise**: Optimizes the entire ranked list directly using IR metrics like [[NDCG]] or MAP.
>    - Loss: $\mathcal{L}(\text{ranked\_list}, \text{ground\_truth})$ (e.g., LambdaMART)

> [!intuition] Beyond One Feature
> Traditional IR is like a one-string guitar (mostly using word overlap). LTR is an orchestra. It can use BM25 scores, but also PageRank, document age, user location, font size of headings, and URL length. The ML model learns the perfect "recipe" to combine these features for the best user experience.

## Key Algorithms

- **LambdaMART**: A Gradient Boosted Decision Tree (GBDT) approach that is widely considered the state-of-the-art for feature-based LTR.
- **RankNet**: An early neural pairwise approach.
- **LambdaLoss**: A generalized framework for optimizing listwise metrics.

## Connections

- **Usage**: Typically used as a **reranker** (Stage 2) after a fast model like [[BM25]] (Stage 1) suggests candidates.
- **Features**: Uses scores from [[BM25]], [[Vector Space Model]], and others as inputs.
- **Modern Context**: Often integrated with [[BERT for IR]] in multi-stage pipelines.

## Appears In

- [[IR-L05 - Learning to Rank]]
