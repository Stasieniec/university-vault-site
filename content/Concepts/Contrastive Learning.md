---
type: concept
aliases: [Contrastive Learning]
course: [IR, RL]
tags: [neural-ir, training, deep-learning, representation-learning]
status: complete
---

# Contrastive Learning

> [!definition] Contrastive Learning
> **Contrastive Learning** is a training paradigm that learns representations by encouraging positive pairs to be close together in the embedding space and negative pairs to be far apart.

> [!formula] InfoNCE Loss
> For a query $q$, a positive document $d^+$, and a set of negative documents $\{d^-_1, ..., d^-_n\}$, the contrastive loss is often defined as:
> $$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, d^+)/\tau)}{\exp(\text{sim}(q, d^+)/\tau) + \sum_{i=1}^n \exp(\text{sim}(q, d^-_i)/\tau)}$$
> 
> where:
> - $\text{sim}(\cdot, \cdot)$ — A similarity metric (e.g., dot product or cosine similarity)
> - $\tau$ — Temperature parameter scaling the distribution

## In Information Retrieval

Contrastive learning is the foundation of **Dense Retrieval** models like [[DPR]].
- **Positive Pair**: (Query, Relevant Document).
- **Negative Pair**: (Query, Irrelevant Document).

### Negative Strategies
The choice of negatives is the most important factor in contrastive training:
- **Random Negatives**: Documents sampled randomly from the collection (too easy).
- **In-batch Negatives**: Efficiently using other positive documents in the current training batch as negatives.
- **Hard Negative Mining**: Deliberately selecting documents that "look" relevant but aren't (e.g., high BM25 score but labeled irrelevant).

## Connections

- Main application: Training [[DPR]] bi-encoders.
- Optimization technique: [[Hard Negative Mining]].
- Metric: Often evaluated using MRR or Recall@k.

## Appears In

- [[IR-L06 - Dense Retrieval]]
