---
type: concept
aliases: [NDCG, nDCG, normalized discounted cumulative gain]
course: [IR]
tags: [evaluation, key-formula, exam-topic]
status: complete
---

# Normalized Discounted Cumulative Gain (NDCG)

> [!definition] NDCG
> **NDCG** is an evaluation metric for ranked retrieval that handles **graded relevance** judgments (not just binary relevant/non-relevant). It measures how well the ranking places highly relevant documents at the top.

> [!formula] DCG and NDCG
> $$\text{DCG}@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$
> 
> $$\text{NDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}$$
> 
> where:
> - $rel_i$ — relevance grade of the document at position $i$
> - $\text{IDCG}@k$ — ideal DCG (DCG of the perfect ranking)
> - NDCG ∈ [0, 1], where 1 = perfect ranking

> [!intuition] What It Captures
> - **Graded relevance**: A doc with relevance 3 is worth more than relevance 1 (exponential gain: $2^{rel} - 1$)
> - **Position discount**: Higher-ranked results matter more ($1/\log_2(i+1)$ decay)
> - **Normalization**: Dividing by IDCG makes scores comparable across queries

## Comparison with Other Metrics

| Metric | Graded? | Position-aware? | Normalized? |
|--------|---------|-----------------|-------------|
| [[Precision]] | ❌ Binary | ❌ | ❌ |
| [[Precision at K]] | ❌ Binary | Partially (cutoff) | ❌ |
| [[MAP]] | ❌ Binary | ✅ | ✅ |
| **NDCG** | ✅ Graded | ✅ | ✅ |

## Appears In

- [[IR-L04 - Evaluation]]
