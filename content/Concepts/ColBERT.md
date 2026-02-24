---
type: concept
aliases: [ColBERT, colbert, contextualized late interaction]
course: [IR]
tags: [neural-ir, exam-topic]
status: complete
---

# ColBERT

> [!definition] ColBERT
> **ColBERT** (Contextualized Late Interaction over BERT) is a retrieval model that combines the efficiency of [[Bi-Encoder]] (pre-computed document representations) with fine-grained token-level matching via **late interaction**.

> [!formula] MaxSim Scoring
> $$s(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \mathbf{q}_i^\top \mathbf{d}_j$$
> 
> For each query token, find its best-matching document token (max similarity), then sum across all query tokens.

**Key properties:**
- Documents encoded independently → can be pre-computed and indexed
- Token-level representations (not single-vector) → richer matching than standard bi-encoder
- Uses [[Approximate Nearest Neighbor]] search with deferred interaction
- Much faster than cross-encoder, more effective than single-vector bi-encoder

## Appears In

- [[IR-L06 - Dense Retrieval]]
