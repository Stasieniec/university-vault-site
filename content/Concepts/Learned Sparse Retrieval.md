---
type: concept
aliases: [learned sparse retrieval, LSR, neural sparse retrieval]
course: [IR]
tags: [neural-ir, exam-topic]
status: complete
---

# Learned Sparse Retrieval

> [!definition] Learned Sparse Retrieval
> **Learned Sparse Retrieval (LSR)** uses neural models to produce sparse document/query representations that can be stored in and searched with traditional [[Inverted Index]] infrastructure, combining neural effectiveness with sparse retrieval efficiency.

## Approaches

| Method | Type | Key Idea |
|--------|------|----------|
| doc2query / docTTTTTquery | [[Document Expansion]] | Generate expansion terms with seq2seq model |
| [[DeepCT]] | Term reweighting | Predict context-aware term importance |
| [[DeepImpact]] | Term reweighting | Predict impact scores per term |
| [[uniCOIL]] | Token weights | Single-vector per-token importance |
| [[SPLADE]] | Full vocabulary | Log-saturated weights over entire vocabulary |

## Why LSR?

> [!intuition] Best of Both Worlds
> - **Like BM25**: Uses inverted index → fast retrieval, scalable
> - **Like dense retrieval**: Learns semantic term importance from data
> - **Plus expansion**: Can add semantically related terms not in the original text

## Appears In

- [[IR-L07 - Learned Sparse Retrieval]]
