---
type: concept
aliases: [DPR, Dense Passage Retrieval, Dual-Encoder]
course: [IR]
tags: [neural-ir, dense-retrieval, embedding-search]
status: complete
---

# DPR

> [!definition] DPR (Dense Passage Retrieval)
> **DPR** is a retrieval model that maps queries and documents into a shared dense vector space using BERT-based bi-encoders. Retrieval is performed using maximum inner product search (MIPS) on these embeddings.

> [!formula] Scoring via Dual Encoders
> The relevance score $s(q, d)$ is the dot product of two independently computed embeddings:
> $$s(q, d) = E_Q(q)^\top E_D(d)$$
> 
> where:
> - $E_Q(q)$ — Dense representation of the query (Query Encoder)
> - $E_D(d)$ — Dense representation of the document (Document Encoder)
> - Both encoders are typically BERT-based.

## Key Components

1. **Bi-Encoder Architecture**: Unlike [[Cross-Encoder|Cross-Encoders]], the query and document do not see each other during encoding.
2. **Indexing**: Document embeddings $E_D(d)$ are pre-computed and stored in an [[Approximate Nearest Neighbor|ANN]] index (e.g., **FAISS**).
3. **Training**: Uses **Contrastive Learning** with an InfoNCE-like objective.
4. **Negative Sampling**: Crucially uses **In-batch Negatives** — for a batch of $B$ queries and their relevant documents, the other $B-1$ documents in the batch serve as negatives for each query.

> [!intuition] Semantic Search
> Because it uses dense vectors, DPR can find documents that are topically relevant but don't share any words with the query (solving the "lexical mismatch" problem of [[BM25]]).

## Connections

- Trained via: [[Contrastive Learning]], [[Hard Negative Mining]].
- Efficiency: Uses [[Approximate Nearest Neighbor]] for retrieval.
- Role: Often the first stage in [[Multi-Stage Ranking]].
- Comparison: Faster than [[MonoBERT]] (scoring is just a dot product).

## Appears In

- [[IR-L06 - Dense Retrieval]]
