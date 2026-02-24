---
type: concept
aliases: [IR, information retrieval]
course: [IR]
tags: [foundations]
status: complete
---

# Information Retrieval

> [!definition] Information Retrieval
> **Information Retrieval (IR)** is the activity of obtaining information system resources that are relevant to an information need from a collection of those resources. It involves finding material (usually documents) of an unstructured nature (usually text) that satisfies an information need from within large collections.

## Core Components

1. **Document collection** (corpus): The set of documents to search
2. **Query**: A user's expression of their information need
3. **Relevance**: The degree to which a document satisfies the information need
4. **Ranking function**: Scores and orders documents by estimated relevance

## IR Pipeline

```
Query → [Query Processing] → [Matching/Retrieval] → [Ranking] → Ranked Results
                                     ↑
                              [Inverted Index]
                                     ↑
                     Documents → [Indexing] → [Text Processing]
```

## Retrieval Paradigms

| Paradigm | Example | How it works |
|----------|---------|-------------|
| **Sparse retrieval** | [[BM25]], [[TF-IDF]] | Exact term matching via [[Inverted Index]] |
| **Dense retrieval** | [[DPR]], [[ColBERT]] | Semantic matching via learned embeddings |
| **Learned sparse** | [[SPLADE]] | Neural term weights in inverted index |
| **Generative** | [[DSI]], [[GENRE]] | Generate document IDs directly |
| **Reranking** | [[MonoBERT]] | Re-score top-k from first-stage retrieval |

## Appears In

- [[IR-L02 - IR Fundamentals]]
- All IR lectures
