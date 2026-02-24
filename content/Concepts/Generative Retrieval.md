---
type: concept
aliases: [generative retrieval, generative IR]
course: [IR]
tags: [neural-ir, exam-topic]
status: complete
---

# Generative Retrieval

> [!definition] Generative Retrieval
> **Generative retrieval** replaces the traditional index-then-rank pipeline with a single model that directly generates document identifiers given a query. The model **memorizes** the corpus during training and **retrieves** by generating relevant doc IDs autoregressively.

## Key Idea

```
Traditional: Query → [Index Search] → [Rerank] → Doc IDs
Generative:  Query → [Seq2Seq Model] → Doc IDs directly
```

The model parameters serve as the "index" — no separate data structure needed.

## Notable Models

| Model | Approach |
|-------|----------|
| [[DSI]] (Differentiable Search Index) | Encode docs into model params; generate hierarchical doc IDs |
| [[GENRE]] | Autoregressive entity retrieval; generate entity names directly |
| SEAL | Generate n-grams, then map to documents |

## Challenges

- **Scalability**: Hard to scale to millions of documents
- **Corpus updates**: Adding new documents requires retraining
- **Doc ID design**: Choice of ID scheme affects performance significantly (atomic, string, semantic, hierarchical)

## Appears In

- [[IR-L08 - Generative Retrieval]]
