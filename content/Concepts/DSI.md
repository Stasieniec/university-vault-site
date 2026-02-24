---
type: concept
aliases: [DSI, Differentiable Search Index]
course: [IR]
tags: [neural-ir, generative-retrieval]
status: complete
---

# DSI

> [!definition] DSI (Differentiable Search Index)
> **DSI** is a "generative retrieval" paradigm where a single neural model (e.g., T5) maps a query directly to a **Document ID** (docid). The model itself acts as the search index.

> [!intuition] Model as Index
> In traditional IR ([[BM25]], [[DPR]]), the index is a separate data structure (inverted index or vector store). In DSI, the "index" is stored in the **model parameters**. Retrieval is simply sequence-to-sequence generation:
> $$Query \xrightarrow{Model} DocID$$

## Phases in DSI

1. **Indexing Phase**: The model is trained to memorize the mapping: $Document\_Content \to DocID$.
2. **Retrieval Phase**: At inference time, given a $Query$, the model generates the most likely $DocID$.

## Identifier Design (DocIDs)
The choice of DocID is critical for performance:
- **Unstructured IDs**: Arbitrary integer for each document (hard for the model to learn).
- **Semantic IDs**: Hierarchical IDs (e.g., `1.4.2`) derived from clustering document content. This allows the model to predict the "category" first.

### Pros and Cons
- **Pros**: End-to-end differentiable; no separate ANN index required.
- **Cons**: Difficult to update (adding a new document requires retraining/fine-tuning); limited to smaller collections.

## Connections

- Category: Generative Retrieval.
- Comparison: Different from [[DPR]] (search-by-embedding) or [[GENRE]] (search-by-entity-name).
- Components: Uses hierarchical clustering for DocIDs.

## Appears In

- [[IR-L08 - Advanced Neural IR]]
