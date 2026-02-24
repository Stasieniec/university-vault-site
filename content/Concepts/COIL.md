---
type: concept
aliases: [COIL, Contextualized Inverted List]
course: [IR]
tags: [neural-ir, learned-sparse-retrieval]
status: complete
---

# COIL

> [!definition] COIL (Contextualized Inverted List)
> **COIL** is a retrieval model that combines the efficiency of exact lexical matching with the power of contextualized token representations. It stores low-dimensional BERT-based token embeddings directly in an inverted index, allowing for "matching by embedding" within the framework of traditional sparse retrieval.

> [!intuition] The Bridge Between Sparse and Dense
> - **Sparse (BM25)**: Matches exact tokens, but forgets context (polysemy).
> - **Dense (DPR)**: Captures global context, but might miss exact keyword matches.
> - **COIL**: Matches the same token (lexical) but checks if they have similar embeddings (contextual). It asks: "Is the word 'bank' in the query used in the same sense as 'bank' in the document?"

## Key Mechanism
1. **Encoding**: Each token in the corpus is encoded by a BERT-like transformer into a contextual vector.
2. **Indexing**: The inverted index stores entries in the form: `term -> (doc_id, vector)`.
3. **Retrieval**: At query time, the system finds documents containing the query terms and computes the similarity (e.g., dot product) between query token vectors and document token vectors.

## Connections
- Comparison: A more efficient alternative to [[ColBERT]], which uses late interaction across all tokens.
- Category: [[Learned Sparse Retrieval]].
- Successors: Influenced models like [[SPLADE]] which further sparsify the representation.

## Appears In
- [[IR-L07 - Neural IR 1]]
