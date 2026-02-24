---
type: concept
aliases: [DocIDs, Document IDs]
course: [IR]
tags: [neural-ir, generative-retrieval]
status: complete
---

# Document Identifiers

> [!definition] Document Identifiers (DocIDs)
> In the context of [[Autoregressive Retrieval]] and [[DSI]], **Document Identifiers** (DocIDs) are the target sequences that a generative model is trained to produce to represent a document. The choice of identifier scheme is critical because it defines how the model "navigates" the document space.

## Identifier Schemes

| Scheme | Description | Pros/Cons |
|--------|-------------|-----------|
| **Atomic IDs** | Each document is assigned a single, unique token (e.g., `doc_742`). | Does not scale; no semantic meaning. |
| **Naive String IDs**| Documents are identified by their content (e.g., the first 10 words or the title). | Good for known-item search; bad for long docs. |
| **Numeric Strings** | Documents get random or sequential numbers (e.g., `1`, `2`, `3`). | Easy to assign; hard for model to learn relationships. |
| **Semantic IDs** | IDs based on a hierarchy (e.g., `10.2.5.1`) created via hierarchical clustering of document embeddings. | **Best performance**; related documents share ID prefixes. |

> [!intuition] The Power of Semantic Structure
> If two documents are about "Deep Learning" and "Neural Networks", a good semantic ID scheme might assign them IDs starting with the same prefix (e.g., `5.1.x`). This helps the model generalize: if the model predicts the first few digits correctly, it has already narrowed the search down to the correct topical "neighborhood."

## Connections
- Core component of: [[DSI]], [[Autoregressive Retrieval]].
- Design inspired by: [[Product Quantization]] and clustering techniques.

## Appears In
- [[IR-L08 - Advanced Neural IR]]
