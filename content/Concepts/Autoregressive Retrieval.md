---
type: concept
aliases: [Autoregressive Retrieval, Generative Retrieval]
course: [IR]
tags: [neural-ir, generative-retrieval]
status: complete
---

# Autoregressive Retrieval

> [!definition] Autoregressive Retrieval
> **Autoregressive Retrieval** is the core mechanism behind "generative retrieval" models (like [[DSI]] or [[GENRE]]). Instead of scoring documents in a list, the model retrieves documents by generating their **Identifiers** (DocIDs) token-by-token using an autoregressive decoder.

> [!intuition] Search as Generation
> In traditional search, we *lookup* documents in a database. In autoregressive retrieval, the model *writes* the answer. If the model wants to retrieve document #1234, it first predicts `1`, then `2`, then `3`, then `4`. This allows for a purely differentiable end-to-end retrieval process.

## Constrained Decoding
Since the model could technically generate any sequence of tokens (many of which would not correspond to real documents), **constrained decoding** is used. 
- Usually implemented via a **Trie** (prefix tree) of all valid DocIDs.
- At each step, the model only considers tokens that lead to a valid document path in the Trie.

## Key Examples
- [[DSI]]: Uses hierarchical clusters or numeric strings as DocIDs.
- **GENRE**: Uses Wikipedia entity titles (e.g., "Artificial Intelligence") as identifiers for entity retrieval.

## Connections
- Foundation of: [[DSI]], [[GENRE]].
- Related to: [[Document Identifiers]] — the way identifiers are structured determines the difficulty of the autoregressive task.
- Contrast: [[Dense Retrieval]] (nearest neighbor search in vector space).

## Appears In
- [[IR-L08 - Advanced Neural IR]]
