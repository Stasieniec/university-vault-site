---
type: concept
aliases: [GENRE]
course: [IR]
tags: [neural-ir, entity-retrieval, autoregressive]
status: complete
---

# GENRE

> [!definition] GENRE (Generative Entity REtrieval)
> **GENRE** is a system that retrieves entities by generating their unique names (titles) token-by-token in an autoregressive manner using a sequence-to-sequence model (BART).

> [!intuition] Searching by Generating
> Instead of mapping a query to a vector and looking for the closest entity ([[DPR]]), GENRE behaves like a human typing an entity name. If you ask about "The capital of France," the model generates the string `Paris`.

## Key Mechanism: Constrained Beam Search

A vanilla seq2seq model might generate a name that doesn't exist in the database (hallucination). GENRE solves this with **Constrained Beam Search**:
- It uses a **trie** (prefix tree) of all valid entity names in Wikipedia.
- At each generation step, it only allows tokens that form a prefix of a real entity.
- This guarantees that every output corresponds to a valid, existing entity.

## Use Cases
- Entity Linking (linking mentions in text to a KB).
- Slot Filling / Factoid QA.
- Document Retrieval (if documents are identified by unique titles).

## Connections

- Category: Generative Retrieval.
- Kindred to: [[DSI]] (but generates names, not abstract IDs).
- Model: Built on BART architecture.

## Appears In

- [[IR-L08 - Advanced Neural IR]]
