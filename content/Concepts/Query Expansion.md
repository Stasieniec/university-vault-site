---
type: concept
aliases: [QE, Pseudo-Relevance Feedback, PRF]
course: [IR]
tags: [neural-ir]
status: complete
---

# Query Expansion

> [!definition] Query Expansion
> **Query Expansion** is the process of reformulating a user's search query to improve retrieval performance by adding related terms. This helps resolve the vocabulary mismatch problem where the user uses different words than the document author.

## Methods

### 1. Relevance Feedback
- **Manual**: The user marks documents as relevant/not relevant; the system updates the query.
- **Pseudo-Relevance Feedback (PRF)**: The system *assumes* the top-$k$ results from the first retrieval pass are relevant and extracts common terms from them to add to a second-pass query.

### 2. Knowledge-Based Expansion
- Uses **Thesauri** (e.g., WordNet) or **Ontologies** to find synonyms, hypernyms, or related concepts.

### 3. Neural Expansion
- Uses Large Language Models (LLMs) to rewrite the query or generate a hypothetical passage that answers the query (e.g., **HyDE**).

> [!intuition] Why it helps
> If you search for "infant," you likely want documents containing "baby" as well. Query expansion adds "baby" automatically.

## Comparison with Document Expansion

| Feature | Query Expansion | [[Document Expansion]] |
|---------|-----------------|--------------------|
| **Timing** | Query time (Latency!) | Indexing time |
| **Context** | Specific to this user query | General to the document |
| **Risk** | **Query Drift**: Adding a wrong word can ruin the search | Bloating the index |

## Connections

- Relates to: [[BM25]] (often the second pass in PRF).
- Foundational: [[Rocchio Algorithm]].
- Modern: Part of [[Retrieval-Augmented Generation]] (RAG) pipelines.

## Appears In

- [[IR-L07 - Neural IR]]
