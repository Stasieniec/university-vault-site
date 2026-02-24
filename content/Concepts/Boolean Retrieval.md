---
type: concept
aliases: [Boolean Retrieval Model, Exact Match]
course: [IR]
tags: [foundations]
status: complete
---

# Boolean Retrieval

> [!definition] Boolean Retrieval
> **Boolean Retrieval** is a retrieval model where documents are represented as sets of terms, and queries are expressed as Boolean expressions of terms using operators like `AND`, `OR`, and `NOT`.

## Mechanism

- **Input**: A query like `(Brutus AND Caesar) AND NOT Cassius`.
- **Representation**: Often visualized using a **Term-Document Incidence Matrix**, where rows represent terms and columns represent documents (1 if present, 0 if absent).
- **Processing**: The system performs bitwise operations on the incidence vectors (or more commonly, intersects posting lists in an [[Inverted Index]]).

> [!intuition] Match vs. Rank
> In Boolean Retrieval, there is no "ranking." A document either matches the boolean condition (stays in the result set) or it doesn't. There is no concept of a document being "more relevant" than another.

## Pros and Cons

| Pros | Cons |
|------|------|
| Precise control for expert users | No relevance ranking (all or nothing) |
| predictable and transparent results | "Feast or famine": too many or zero results |
| Efficient technical implementation | Difficult for non-expert users to write queries |

## Connections

- Data Structure: Usually implemented via an [[Inverted Index]].
- Evolution: Succeeded by ranked retrieval models like [[Vector Space Model]] and [[BM25]].
- Operations: Uses [[Tokenization]] and [[Stemming]] to normalize terms before matching.

## Appears In

- [[IR-L02 - Indexing and Boolean Retrieval]]
