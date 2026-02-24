---
type: concept
aliases: [Pooling, Depth-k Pooling]
course: [IR]
tags: [evaluation]
status: complete
---

# Pooling

> [!definition] Pooling
> **Pooling** is a technique used to create relevance judgments (Qrels) for large-scale document collections without judging every single document. It involves taking the top-$k$ results from multiple diverse retrieval systems and forming a "pool" for human judges to evaluate.

## The Process

1. **Submit Queries**: Run the same set of queries through $N$ different retrieval systems.
2. **Collect Top-k**: Take the top $k$ (e.g., $k=100$) documents returned by each system.
3. **Union**: Create a unique set of documents from the union of these results.
4. **Judge**: Human annotators judge only the documents in this union.
5. **Assumption**: Any document *not* in the pool is assumed to be **not relevant**.

> [!intuition] Why it works
> It is impossible to judge millions of documents for every query. Pooling assumes that if a document is highly relevant, at least one of the many retrieval systems will find it and place it in its top-$k$ list.

## Evaluation and Bias

- **Unjudged = Non-relevant**: This is the core assumption. If a new system finds a relevant document that wasn't in the original pool, that system might be unfairly penalized (its precision will look lower than it is).
- **Leave-One-Out (LOO) Tests**: Used to check if the pool is "fair" by removing one system's contributions and seeing if the rankings of other systems change.

## Connections

- Necessary for: [[Cranfield Paradigm]] in large collections like TREC.
- Relates to: [[Precision and Recall]], [[MAP]] (which are calculated based on these judgments).

## Appears In

- [[IR-L04 - Evaluation]]
