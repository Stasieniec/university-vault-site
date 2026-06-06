---
type: concept
aliases: []
course: [IR]
tags: [neural-ir, document-expansion]
status: complete
---

# Document Expansion

## Definition

> [!definition] Document Expansion
> **Document expansion** enriches a document's representation with additional terms (or term
> weights) **at index time**, before retrieval, to reduce the **vocabulary mismatch** between
> how authors write documents and how users phrase queries. The expanded document is then
> indexed by a standard retriever (e.g. [[BM25]]), so the cost is paid once during indexing and
> queries stay fast.

## Intuition

> [!intuition] Meet the query halfway
> Lexical retrieval rewards exact term overlap, but a relevant document may simply never use the
> user's words ("heart attack" vs "myocardial infarction"). Instead of expanding the *query* at
> search time ([[Query Expansion]]), document expansion expands the *document* ahead of time —
> predicting the words and questions a document could answer and folding them into its index
> entry. Because it happens offline, it adds no query-time latency.

## Mathematical Formulation

There is no single equation; the operation augments the term set / term frequencies used by the
scorer. For a generative expansion that appends predicted queries $\{\hat{q}_1,\dots,\hat{q}_N\}$
to document $d$:

$$d' = d \,\Vert\, \hat{q}_1 \,\Vert\, \cdots \,\Vert\, \hat{q}_N, \qquad
\text{score}(q,d) = \text{BM25}(q, d')$$

where:
- $d'$ — the expanded document fed to the index
- $\hat{q}_i$ — a synthetic query generated from $d$ (raising the term frequencies of useful, possibly novel terms)
- $\Vert$ — concatenation

For reweighting-based expansion the term set is unchanged but the per-term weights $w(t,d)$ are
learned/predicted rather than taken from raw counts.

## Key Properties / Variants

- **Generative (term addition):** [[DocT5Query]] / doc2query uses a seq2seq model (T5) to
  generate likely queries per document and appends them, adding *new* vocabulary.
- **Reweighting (no new terms):** [[DeepCT]] and [[DeepImpact]] predict context-aware term
  weights for the terms already present — a bridge toward [[Learned Sparse Retrieval]].
- **Index-time vs query-time:** complementary to [[Query Expansion]] (which augments the query).
- **Backbone-agnostic:** the expanded index is still a sparse [[Inverted Index]], so it inherits
  the efficiency and interpretability of lexical retrieval while recovering some semantic recall.

## Connections

- Instances: [[DocT5Query]], [[DeepCT]], [[DeepImpact]]
- Contrast with: [[Query Expansion]] (query-side)
- Leads toward: [[Learned Sparse Retrieval]], [[SPLADE]]
- Improves: [[BM25]] recall by mitigating vocabulary mismatch

## Appears In

- [[Query Expansion]]
- [[Learned Sparse Retrieval]]
- [[IR-L07 - Learned Sparse Retrieval]]
- [[IR-PTR Ch4 - Refining Query and Document Representations]]
