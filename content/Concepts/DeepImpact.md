---
type: concept
aliases: [DeepImpact]
course: [IR]
tags: [neural-ir]
status: complete
---

# DeepImpact

> [!definition] DeepImpact
> **DeepImpact** is a neural retrieval model that predicts "impact scores" for terms in a document. It combines document expansion (using [[DocT5Query]]) and contextual term weighting (using BERT) to produce a sparse representation where the weight directly represents the term's contribution to retrieval.

> [!formula] Impact Scoring
> DeepImpact bypasses traditional [[BM25]] components (like TF, IDF, and length normalization) by directly learning an end-to-end impact score $I(t, d)$:
> 
> $$\text{Score}(q, d) = \sum_{t \in q \cap d} I(t, d)$$
> 
> These impact scores are typically computed by:
> 1. Expanding the document $d$ using [[DocT5Query]].
> 2. Passing the expanded document through a BERT-based model to predict a score $I(t, d)$ for every term $t$.

> [!intuition] Learning Significance
> Instead of using heuristics (like "longer documents get penalized"), DeepImpact asks a neural model: "If someone searches for word $X$, how relevant is this document?" The model learns to assign high scores to terms that appear in the document (or were added via expansion) that are likely to be useful query terms.

## Key Features

- **Direct Sparse Retrieval**: The scores are stored in a standard inverted index. At query time, simple summation of these scores (found in the posting lists) provides the final ranking.
- **Deep Document Expansion**: Utilizes [[DocT5Query]] to handle vocabulary mismatch by adding potential query terms that were not originally in the document.
- **Efficiency**: Neural processing is performed at index time. Retrieval is exceptionally fast as it only involves pointer traversal and addition.

## Connections

- **Predecessors**: [[DocT5Query]], [[DeepCT]], [[BERT for IR]]
- **Successors**: [[uniCOIL]] (which optimizes the process)
- **Comparison**: Directly competes with [[BM25]] in speed while outperforming it in accuracy.

## Appears In

- [[IR-L07 - Learned Sparse Retrieval]]
