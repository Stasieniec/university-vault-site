---
type: concept
aliases: [DeepCT, Context-Aware Term Weighting]
course: [IR]
tags: [neural-ir]
status: complete
---

# DeepCT

> [!definition] DeepCT (Deep Contextualized Term Weighting)
> **DeepCT** is a methodology that utilizes BERT to estimate context-aware term importance (weighting) for documents and queries. It maps BERT's contextual embeddings to a single scalar importance score for each term, which can then be used to replace or augment traditional Term Frequency (TF) in a standard inverted index.

> [!formula] DeepCT Weighting
> For a document $d$, DeepCT predicts a weight $w_t^{(d)}$ for each term $t$:
> $$w_t^{(d)} = \text{round}(v_t \cdot N)$$
> 
> where:
> - $v_t$ is the predicted importance score from BERT ($v_t \in [0, 1]$)
> - $N$ is a scaling factor to convert scores into integer frequencies (often $N=100$)
> - These predicted scores replace the raw $f(t, d)$ in the [[BM25]] formula.

> [!intuition] Context over Count
> Traditional IR relies on raw Term Frequency (counting occurrences). DeepCT recognizes that a term appearing once in a critical, descriptive sentence (e.g., a title or summary) is often more important than a term appearing multiple times in tangential contexts. By using BERT, DeepCT "sees" the context and predicts how much a term actually contributes to the document's meaning.

## Key Features

- **Inverted Index Compatibility**: Because it produces integer weights, the output can be stored in any standard search engine (like Lucene/Elasticsearch) without changing the retrieval architecture.
- **Improved Recall/Precision**: Better identifies central terms while down-weighting "stop-word-like" occurrences of common terms in specific contexts.
- **Efficiency**: Expensive neural computation is done offline during indexing; retrieval remains as fast as standard [[BM25]].

## Connections

- **Foundation**: [[BERT for IR]]
- **Input for**: [[BM25]] (replaces raw TF)
- **Related**: [[uniCOIL]] (evolution of sparse weights), [[DeepImpact]]

## Appears In

- [[IR-L07 - Learned Sparse Retrieval]]
