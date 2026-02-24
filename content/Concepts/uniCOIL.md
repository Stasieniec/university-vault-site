---
type: concept
aliases: [uniCOIL]
course: [IR]
tags: [neural-ir]
status: complete
---

# uniCOIL

> [!definition] uniCOIL
> **uniCOIL** is a learned sparse retrieval model that produces a single-vector representation of a document where each dimension corresponds to a term in the vocabulary. It simplifies earlier models like [[DeepImpact]] and [[COIL]] by assigning a single importance weight to each term (token) using BERT.

> [!formula] uniCOIL Retrieval
> For a query $q$ and document $d$, the relevance score is the inner product of their sparse weight vectors:
> 
> $$\text{Score}(q, d) = \sum_{t \in q \cap d} w_t^{(q)} \cdot w_t^{(d)}$$
> 
> where:
> - $w_t^{(d)}$ is the weight of term $t$ in document $d$, predicted by a BERT encoder $(w_t \geq 0)$.
> - Typically, $w_t^{(q)}$ is simplified to a binary indicator (1 if term is in query, 0 otherwise).

> [!intuition] Sparse Neural Weights
> Modern neural IR often uses "dense" vectors (long lists of numbers) that are hard to search quickly. uniCOIL keeps the "sparse" nature of classic IR (words in an index) but uses BERT to decide the "volume" of each word. If a word is very important in a sentence, it gets a loud volume (high weight); if it's just filler, it gets silenced (zero weight).

## Key Features

- **Simplification**: Unlike [[COIL]] (which stores multiple vectors per term), uniCOIL uses one weight per term, making it compatible with standard inverted indexes like Lucene.
- **Effective Expansion**: When combined with [[DocT5Query]], it can assign weights to terms that weren't in the original text but are relevant to the topic.
- **Extreme Speed**: Because it calculates a simple sum of weights during retrieval, it is as efficient as [[BM25]] but with the semantic understanding of [[BERT for IR]].

## Connections

- **Related Models**: [[DeepCT]], [[DeepImpact]], [[SPLADE]]
- **Underlying Tech**: [[BERT for IR]], [[DocT5Query]]
- **Opposite Approach**: [[Dense Retrieval]] (e.g., DPR).

## Appears In

- [[IR-L07 - Learned Sparse Retrieval]]
