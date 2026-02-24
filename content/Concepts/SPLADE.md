---
type: concept
aliases: [SPLADE, splade]
course: [IR]
tags: [neural-ir, exam-topic]
status: complete
---

# SPLADE

> [!definition] SPLADE
> **SPLADE** (SParse Lexical AnD Expansion model) is a [[Learned Sparse Retrieval]] model that produces sparse term-weight vectors using transformer-based models. It combines the efficiency of inverted indexes with the effectiveness of neural representations.

> [!formula] SPLADE Term Weight
> $$w_j = \sum_{i=1}^{|T|} \log(1 + \text{ReLU}(h_{ij}))$$
> 
> where $h_{ij}$ is the logit for vocabulary token $j$ at position $i$. The $\log(1+\text{ReLU}(\cdot))$ ensures non-negative, sparse weights.

**Key features:**
- Produces sparse vectors over the full vocabulary → stored in inverted index
- **Expansion**: Can assign non-zero weights to terms NOT in the original text (learned query/document expansion)
- **FLOPS regularization**: Controls sparsity by penalizing expected number of floating point operations at retrieval time
- Competitive with [[Dense Retrieval]] while using standard inverted indexes

## Appears In

- [[IR-L07 - Learned Sparse Retrieval]]
