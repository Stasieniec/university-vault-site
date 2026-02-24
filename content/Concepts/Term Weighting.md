---
type: concept
aliases: [Term Weighting]
course: [IR]
tags: [foundations]
status: complete
---

# Term Weighting

> [!definition] Term Weighting
> **Term Weighting** is the process of assigning a numerical value to a term in a document or query to represent its importance for retrieval. In effective weighting schemes, terms that are descriptive of the document's content receive high weights, while common or "noisy" terms receive low weights.

> [!formula] Classic TF-IDF Weighting
> The most fundamental weighting scheme is **TF-IDF**:
> 
> $$w(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$
> 
> where:
> - $\text{TF}(t, d)$ (Term Frequency): How many times term $t$ appears in document $d$.
> - $\text{IDF}(t)$ (Inverse Document Frequency): $\log \frac{N}{n_t}$, where $N$ is the total number of docs and $n_t$ is the number of docs containing $t$.

> [!intuition] The Local vs. Global Balance
> Good weighting balances two signals:
> 1. **Local importance (TF)**: If a word appears many times in *this* document, it likely describes what this document is about.
> 2. **Global rarity (IDF)**: If a word appears in *every* document (like "the" or "is"), it's useless for distinguishing between them. We want words that are specific to a few documents.

## Main Weighting Components

| Method | Intuition |
|--------|-----------|
| **TF (Term Frequency)** | More occurrences = more relevance. |
| **IDF (Inv. Doc Frequency)** | Rare words are better signals than common ones. |
| **Length Normalization** | Prevents long documents from winning just by having more words. |
| **Learned Weights** | Modern neural methods (like [[DeepCT]]) use [[BERT for IR]] to predict weights based on context. |

## Connections

- **Core models**: [[BM25]] (standard modern weighting), [[TF-IDF]] (classic)
- **Neural variants**: [[DeepCT]], [[DeepImpact]], [[uniCOIL]]
- **Usage**: Fundamental step in building an Inverted Index.

## Appears In

- [[IR-L02 - Indexing and Basic Models]]
- [[IR-L03 - Retrieval Models]]
