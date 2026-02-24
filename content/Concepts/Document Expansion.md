---
type: concept
aliases: [Stop Words, Stopword Removal]
course: [IR]
tags: [foundations]
status: complete
---

# Stop Words

> [!definition] Stop Words
> **Stop Words** are high-frequency words that carry very little semantic weight or information for the purpose of distinguishing documents. Examples include "the", "is", "at", "which", and "on".

## Why Remove Them?

- **Index Size**: Stop words can account for 20-30% of the postings in an [[Inverted Index]]. Removing them significantly reduces disk space and memory usage.
- **Efficiency**: Processing queries with "the" is computationally expensive because the posting list for "the" is massive.
- **Relevance**: In vector-based models, high-frequency words can "wash out" the signal from rarer, more meaningful terms.

> [!warning] The Modern Perspective
> Modern Information Retrieval (especially Neural IR and LLM-based systems) often **keeps** stop words. They are crucial for understanding phrases ("To be or not to be"), dependency structures, and local context. For large-scale web search, storage is cheap enough that the benefits of keeping them outweigh the costs.

## Connections

- Preprocessing pipeline: [[Tokenization]] → **Stop Word Removal** → [[Stemming]].
- Term Weighting: [[TF-IDF]] naturally downweights stop words via the IDF component even if they aren't explicitly removed.

## Appears In

- [[IR-L02 - Indexing and Boolean Retrieval]]
