---
type: concept
aliases: [Bag of Words, BoW]
course: [IR]
tags: [foundations]
status: complete
---

# Bag of Words

> [!definition] Bag of Words (BoW)
> The **Bag of Words** model is a simplifying representation used in NLP and Information Retrieval where a text (such as a sentence or a document) is represented as the multiset (bag) of its words, disregarding grammar and word order but keeping multiplicity.

## Characteristics

- **Order Independence**: "John likes Mary" and "Mary likes John" have identical BoW representations.
- **Feature Vector**: Each document becomes a vector where each dimension corresponds to a term in the vocabulary, and the value is the count (or weight) of that term.

> [!intuition] The Multiplicity of Terms
> While it loses "meaning" derived from syntax, it captures "meaning" derived from statistics. If the word "algorithm" appears 20 times in a document, it is highly likely the document is about algorithms.

## Role in IR models

BoW is the fundamental assumption behind almost all classical retrieval models:
- [[Vector Space Model]]: Terms are dimensions in a high-dimensional space.
- [[TF-IDF]]: Weights the "bag" entries by relative importance.
- [[BM25]]: Uses term frequencies within the bag for probabilistic ranking.

## Connections

- Preprocessing: Depends on [[Tokenization]], [[Stemming]], and [[Stop Words]].
- Contrast: Overcome by sequence-aware models like [[Transformers]] or [[Word Embeddings]].

## Appears In

- [[IR-L02 - Indexing and Boolean Retrieval]]
