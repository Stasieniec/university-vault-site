---
type: concept
aliases: [Word Embeddings, distributed representations]
course: [IR, RL]
tags: [foundations, nlp, embeddings]
status: complete
---

# Word Embeddings

> [!definition] Word Embeddings
> **Word Embeddings** are dense, low-dimensional vector representations of words in a continuous vector space where semantically similar words are mapped to nearby points.

> [!intuition] Beyond One-Hot Encoding
> Traditional "one-hot" encodings treat every word as a completely independent dimension (e.g., "cat" and "dog" are as different as "cat" and "refrigerator"). Word embeddings learn that words which appear in similar contexts (like "cat" and "kitten") should have similar vector coordinates. This allows models to generalize from "I saw a cat" to "I saw a feline."

## Major Models

### 1. Word2Vec (2013)
Based on the **Distributional Hypothesis**: "You shall know a word by the company it keeps."
- **CBOW (Continuous Bag of Words)**: Predicts a target word from its context.
- **Skip-gram**: Predicts context words given a target word.

### 2. GloVe (Global Vectors)
Learns embeddings by factorizing a global word-word co-occurrence matrix. Combines global statistics with local window-based methods.

## Properties
- **Linear Relationships**: Famous for capturing analogies: $vec("King") - vec("Man") + vec("Woman") \approx vec("Queen")$.
- **Dense**: Typically 100-300 dimensions, compared to vocabulary sizes of 100k+.

## Connections
- Predecessor to: Contextualized embeddings used in [[Transformers]].
- Compared to: [[TF-IDF]] and [[BM25]], which rely on sparse, lexical matching.
- Use cases: Initializing neural networks for [[Neural Reranking]] or text classification.

## Appears In
- [[IR-L05 - Neural IR 2]]
- [[RL-L08 - Representations]]
