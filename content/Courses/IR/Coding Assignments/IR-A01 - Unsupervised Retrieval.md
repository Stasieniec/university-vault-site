---
type: coding-assignment
course: IR
week: 2
language: python
concepts:
  - "[[TF-IDF]]"
  - "[[BM25]]"
  - "[[Query Likelihood Model]]"
  - "[[Inverted Index]]"
  - "[[Tokenization]]"
  - "[[Stemming]]"
  - "[[Precision]]"
  - "[[Recall]]"
  - "[[MAP]]"
  - "[[NDCG]]"
status: complete
---

# IR-A01: Assignment 1 — Unsupervised Retrieval

## Overview

Implement term-based matching approaches and evaluation metrics. Done in pairs, PASS/FAIL (≥80% tests).

## What You Implement

### Text Preprocessing
- **Tokenization**: Split text into terms
- **Lowercasing**, **stop word removal**, **[[Stemming]]** (NLTK)

### Indexing
- Build an [[Inverted Index]] with term frequencies and document lengths

### Retrieval Methods
1. **[[TF-IDF]] Search**: TF-IDF weighted cosine similarity
2. **[[BM25]] Search**: Okapi BM25 with $k_1$ and $b$ parameters
3. **[[Query Likelihood Model|QL]] Search**: Query likelihood with Dirichlet smoothing
4. **NaiveQL Search**: QL without smoothing (for comparison)

### Evaluation
- Implement [[Precision]], [[Recall]], [[MAP]], [[NDCG]], [[MRR]]
- Evaluate all retrieval methods and compare results

## Key Implementation Notes

- Only allowed: `nltk`, `numpy`, `matplotlib` (no sklearn, gensim)
- All implementation goes in `modules/` directory, between `BEGIN/END SOLUTION` tags
- Helper methods in docstrings are also tested — implement them
- MS MARCO dataset for benchmarking

## Resources

SEIRiP Sections: 2.3, 4.1-4.3, 5.3, 5.6-5.7, 6.2, 7, 8

## Related Lectures

- [[IR-L02 - IR Fundamentals]] (indexing, preprocessing)
- [[IR-L03 - Retrieval Models]] (BM25, QL, TF-IDF)
- [[IR-L04 - Evaluation]] (metrics)
