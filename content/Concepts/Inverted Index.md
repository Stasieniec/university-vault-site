---
type: concept
aliases: [inverted index, inverted file, posting list, Inverted Indexing]
course: [IR]
tags: [foundations, exam-topic]
status: complete
---

# Inverted Index

> [!definition] Inverted Index
> An **inverted index** is a data structure that maps terms to the documents (and positions) in which they appear. It is the fundamental data structure enabling efficient text search.

## Structure

```
Term       → Posting List
───────────────────────────
"apple"    → [(doc1, tf=3), (doc5, tf=1), (doc12, tf=2)]
"banana"   → [(doc2, tf=1), (doc5, tf=4)]
"computer" → [(doc1, tf=7), (doc3, tf=2), (doc8, tf=1)]
```

**Components:**
- **Dictionary** (vocabulary/lexicon): sorted list of all unique terms
- **Posting list**: for each term, a list of documents containing it
- **Posting**: a (docID, term_frequency) pair; may include positions for phrase queries

## Construction

1. **Collect** all documents
2. **Tokenize** text → terms
3. **Preprocess**: [[Stemming]], [[Stop Words]] removal, lowercasing
4. **Sort** (term, docID) pairs
5. **Merge** duplicates → posting lists with term frequencies
6. **Store** with compression (variable-byte, gamma codes)

## Why "Inverted"?

A forward index maps: document → terms it contains.
An inverted index maps: term → documents containing it.
"Inverted" because it reverses the natural document→terms mapping.

## Connections

- Prerequisite for: [[BM25]], [[TF-IDF]], [[Query Likelihood Model]]
- Enhanced by: [[Learned Sparse Retrieval]] (neural term weights in inverted index)
- Alternative: [[Dense Retrieval]] (no inverted index — uses ANN search instead)

## Appears In

- [[IR-L02 - IR Fundamentals]]
- [[IR-A01 - Unsupervised Retrieval]]
