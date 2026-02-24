---
type: concept
aliases: [DocT5Query, doc2query, doc2query--]
course: [IR]
tags: [neural-ir, document-expansion]
status: complete
---

# DocT5Query

> [!definition] DocT5Query
> **DocT5Query** (also known as doc2query--) is a document expansion method that uses a sequence-to-sequence model (specifically T5) to generate synthetic queries for a document. These generated queries are appended to the document text before indexing with a standard sparse retriever like [[BM25]].

> [!intuition] Bridging the Vocabulary Gap
> Traditional lexical search requires exact term overlap. DocT5Query predicts what questions a document *might* answer. By adding these predicted questions (and their vocabulary) to the document, the model helps bridge the gap between user queries and document language, effectively performing "expansion at index time."

## Key Mechanism
1. **Training**: A T5 model is trained on a dataset of (Query, Document) pairs to predict the query given the document.
2. **Generation**: For every document in the collection, the model generates $N$ (e.g., 40-80) synthetic queries.
3. **Appending**: The generated queries are concatenated to the original document text.
4. **Indexing**: The expanded documents are indexed using a traditional inverted index (e.g., Lucene, Pyserini).

## Connections
- Extends: [[BM25]] by adding semantic context via expansion.
- Comparison: Unlike [[DeepCT]], which re-weights existing terms, DocT5Query adds *new* terms.
- Complementary to: [[Neural Reranking]], where DocT5Query provides a better set of initial candidates.

## Appears In
- [[IR-L07 - Neural IR 1]]
