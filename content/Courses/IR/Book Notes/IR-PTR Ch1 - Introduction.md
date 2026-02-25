---
type: book-chapter
course: IR
book: "Pretrained Transformers for Text Ranking: BERT and Beyond"
chapter: 1
sections: ["1.1", "1.2", "1.3"]
topics:
  - "[[Information Retrieval]]"
  - "[[Neural Reranking]]"
  - "[[Transformers]]"
  - "[[BERT for IR]]"
  - "[[Multi-Stage Ranking]]"
  - "[[Dense Retrieval]]"
status: complete
---

# IR-PTR Chapter 1: Introduction

## Overview
Chapter 1 sets the stage for the book, defining the **text ranking** problem and situating it within the broader field of [[Information Retrieval]] (IR) and Natural Language Processing (NLP). It highlights the "paradigm shift" caused by [[Transformers]], specifically **BERT**, in improving search quality for both academic benchmarks and industrial systems like Google and Bing.

## Text Ranking Problems
The authors argue that text ranking is ubiquitous and appears in several forms beyond the standard "ten blue links" search (ad hoc retrieval):

- **[[Information Retrieval]] (Ad Hoc Retrieval)**: Sorting a corpus by estimated relevance to a query.
- **Question Answering (QA)**: Identifying specific spans of text that answer a query (retriever-reader framework).
- **Community Question Answering (CQA)**: Ranking previously asked questions based on similarity to a new user query.
- **Information Filtering**: Matching a static query against a stream of incoming texts (e.g., push notifications).
- **Text Recommendation**: Suggesting similar or related articles/scientific papers.
- **NLP Tasks**: Entity linking, fact verification, and data augmentation (e.g., finding good training examples via ranking).

## Brief History of Text Ranking

### 1. The Exact Match Era
The foundation of IR was built on **exact term matching**.
- **Early Days**: Transition from manual human indexing to automatic content analysis (Luhn, SMART system).
- **Vector Space Model (VSM)**: Documents and queries are "bags of words" in sparse vectors.
- **[[BM25]]**: The dominant exact-match scoring function based on probabilistic retrieval.

> [!definition] Okapi BM25 Formula
> The relevance score $S$ for a document $d$ and query $q$ is:
> $$BM25(q, d) = \sum_{t \in q \cap d} \text{idf}(t) \cdot \frac{tf(t, d) \cdot (k_1 + 1)}{tf(t, d) + k_1 \cdot (1 - b + b \cdot \frac{l_d}{L})}$$
> Where:
> - $idf(t)$: Inverse Document Frequency
> - $tf(t, d)$: Term frequency in document
> - $l_d, L$: Document length and average length
> - $k_1, b$: Free parameters

- **The Vocabulary Mismatch Problem**: Exact match fails when different words describe the same concept (e.g., "star-crossed lovers" vs. "tragic love story").

### 2. [[Learning to Rank]] (LTR)
Supervised machine learning using hand-crafted features (statistical properties, anchor text, PageRank).
- Models like **RankNet** and **LambdaMART** (gradient-boosted decision trees) became the state-of-the-art.
- **Limitation**: Still relies heavily on manual feature engineering.

### 3. Deep Learning (Pre-BERT)
Neural networks promised to replace hand-crafted features with learned representations.
- **Representation-based Models**: Learn independent vectors for query and document (e.g., DSSM). Comparison is fast via cosine similarity.
- **Interaction-based Models**: Build a similarity matrix of all query-document term pairs to capture nuanced matching (e.g., KNRM, DRMM).

### 4. The BERT Revolution
Before 2018, neural models often struggled to beat well-tuned BM25 on smaller datasets. BERT changed this by enabling "soft" or semantic matching through pretraining.
- **Milestone**: In Jan 2019, Nogueira and Cho applied BERT to MS MARCO, jumping effectiveness by ~30% relative to previous bests.
- **Why it worked**: Pretraining on massive text corpora allowed the model to understand context and language nuances far better than task-specific training alone.

## Roadmap of the Book
The book follows a structured progression:
1. **Multi-Stage Architectures**: Using expensive models as rerankers for an initial cheap retrieval stage.
2. **Refining Representations**: Techniques for **Query Expansion** and **Document Expansion** (e.g., doc2query).
3. **[[Dense Retrieval]]**: Learning to map queries and documents into a shared embedding space for efficient retrieval using Approximate Nearest Neighbor (ANN) search.

## Summary
- **Text Ranking** is the core of information access.
- We have moved from **Exact Match** ([[BM25]], [[TF-IDF]], [[Inverted Index]]) $\to$ **[[Learning to Rank]]** (feature engineering) $\to$ **Deep Learning** ([[Word Embeddings]]) $\to$ **[[Transformers]]** (BERT, [[Cross-Encoder]], [[Bi-Encoder]]).
- BERT allows for "semantic matching," solving the vocabulary mismatch problem that plagued earlier systems.

> [!intuition] Beyond "Ten Blue Links"
> Modern IR is moving away from just matching keywords toward understanding the *intent* of the query and the *context* of the document using the rich representations provided by Transformer models.
