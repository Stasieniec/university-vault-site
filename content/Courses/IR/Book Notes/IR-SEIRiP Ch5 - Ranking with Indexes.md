---
type: book-chapter
course: IR
book: "Search Engines: Information Retrieval in Practice"
chapter: 5
sections: ["5.1", "5.2"]
topics:
  - "[[Inverted Index]]"
  - "[[Information Retrieval]]"
status: complete
---

# IR: Ranking with Indexes (Ch 5)

## Overview
Ranking with indexes is the core of modern [[Information Retrieval]]. While standard data structures like arrays or hash tables are useful for general computing, the scale of web search (billions of pages, millions of queries per day) and the specific nature of text retrieval necessitate the use of the [[Inverted Index]]. 

The "inverted index" is an umbrella term for various structures that share a general philosophy: associating terms with the documents that contain them. The choice of index structure is heavily dictated by the ranking function being used.

## Abstract Model of Ranking
Ranking is the process of transforming human-language documents and queries into numerical features to estimate relevance.

### Conceptual Components
- **Documents**: Transformed into index terms or document features.
- **Features**: Numerical attributes of a document.
    - **Topical Features**: Estimate the degree to which a document is about a subject (e.g., frequency of "[[TF-IDF|term weighting]]").
    - **Quality Features**: Estimate document "goodness" independent of the query (e.g., incoming links, update frequency).
- **Ranking Function**: A mathematical "cloud" that processes features and query data to produce a **Score**.

> [!definition] Ranking Function
> A mathematical expression that takes data from document features combined with a query to produce a score (usually a real number). Documents are subsequently sorted by this score to present results to the user.

### Concrete Model
Most modern systems use a ranking function $R(Q, D)$ that follows a summation model:

$$R(Q, D) = \sum_{i} g_i(Q) f_i(D)$$

Where:
- $f_i(D)$ is a feature function extracting a value from the document.
- $g_i(Q)$ is a feature function extracting a value from the query.

> [!intuition] Efficiency through Sparsity
> If a search engine had to sum over millions of features for every document, it would be impractical. In practice, most $g_i(Q)$ values are zero (the query only contains a few terms), limiting the summation only to the non-zero query features.

## Inverted Indexes
An [[Inverted Index]] is the computational equivalent of the index found at the back of a textbook. 

> [!intuition] The Inversion
> Standard document storage is "Forward": Document $\to$ Terms.
> An Inverted Index flips this: Term $\to$ Documents.

### Structure and Terms
- **Lexicon**: The list of terms (often alphabetized or hashed).
- **Inverted List**: The list of document references for a specific term.
- **Posting**: A single entry in the list representing an occurrence.
- **Pointer**: The part of the posting that refers to a specific document ID.

### Types of Inverted Indexes

#### 1. Document Indexes (Binary)
The simplest form. Stores only whether a document contains a term ($1$) or not ($0$).
- **Use Case**: [[Boolean Retrieval]].
- **Pros**: Smallest storage footprint.
- **Cons**: Too coarse for ranking many documents; cannot distinguish deep relevance from passing mentions.

#### 2. Count Indexes
Stores document IDs along with word counts (term frequency).
- **Use Case**: [[Term Weighting]], [[TF-IDF]], and [[BM25]].
- **Pros**: Helps distinguish documents focused on a topic from those that just mention it.
- **Model**: Matches the [[Bag of Words]] assumption.

> [!example] Count Posting
> A posting for "fish" in Sentence 2 might look like `2:3`, meaning Document 2 contains the word "fish" 3 times.

#### 3. Position Indexes
Stores the exact location (word index) of terms within the document.
- **Use Case**: **Phrase Queries** (e.g., "[[Information Retrieval]]") and **Proximity Search**.
- **Pros**: Allows checking if terms appear next to each other.
- **Cons**: Significantly larger index size (one posting per occurrence rather than per document).

#### 4. Fields and Extents
Documents are structured (Titles, Headers, Body, Metadata).
- **Structure**: Fields allow the search engine to weight a match in the **Title** higher than a match in the **Footer**.
- **Extents**: Represent contiguous spans of text (e.g., sentences or paragraphs).

## Summary
The inverted index is the most efficient and flexible data structure for [[Information Retrieval]]. By storing metadata like counts and positions, it enables the implementation of complex retrieval models while maintaining the speed required for large-scale applications. The evolution from simple binary indexes to position-aware and field-aware indexes represents the balance between retrieval effectiveness and computational efficiency.
