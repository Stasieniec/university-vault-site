---
type: book-chapter
course: IR
book: "Pretrained Transformers for Text Ranking: BERT and Beyond"
chapter: 2
sections: ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9"]
topics:
  - "[[Information Retrieval]]"
  - "[[Precision]]"
  - "[[Recall]]"
  - "[[NDCG]]"
  - "[[MAP]]"
  - "[[MRR]]"
  - "[[BM25]]"
  - "[[Cranfield Paradigm]]"
status: complete
---

# IR-PTR Ch2 - Setting the Stage

## Overview
This chapter formally characterizes the **[[Information Retrieval]]** (IR) ranking problem, specifically focusing on **top-k retrieval** (ad hoc retrieval). The evaluation foundation is built upon the **[[Cranfield Paradigm]]**, a system-oriented approach to batch evaluation that has dominated the field for over half a century. While alternative paradigms like interactive evaluations and A/B testing exist, the Cranfield paradigm remains the primary vehicle for ranking research due to its reproducibility and scale.

> [!intuition]
> The "Core" Ranking Problem: Given an information need (query $q$), return a ranked list of $k$ texts from a collection $C$ that maximizes a specific metric of interest.

## Texts
The [[Information Retrieval]] task assumes a corpus $C = \{d_i\}$ of mostly unstructured natural language text.
- **Granularity**: Texts can range from sentences to entire books. The unit of retrieval (e.g., paragraphs, passages, or full documents) is a design choice known as **passage retrieval**.
- **Constraints**: Modern systems must handle billions of pages, necessitating high computational efficiency. Transformer models introduce specific challenges due to fixed maximum sequence lengths and memory/latency overhead for long texts.
- **Multilinguality**: Typically focused on English, with extensions to mono-lingual or cross-lingual retrieval being orthogonal to the core ranking architecture discussed here.

## Information Needs
There is a critical distinction between a user's internal **information need** and the external **query** provided to a system. 

- **Anomalous State of Knowledge (ASK)**: Information needs arise from gaps in a user's cognitive state.
- **TREC Topics**: Often operationalized as "topics" with three fields:
    - **Title**: Short keyword query (standard input for models).
    - **Description**: Natural language sentence.
    - **Narrative**: Detailed prose (often leading to poor results due to "distractor" terms in keyword-matching systems).

> [!tip]
> In most IR evaluations, it is assumed that the **topic title** was used as the query unless stated otherwise.

## Relevance
Relevance is the foundational relation between a text and an information need. It is complex, subjective, and "in the eye of the beholder."

- **Dimensions**:
    - **Topical Relevance**: The "aboutness" of the text.
    - **Cognitive Relevance**: Understandability/Expertise level.
    - **Situational Relevance**: Utility for a specific task.
- **Subjectivity**: Relevance is an *opinion* (assessor judgment), not a platonic truth. 
- **The Paradox of Agreement**: Inter-assessor agreement is surprisingly low (~60% overlap). However, the *ranking* of systems is highly stable across different assessors (Kendall's $\tau > 0.9$).

> [!definition]
> **[[Cranfield Paradigm]] Assumption**: While absolute evaluation scores vary by assessor, the relative comparison between system A and system B remains consistent.

## Judgments
**Relevance Judgments** (or **qrels**) are the "ground truth" (opinions) used for training and evaluation.

- **Format**: Tríples of $(q, d, r)$ where $q$ is the query, $d$ the document ID, and $r$ the relevance grade.
- **Scales**:
    - **Binary**: Relevant vs. Not Relevant.
    - **Graded**: e.g., PEGFB (Perfect, Excellent, Good, Fair, Bad).
- **Human vs. Heuristic**: While positive labels are usually provided by humans, non-relevant labels in large datasets (e.g., MS MARCO) are often heuristically sampled (e.g., using **[[BM25]]** results not marked as relevant).

## Ranking Metrics
Metrics quantify the "goodness" of a ranked list. Symbols $R$ denotes the ranked list, $l$ its length, and $k$ the evaluation cutoff.

### Precision and Recall
> [!formula] **[[Precision]]**
> The fraction of retrieved documents that are relevant.
> $$\text{Precision}(R, q) = \frac{\sum_{(i, d) \in R} \text{rel}(q, d)}{|R|}$$
> Commonly used as **Precision at K** (**P@k**).

> [!formula] **[[Recall]]**
> The fraction of all relevant documents in the collection that are retrieved.
> $$\text{Recall}(R, q) = \frac{\sum_{(i, d) \in R} \text{rel}(q, d)}{\sum_{d \in C} \text{rel}(q, d)}$$

- **F-Measure**: The harmonic mean of Precision and Recall.

### Reciprocal Rank (RR)
Focuses on the position of the *first* relevant document.
> [!formula] **[[MRR]]** (Mean Reciprocal Rank)
> $$\text{RR}(R, q) = \frac{1}{\text{rank}_i}$$
> Where $\text{rank}_i$ is the rank of the first relevant result. Best for tasks where one answer suffices (e.g., Factoid QA).

### Average Precision (AP)
The primary metric used when recall is important. It is the average of precision scores at each relevant document's rank.
> [!formula] **[[MAP]]** (Mean Average Precision)
> $$\text{AP}(R, q) = \frac{\sum_{(i, d) \in R} \text{Precision}@i(R, q) \cdot \text{rel}(q, d)}{\sum_{d \in C} \text{rel}(q, d)}$$

### NDCG (Normalized Discounted Cumulative Gain)
Specifically designed for **graded relevance**.
> [!formula] **[[NDCG]]**
> First, calculate **DCG**:
> $$\text{DCG}(R, q) = \sum_{(i, d) \in R} \frac{2^{\text{rel}(q, d)} - 1}{\log_2(i + 1)}$$
> Then normalize by the Ideal DCG (**IDCG**):
> $$\text{nDCG}(R, q) = \frac{\text{DCG}(R, q)}{\text{IDCG}(R, q)}$$
> Normalization ensures the score is in $[0, 1]$.

> [!tip]
> **Unjudged Documents**: Standard tools like `trec_eval` treat unjudged documents as non-relevant. This can penalize models that surface valid but unjudged results (the "lamplight" bias).

## Community Evaluations
Evaluations like **TREC** (Text Retrieval Conferences) provide the shared infrastructure for progress.

- **Pooling**: Since checking every document for every query is impossible, organizers use **top-k pooling**. Only the top results from many different participating systems are merged and judged by humans.
- **Reusability**: A test collection is "reusable" if it can accurately evaluate new systems that did not participate in the original pooling.
- **Bias**: If pools only contain keyword-matching systems, the resulting judged documents may be biased against neural/semantic models.

## Test Collections
### MS MARCO (Microsoft MAchine Reading COmprehension)
The catalyst for the current transformer era.
- **Passage Ranking**: 8.8M passages. sparse judgments (avg. 1 relevant per query). MRR@10 is the official metric.
- **Document Ranking**: 3.2M documents.
- **Impact**: Enabled supervised training of massive models like BERT for ranking.

### TREC Deep Learning Tracks (2019+)
Built on MS MARCO but with NIST-standard **"dense" judgments**. Provides richer labels for better discrimination between models.

### Robust04
A veteran collection (Newswire) with 249 topics and deep, highly reliable judgments. It remains a standard benchmark for testing the generality of new models.

## Keyword Search
Despite the rise of Neural IR, keyword search is the standard **candidate generation** (first-stage) mechanism.

- **Inverted Index**: The data structure that enables fast term lookups.
- **[[BM25]]**: The de facto standard ranking function.
- **[[TF-IDF]]**: Term Frequency-Inverse Document Frequency, the predecessor logic to BM25.
- **Query Expansion**: Using **Pseudo-Relevance Feedback** (e.g., **RM3**) to add terms from top-ranked documents to the query, mitigating the vocabulary mismatch problem.

## Summary
- **Effectiveness vs. Efficiency**: IR distinguishes between output quality (Effectiveness) and system speed (Efficiency).
- **Terminology**: "Retrieval" often implies the retrieval-and-ranking process; "Reranking" specifically refers to reordering a candidate list.
- **Foundation**: Chapter 2 provides the "Cranfield" framework—Corpus, Queries, Relevance, and Metrics—that allows ranking to be treated as an optimization problem for supervised machine learning models like Transformers.

---
*Note: This chapter sets the stage for Chapter 3, which focuses on specific Transformer architectures.*
