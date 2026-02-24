---
type: concept
aliases: [Cranfield Paradigm, Cranfield Evaluation, Laboratory Evaluation]
course: [IR]
tags: [evaluation, exam-topic]
status: complete
---

# Cranfield Paradigm

> [!definition] Cranfield Paradigm
> The **Cranfield Paradigm** is the foundational framework for the laboratory evaluation of Information Retrieval systems. Established by Cyril Cleverdon in the 1960s, it enables reproducible and comparable offline evaluation by using standardized test collections.

## Components of a Test Collection

To evaluate a system under this paradigm, three components are required:
1. **Document Collection**: A static corpus of documents (e.g., Wikipedia dump, news articles).
2. **Topic Set**: a set of information needs (queries), typically including a title (short query) and a detailed description.
3. **Ground Truth (Qrels)**: Releance judgments indicating which documents are relevant to which topics, usually provided by human annotators.

> [!intuition] The "Laboratory" Approach
> By keeping the documents and queries fixed, researchers can change their retrieval algorithms and immediately see if the results improve compared to a "Gold Standard" (the Qrels). This removes the variability of live users.

## Key Assumptions

- **Relevance is binary** (or multi-level) and independent of other documents.
- **Relevance is static**: A document is either relevant to a topic or it isn't, regardless of time or user context.
- **Completeness**: All relevant documents in the collection are known (though often mitigated by [[Pooling]]).

## Connections

- Measurement: Evaluated using metrics like [[Precision and Recall]], [[MAP]], and [[NDCG]].
- Methodology: Utilizes [[Pooling]] to handle large collections.
- Contrast: Differs from **Online Evaluation** (e.g., A/B testing, click-through rates).

## Appears In

- [[IR-L04 - Evaluation]]
