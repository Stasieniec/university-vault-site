---
type: concept
aliases: [BIM, Binary Independence Model]
course: [IR]
tags: [retrieval-models]
status: complete
---

# Binary Independence Model

> [!definition] Binary Independence Model (BIM)
> The **Binary Independence Model** is a classic probabilistic model for IR. It makes two fundamental assumptions:
> 1. **Binary**: Documents and queries are represented as binary incidence vectors (a term is either present or absent).
> 2. **Independence**: The presence of one term is independent of the presence of any other term, given the relevance or non-relevance of the document.

> [!formula] Retrieval Status Value (RSV)
> The BIM ranks documents using the log-odds of relevance:
> 
> $$\text{RSV} = \sum_{t \in q \cap d} \log \frac{p_t (1 - u_t)}{u_t (1 - p_t)}$$
> 
> where:
> - $p_t = P(x_t = 1 | R)$ — probability that term $t$ is present in a relevant document.
> - $u_t = P(x_t = 1 | \bar{R})$ — probability that term $t$ is present in a non-relevant document.

> [!intuition] Counting Evidence
> BIM treats terms as clues. If a term is very likely to appear in "good" docs and very unlikely in "bad" docs, seeing that term in a document is strong evidence for relevance. By assuming independence, we can simply add up the "weight of evidence" for every matching term to get a final score.

## Key Properties

- **Simplistic but Foundation**: It ignores term frequency (TF) and document length, which makes it less effective than [[BM25]] on its own.
- **Basis for BM25**: [[BM25]] was created by extending BIM with TF and length normalization.
- **Probability Ranking Principle**: BIM is a direct implementation of the principle that a system should rank documents by their probability of relevance.

## Connections

- **Evolved into**: [[BM25]]
- **Contrast**: [[Vector Space Model]] (geometric), [[Language Model for IR]] (generative)
- **Assumptions**: Term independence (similar to Naive Bayes).

## Appears In

- [[IR-L03 - Retrieval Models]]
