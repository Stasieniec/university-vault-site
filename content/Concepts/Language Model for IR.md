---
type: concept
aliases: [Language Modeling for IR, LM-IR, Language Models for IR]
course: [IR]
tags: [retrieval-models]
status: complete
---

# Language Model for IR

## Definition

> [!definition] Language Model for IR
> The **language-modeling (LM) approach to IR** treats retrieval as a **generative** problem:
> estimate a probabilistic language model $M_d$ for each document, then rank documents by how
> likely that model is to have produced the query (or by how close it is to a query model).
> It reframes the question from "is $d$ relevant to $q$?" to "could $d$'s model have generated
> $q$?" — a probabilistic alternative to [[Vector Space Model]] and [[BM25]].

> [!intuition] The generative bridge (the urn)
> Imagine each document is an urn filled with words in document-specific proportions. A user with
> a document in mind samples query words from its urn. Retrieval inverts this: find the urn most
> likely to have produced the observed query words.

## The Family of LM Approaches

- **[[Query Likelihood Model]]** — the dominant instance: rank by $P(q \mid M_d) = \prod_{t\in q} P(t\mid M_d)$, with [[Smoothing]] against the collection model. (Full formula, Jelinek–Mercer and Dirichlet smoothing, and intuition live in that note.)
- **Document likelihood** — rank by $P(d \mid M_q)$, generating the document from a query model (less common; query models are sparse).
- **Model comparison (KL divergence / Relevance Models)** — build a query model $M_q$ and a document model $M_d$ and rank by their divergence $-\,\mathrm{KL}(M_q \,\Vert\, M_d)$. This **generalizes** query likelihood and underlies pseudo-relevance-feedback relevance models.

> [!formula] Unifying view
> $$\text{score}(q,d) \;\propto\; -\,\mathrm{KL}\!\big(M_q \,\Vert\, M_d\big)
> \quad\xrightarrow{\;M_q=\text{empirical } q\;}\quad \sum_{t\in q} P(t\mid q)\log P(t\mid M_d)$$
>
> where setting the query model $M_q$ to the raw query terms recovers (a rank-equivalent of) the
> [[Query Likelihood Model]].

## Key Properties

- **Per-document models** — one distribution $M_d$ per document in the collection.
- **Smoothing is essential** — without it, one missing query term zeroes the product; [[Smoothing]] borrows mass from the collection model.
- **Probabilistic footing** — a principled alternative to heuristic term-weighting; connects naturally to modern generation ([[Retrieval-Augmented Generation]]).

## Connections

- Primary instance: [[Query Likelihood Model]]
- Alternative to: [[BM25]], [[Vector Space Model]], [[Binary Independence Model]]
- Uses: [[Smoothing]]
- Modern extension: [[Retrieval-Augmented Generation]]

## Appears In

- [[IR-L03 - Retrieval Models]]
- [[IR-SEIRiP Ch7 - Retrieval Models]]
