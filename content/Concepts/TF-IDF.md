---
type: concept
aliases: [TF-IDF, tf-idf, term frequency-inverse document frequency]
course: [IR]
tags: [retrieval-models, key-formula, exam-topic]
status: complete
---

# TF-IDF

> [!definition] TF-IDF
> **TF-IDF** (Term Frequency–Inverse Document Frequency) is a term weighting scheme that reflects how important a word is to a document in a collection. It combines two intuitions: terms that appear frequently in a document are important (TF), and terms that appear in few documents are more discriminative (IDF).

> [!formula] TF-IDF Weight
> $$\text{tf-idf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)$$
> 
> where:
> - $\text{tf}(t, d)$ — term frequency (count of term $t$ in document $d$, or log-scaled: $1 + \log \text{tf}$)
> - $\text{idf}(t) = \log \frac{N}{df(t)}$ — inverse document frequency ($N$ = total docs, $df(t)$ = docs containing $t$)

## TF Variants

| Variant | Formula | Behavior |
|---------|---------|----------|
| Raw | $f(t,d)$ | Linear with count |
| Log-scaled | $1 + \log f(t,d)$ | Sublinear — diminishing returns |
| Boolean | $1$ if $t \in d$ else $0$ | Presence/absence only |
| Augmented | $0.5 + 0.5 \cdot \frac{f(t,d)}{\max_t f(t,d)}$ | Normalized by max TF |

## Scoring with TF-IDF

Documents and queries are represented as TF-IDF weighted vectors in the [[Vector Space Model]]. Similarity is computed via cosine:

$$\text{sim}(q, d) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}| \cdot |\vec{d}|}$$

## Connections

- Foundation of: [[Vector Space Model]]
- Extended by: [[BM25]] (adds saturation + length normalization)
- Used in: [[Inverted Index]] for term weighting

## Appears In

- [[IR-L02 - IR Fundamentals]], [[IR-L03 - Retrieval Models]]
- [[IR-A01 - Unsupervised Retrieval]]
