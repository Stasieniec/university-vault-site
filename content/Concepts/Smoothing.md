---
type: concept
aliases: [Smoothing, Linear Interpolation, Dirichlet Smoothing, Absolute Discounting]
course: [IR]
tags: [retrieval-models, exam-topic]
status: complete
---

# Smoothing

> [!definition] Smoothing
> **Smoothing** is a technique used in language models for information retrieval to adjust probability estimates. Its primary goal is to prevent zero-probability estimates for terms that do not appear in a specific document but are present in the general collection, while also accounting for the document's content.

> [!intuition] Why Smooth?
> Without smoothing, if a document is missing even one term from a multi-word query, the language model would assign it a probability of 0 ($P(Q|D) = 0$). Smoothing "steals" a small amount of probability mass from seen terms and redistributes it to unseen terms using the background collection model.

## Common Smoothing Methods

### 1. Jelinek-Mercer (Linear Interpolation)
Mixes the document model with the collection model using a fixed weight $\lambda$.
$$P_\lambda(t|M_d) = (1 - \lambda) P_{mle}(t|M_d) + \lambda P_{mle}(t|M_c)$$
- **$\lambda$**: Higher values (e.g., 0.7) favor the collection (better for long queries), lower values (e.g., 0.1) favor the document (better for short queries).

### 2. Dirichlet Prior
Uses a pseudo-count $\mu$ from the collection.
$$P_\mu(t|M_d) = \frac{f(t, d) + \mu P(t|M_c)}{|d| + \mu}$$
- **Intuition**: As document length $|d|$ increases, the influence of the prior $\mu$ diminishes. It provides stronger length normalization than Jelinek-Mercer.

### 3. Absolute Discounting
Subtracts a constant $d$ from seen term counts.
$$P_\delta(t|M_d) = \frac{\max(f(t, d) - \delta, 0)}{|d|} + \frac{\delta \cdot u}{|d|} P(t|M_c)$$
where $u$ is the number of unique terms in the document.

## Connections

- Foundation: [[Query Likelihood Model]]
- Part of: [[Language Models for IR]]
- Relates to: [[TF-IDF]] (smoothing behaves similarly to IDF by downweighting common terms)

## Appears In

- [[IR-L03 - Retrieval Models]]
