---
type: concept
aliases: [query likelihood, QL, query likelihood model, language model for IR]
course: [IR]
tags: [retrieval-models, key-formula, exam-topic]
status: complete
---

# Query Likelihood Model

> [!definition] Query Likelihood Model
> The **Query Likelihood (QL) model** ranks documents by the probability that the document's language model would generate the query. Instead of asking "is this document relevant?", it asks "how likely is this query given this document?"

> [!formula] Query Likelihood
> $$P(q|d) = \prod_{t \in q} P(t|d)$$
> 
> Log form (for ranking):
> $$\log P(q|d) = \sum_{t \in q} \log P(t|d)$$

## Smoothing

Raw maximum likelihood estimation ($P(t|d) = \frac{f(t,d)}{|d|}$) assigns zero probability to terms not in the document. **[[Smoothing]]** fixes this:

### Jelinek-Mercer (Linear Interpolation)
$$P(t|d) = (1-\lambda) \frac{f(t,d)}{|d|} + \lambda P(t|C)$$

### Dirichlet Prior
$$P(t|d) = \frac{f(t,d) + \mu P(t|C)}{|d| + \mu}$$

> [!intuition] Smoothing Intuition
> - $P(t|C)$ is the collection language model (background probability)
> - Smoothing says: "If this term isn't in the document, fall back to how common it is in the whole collection"
> - Dirichlet $\mu$: pseudo-count. Large $\mu$ → more smoothing (trust the collection more)

## Connections

- Alternative to: [[BM25]], [[TF-IDF]]
- Uses: [[Smoothing]]
- Foundation of: [[Language Model for IR]]

## Appears In

- [[IR-L03 - Retrieval Models]]
- [[IR-A01 - Unsupervised Retrieval]]
