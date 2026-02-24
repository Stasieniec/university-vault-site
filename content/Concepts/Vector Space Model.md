---
type: concept
aliases: [VSM, vector space model]
course: [IR]
tags: [retrieval-models, foundations]
status: complete
---

# Vector Space Model

> [!definition] Vector Space Model
> The **VSM** represents documents and queries as vectors in a high-dimensional term space. Each dimension corresponds to a term; weights are typically [[TF-IDF]]. Similarity is computed via cosine similarity.

> [!formula] Cosine Similarity
> $$\text{sim}(q, d) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}| \cdot |\vec{d}|} = \frac{\sum_i q_i \cdot d_i}{\sqrt{\sum_i q_i^2} \cdot \sqrt{\sum_i d_i^2}}$$

**Properties:**
- Handles partial matching (unlike Boolean retrieval)
- Length normalization via cosine
- Foundation extended by [[BM25]] and [[Dense Retrieval]] (learned dense vectors)

## Appears In

- [[IR-L02 - IR Fundamentals]], [[IR-L03 - Retrieval Models]]
