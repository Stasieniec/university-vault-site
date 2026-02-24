---
type: concept
aliases: [Rocchio Algorithm, Rocchio feedback]
course: [IR]
tags: [retrieval-models, relevance-feedback, key-formula]
status: complete
---

# Rocchio Algorithm

> [!definition] Rocchio Algorithm
> The **Rocchio Algorithm** is a classic technique for implementing **Relevance Feedback**. It updates an initial query vector by moving it toward the centroid of known relevant documents and away from the centroid of known non-relevant documents.

> [!formula] Rocchio Query Update
> $$\vec{q}_m = \alpha \vec{q}_0 + \beta \frac{1}{|D_r|} \sum_{\vec{d} \in D_r} \vec{d} - \gamma \frac{1}{|D_{nr}|} \sum_{\vec{d} \in D_{nr}} \vec{d}$$
> 
> where:
> - $\vec{q}_m$ — the modified query vector
> - $\vec{q}_0$ — the original query vector
> - $D_r$ — set of known relevant documents
> - $D_{nr}$ — set of known non-relevant documents
> - $\alpha, \beta, \gamma$ — weights (hyperparameters) controlling the balance between original intent, positive feedback, and negative feedback.

> [!intuition] Vector Space Navigation
> Imagine the Vector Space Model where points represent documents. A user's query is also a point. If the user marks some results as "good," Rocchio literally "nudges" the query point closer to those good results. Usually, we weigh relevant documents more heavily than non-relevant ones ($\beta > \gamma$).

## Pseudo-Relevance Feedback (PRF)
Since users rarely provide explicit feedback, we often use **Pseudo-Relevance Feedback**:
1. Run initial search.
2. Assume the top $K$ documents are relevant (no $D_{nr}$).
3. Apply Rocchio to expand the query.
4. Run second search with $\vec{q}_m$.

## Connections
- Context: Used within the Vector Space Model (VSM).
- Related concepts: [[BM25]] (which typically doesn't use vectorcentroids), [[Query Expansion]].
- Modern link: Modern PRF methods often use [[Transformers]] to expand the query instead of vector arithmetic.

## Appears In
- [[IR-L03 - Retrieval Models]]
