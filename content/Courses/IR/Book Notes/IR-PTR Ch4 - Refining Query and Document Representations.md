---
type: book-chapter
course: IR
book: "Pretrained Transformers for Text Ranking: BERT and Beyond"
chapter: 4
sections: ["4.1", "4.2", "4.3", "4.4", "4.5", "4.6"]
topics:
  - "[[Query Expansion]]"
  - "[[Document Expansion]]"
  - "[[DocT5Query]]"
  - "[[DeepCT]]"
  - "[[DeepImpact]]"
  - "[[Learned Sparse Retrieval]]"
status: complete
---

# IR-PTR Ch4: Refining Query and Document Representations

## Overview
This chapter explores techniques to mitigate the **vocabulary mismatch problem**, where the terms used in queries differ from those in relevant documents. While [[Transformers]] solve this via semantic matching in reranking, the initial candidate generation stage (e.g., [[BM25]]) remains a bottleneck. Refining query and document representations allows these techniques to bridge the gap between classical [[Information Retrieval]] (exact match) and neural approaches.

---

## 4.1 General Remarks on Expansion
Both [[Query Expansion]] and [[Document Expansion]] aim to align representations by adding or reweighting terms.

| Feature | Query Expansion | Document Expansion |
| :--- | :--- | :--- |
| **Advantage** | Flexibility, short experimental cycles, can aggregate evidence (post-retrieval). | Richer context for transformers, embarrassingly parallel, pushes inference to indexing time. |
| **Disadvantage** | Longer queries increase retrieval latency. | Indexing time/cost, requires re-indexing for any model change. |

> [!intuition]
> Document expansion is like "predictive annotation"—adding terms a document *should* have been tagged with to be findable.

---

## 4.2 Pseudo-Relevance Feedback with Contextualized Embeddings: CEQE
Pseudo-relevance feedback (PRF) assumes the top-k results of an initial search are relevant and uses them to expand the query.

> [!definition] Rocchio Algorithm
> One of the earliest PRF methods, performing manipulations in the vector space model.
> > [!formula]
> > $$\vec{q}_{new} = \alpha \vec{q}_{old} + \beta \frac{1}{|D_{rel}|} \sum_{d \in D_{rel}} \vec{d} - \gamma \frac{1}{|D_{nrel}|} \sum_{d \in D_{nrel}} \vec{d}$$

### CEQE (Contextualized Embeddings for Query Expansion)
Standard PRF with [[BERT for IR]] is difficult because BERT expects natural language, not keyword lists. CEQE uses BERT's contextual embeddings from the 11th layer to calculate the probability of an expansion term $w$.

> [!formula] CEQE Probability
> $$p(w, Q, D) = \sum_{D} p(w|Q, D) p(Q|D) p(D)$$
> Where $p(w|Q, D)$ is calculated using cosine similarity between term mentions and a query centroid (or term-based representation with pooling).

---

## 4.3 Document Expansion via Query Prediction: doc2query
Also known as **docTTTTTquery** when using T5.

> [!definition] doc2query
> A sequence-to-sequence model (like T5) is trained to generate potential queries given a document as input. These queries are then appended to the original document before indexing.

### Key Findings:
- **New vs. Copied Terms**: Approximately 31% of predicted terms are "new" (not in the doc), helping bridge the vocabulary mismatch. 69% are "copied" (term reweighting).
- **Effectiveness**: Often achieves the effectiveness of non-BERT neural models using only basic keyword search.
- **Independence**: The technique is a "free boost" for first-stage retrieval that doesn't require GPU inference at query time.

---

## 4.4 Term Reweighting as Regression: DeepCT
Unlike doc2query, which reweights terms indirectly by repetition, [[DeepCT]] directly predicts term importance.

> [!formula] Query Term Recall (QTR)
> The label used to train DeepCT is the fraction of relevant queries containing term $t$:
> $$QTR(t, d) = \frac{|Q_{d,t}|}{|Q_d|}$$

### Mechanism:
1. **Regression**: A BERT-based model takes document $d$ and outputs importance score $\hat{y}_{t,d}$ for each term $t$.
2. **Indexing**: Scores are rescaled (e.g., 0-100) and treated as term frequencies in a standard [[Inverted Index]].
3. **Efficiency**: Only one inference pass per document is needed, compared to multiple sampling passes for doc2query.

---

## 4.5 Term Reweighting with Weak Supervision: HDCT
**HDCT** (Hierarchical Document Term Weighting) extends DeepCT for long documents.

> [!example] Workflow
> 1. Split document into passages.
> 2. Pass each passage through BERT to get term weights $\hat{y}_{t,p}$.
> 3. Aggregate passage weights into a document-level weight: $D(d) = \sum pwi \times P(pi)$.
> 4. Use "Sum" or "Decay" (discounting later passages) for weights.

**Weak Supervision**: Since passage-level judgments are rare, HDCT uses document titles or pseudo-relevant documents to generate synthetic training labels.

---

## 4.6 Combining Expansion and Reweighting: DeepImpact
[[DeepImpact]] combines doc2query's expansion with a scoring model to obtain the "best of both worlds."

1. **Expand**: Generate terms with doc2query-T5.
2. **Weight**: Use BERT and an MLP to predict "impact" weights for both original and expansion terms.
3. **Index**: Store quantized weights (impacts) in the term frequency position.

> [!intuition]
> DeepImpact allows keyword retrieval to approach the effectiveness of [[Neural Reranking]] (like monoBERT) while being an order of magnitude faster and requiring no query-time neural inference.

---

## 4.7 Connection to Sparse Retrieval
These techniques refine textual representations but still utilize the [[Inverted Index]]. They represent a bridge toward [[Learned Sparse Retrieval]] (e.g., [[SPLADE]]), where the model learns a high-dimensional sparse vector across the entire vocabulary rather than just weighting existing/predicted terms.

## Summary
- **Query refinement** (CEQE) is powerful but computationally expensive at query time.
- **Document refinement** (doc2query, DeepCT, DeepImpact) pushes the "heavy lifting" to indexing time.
- These methods solve [[Information Retrieval]]'s oldest problem—vocabulary mismatch—within the framework of efficient classical search engines.
