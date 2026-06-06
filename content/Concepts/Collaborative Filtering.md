---
type: concept
aliases: [CF, Model-based Collaborative Filtering]
course: [RecSys]
tags: [collaborative-filtering, evaluation, exam-topic]
status: complete
---

# Collaborative Filtering

## Definition

> [!definition] Collaborative Filtering (CF)
> **Collaborative filtering** is a recommendation technique that predicts a user's interest in items by leveraging the **collective knowledge of a large pool of users** — i.e. the observed **user–item interaction data** — rather than item content. The core assumption: users who agreed in the past will agree in the future. If a user $u$ is similar to a user $v$, and $v$ liked an item $u$ has not yet seen, that item is a good recommendation for $u$.
>
> Formally, given users $U = \{u_1, \dots, u_n\}$ and items $I = \{i_1, \dots, i_m\}$, CF frames recommendation as **completing the sparse user–item interaction matrix** $R \in \mathbb{R}^{n \times m}$: predict the missing entry $r_{ui}$ (e.g. a rating, click, or purchase) for a (user, item) pair that has not been observed.

## Intuition

> [!intuition] Wisdom of the crowd, not content
> Content-based methods look at *what an item is* (text, genre, metadata). Collaborative filtering ignores content entirely and looks only at *how users behaved*: "users who like artist X also like Y." The signal comes from patterns of agreement across many users.
>
> Picture the interaction matrix: rows = users, columns = items, most cells empty. CF fills a missing cell `?` in one of two ways:
> - **User-based:** "How do users *similar to me* rate this item?"
> - **Item-based:** "How do I rate items *similar to this one*?" (usually preferred, because item–item relationships are **more stable** over time than user profiles.)

## Mathematical Formulation

CF splits into two families. **Neighborhood (memory-based) CF** makes predictions directly from similarity; **model-based CF** trains a model (e.g. matrix factorization) from the data.

> [!formula] User-based rating prediction (neighborhood CF)
> Predict the rating $\hat{r}_{ui}$ of user $u$ for item $i$ by averaging the ratings that $u$'s **nearest neighbors** gave to $i$:
> $$\hat{r}_{ui} = \frac{1}{|\mathcal{N}_i(u)|} \sum_{v \in \mathcal{N}_i(u)} r_{vi}$$
>
> where:
> - $\mathcal{N}_i(u)$ — set of the $k$ nearest neighbors of $u$ who have rated item $i$ (the $k$-NN)
> - $r_{vi}$ — rating that neighbor $v$ gave to item $i$
> - similarity (defining "nearest") is computed from shared interactions, e.g. cosine or Pearson correlation between rating vectors
>
> (More general weighted variants weight each neighbor by its similarity $\text{sim}(u,v)$ and mean-center to correct for user rating bias.)

> [!formula] Matrix Factorization (model-based CF)
> Approximately factorize the $n \times m$ interaction matrix $R$ into a low-rank product of a user-factor matrix and an item-factor matrix:
> $$R \approx U V^{\top}, \qquad \hat{r}_{ij} = \bar{u}_i \cdot \bar{v}_j = \sum_{f=1}^{k} u_{if}\, v_{jf}$$
>
> where:
> - $U \in \mathbb{R}^{n \times k}$ — each **row** $\bar{u}_i$ is a user factor (preferences over $k$ latent concepts)
> - $V \in \mathbb{R}^{m \times k}$ — each **row** $\bar{v}_j$ is an item factor (properties over the same $k$ concepts)
> - $k$ — number of latent factors (the rank); $k \ll n, m$
> - the predicted preference is the **dot product** of the user and item factors
>
> Latent factors can be interpretable: a rank-2 movie example learns a *history* axis and a *romance* axis, and a rating reconstructs as (user's affinity to history $\times$ item's history-ness) + (user's affinity to romance $\times$ item's romance-ness). The general recipe is: **define a model → define an objective → optimize** (e.g. minimize regularized squared error over observed entries via SGD/ALS).

## Key Properties / Variants

- **Two CF families:**
  - *Neighborhood / memory-based:* no model trained in advance; relies on the similarity of two entities ($k$-NN). **Pros:** simple, efficient, transparent. **Cons:** data sparsity, noise, scalability. Sub-types: [[User-based Collaborative Filtering]] and [[Item-based Collaborative Filtering]] (item-based preferred for stability).
  - *Model-based:* train a model from data (e.g. [[Matrix Factorization]]). **Pros:** scalability, generalization. **Cons:** complexity, black-box, overfitting with insufficient data.
- **Implicit vs explicit feedback:** explicit = ratings; implicit = clicks/views/purchases (treat observed = positive, unobserved = candidate negatives, often with negative sampling).
- **Bayesian Personalized Ranking ([[BPR]]):** a pairwise ranking objective for implicit feedback; optimizes that observed items are scored above unobserved ones — a standard CF baseline.

```pseudo
Algorithm: User-based Neighborhood CF (rating prediction)
──────────────────────────────────────────────────────────
Input: interaction matrix R, target (u, i), neighborhood size k
1. For every other user v that has rated item i:
     compute sim(u, v)   # e.g. cosine / Pearson over co-rated items
2. N_i(u) ← the k users with highest sim(u, v) that rated i
3. r_hat(u,i) ← (1 / |N_i(u)|) * Σ_{v in N_i(u)} r(v,i)
     # (weighted variant: Σ sim(u,v)·r(v,i) / Σ sim(u,v), mean-centered)
4. return r_hat(u,i)        # rank items by r_hat for top-N recommendation
```

- **Neural Collaborative Filtering ([[Neural Collaborative Filtering|NCF]]):** replaces the fixed dot product with a neural network over user/item embeddings, capturing **non-linear** interactions. He et al. (2017) showed **MF is a special case of NCF** (replace the neural layers with an element-wise multiplication layer, a fixed all-ones output weight, and identity activation $\Rightarrow$ recovers the dot product). NCF treats prediction as binary classification: weighted square loss (explicit) or binary cross-entropy (implicit), with negative sampling.
- **When CF fails:** standard CF/MF treats interactions as an **unordered set** and ignores temporal order — motivating [[Sequential Recommendation]] (e.g. FPMC, GRU4Rec, SASRec). It also suffers the [[Cold Start Problem]]: new users/items have no interactions, so no collaborative signal exists.
- **Beyond accuracy:** purely accuracy-optimal CF tends to recommend popular/similar items, hurting [[Diversity]], [[Novelty]], [[Coverage]], and item-side [[Fairness in Recommendation]] (popularity bias, long-tail under-exposure, filter bubbles).
- **No universal winner:** simple, well-tuned neighborhood CF often matches complex neural models (a recurring reproducibility finding); the best method depends on problem formulation, domain, and available data — hybrids frequently win.

## Connections

- Contrasted with: [[Content-Based Filtering]] (uses item content, not interactions); combined in [[Hybrid Recommendation]]
- Sub-types: [[User-based Collaborative Filtering]], [[Item-based Collaborative Filtering]], [[Neighborhood-based Collaborative Filtering]], [[Memory-based Collaborative Filtering]]
- Model-based instance: [[Matrix Factorization]] $\to$ [[Neural Collaborative Filtering]]
- Built on: [[User-Item Interaction Matrix]], [[Implicit Feedback]] / [[Explicit Feedback]]
- Trained with: [[Bayesian Personalized Ranking (BPR)]], [[Negative Sampling]]
- Limitations lead to: [[Cold Start Problem]], [[Data Sparsity]], [[Popularity Bias]], [[Sequential Recommendation]]
- Evaluated with: [[Recall]], [[MRR]], [[NDCG]], [[Hit Rate]] and beyond-accuracy metrics ([[Diversity]], [[Fairness in Recommendation]])

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03a - Sequential Recommendation Models]]
- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
