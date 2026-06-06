---
type: concept
aliases: [Memory-based Collaborative Filtering, User-based Collaborative Filtering, Item-based Collaborative Filtering, User-based Rating Prediction]
course: [RecSys]
tags: [collaborative-filtering, exam-topic]
status: complete
---

# Neighborhood-based Collaborative Filtering

## Definition

> [!definition] Neighborhood-based (Memory-based) CF
> **Neighborhood-based collaborative filtering** is the family of [[Collaborative Filtering|CF]] methods that predict a user's preference for an item directly from the **similarity between entities** — either between users (user-based) or between items (item-based) — rather than by training a parametric model. It is also called **memory-based CF** because it keeps the whole [[User-Item Interaction Matrix|user-item interaction matrix]] in memory and computes predictions on the fly using the ratings of the **$k$ nearest neighbors**.
>
> A prediction $\hat{r}_{ui}$ for user $u$ on item $i$ is obtained by aggregating the ratings of the entities most similar to the target. There is **no model fit in advance**; the algorithm is essentially [[ANN|k-nearest neighbors]] applied to the rating matrix.

## Intuition

> [!intuition] "Users who liked X also liked Y"
> The collaborative signal is the collective behavior of the crowd. If Alice and Bob have historically rated the same movies the same way, then Bob's opinion is good evidence for what Alice will think of a movie she has not seen. Two complementary views of the same matrix:
> - **User-based:** "How do users *similar to me* rate item $i$?" — find neighbors among the **rows** (users) and copy their opinion on the target item.
> - **Item-based:** "How do *I* rate items *similar to* $i$?" — find neighbors among the **columns** (items) and copy my own opinion on similar items.
>
> The whole approach rests on the assumption that similar entities behave similarly, and that similarity estimated from past co-ratings transfers to the unobserved cell we want to fill.

## Mathematical Formulation

The task is framed as **matrix completion**: given a sparse $m \times n$ rating matrix $R$ with entry $r_{ui}$ (user $u$'s rating of item $i$, mostly missing), predict a held-out entry $\hat{r}_{ui}$.

**User-based rating prediction** (the simplest form from the lecture) estimates $\hat r_{ui}$ as the average rating that $u$'s nearest neighbors gave to item $i$:

$$\hat{r}_{ui} = \frac{1}{|\mathcal{N}_i(u)|} \sum_{v \in \mathcal{N}_i(u)} r_{vi}$$

where:
- $\mathcal{N}_i(u)$ — the set of the $k$ **nearest neighbors** of user $u$ that have actually rated item $i$
- $r_{vi}$ — the rating that neighbor $v$ gave to item $i$
- $|\mathcal{N}_i(u)|$ — the number of such neighbors (the normalizer)

The neighbor set $\mathcal{N}_i(u)$ is selected by a **similarity function** $\mathrm{sim}(u, v)$ between the two users' rating vectors. A standard, more general estimator weights each neighbor by similarity instead of taking a flat average:

$$\hat{r}_{ui} = \frac{\sum_{v \in \mathcal{N}_i(u)} \mathrm{sim}(u, v)\, r_{vi}}{\sum_{v \in \mathcal{N}_i(u)} |\mathrm{sim}(u, v)|}$$

A common similarity is **cosine similarity** over the co-rated items:

$$\mathrm{sim}(u, v) = \frac{\sum_{i} r_{ui}\, r_{vi}}{\sqrt{\sum_{i} r_{ui}^2}\,\sqrt{\sum_{i} r_{vi}^2}}$$

(Pearson correlation is the mean-centered variant.) The **item-based** predictor is exactly symmetric: swap the roles of users and items, find $\mathcal{N}_u(i)$ = items similar to $i$ that $u$ has rated, and aggregate $u$'s own ratings on those items weighted by item-item similarity.

## Key Properties / Variants

- **Two dual views of the same matrix.** User-based finds neighbors over rows; item-based finds neighbors over columns. **Item-based CF is usually preferred** because it is **more stable** — item-item relationships drift slower over time than individual user profiles, and there are often fewer items than users.
- **Memory-based, not model-based.** No model is fit in advance; predictions rely directly on stored interactions and a similarity measure. Contrast with model-based CF (e.g., [[Matrix Factorization]], [[Neural Collaborative Filtering|NCF]]) which **trains** latent representations from the data.
- **$k$ is the key hyperparameter.** With $k = 1$ the prediction is just the single nearest neighbor's rating (e.g., in the lecture's toy example, predicting Eric's rating of "Titanic" with Lucy as sole neighbor gives 5).
- **Advantages:** simple, efficient on small data, and **transparent / explainable** ("recommended because users like you liked it").
- **Drawbacks:** **sparsity** (few co-rated items make similarities unreliable), **noise**, and **scalability** — naive neighbor search is costly on giant catalogs and user bases (mitigated with [[ANN Search|approximate nearest neighbor]] search).
- **Ignores order.** Like all CF/MF over a static matrix, it treats interactions as an unordered set; temporal order, recency, and transitions are lost. This is precisely the gap that [[Sequential Recommendation]] models fill.

```pseudo
Algorithm: User-based k-NN Rating Prediction
─────────────────────────────────────────────
Input: rating matrix R, target (u, i), neighborhood size k
Output: predicted rating r̂_ui

1. Candidates ← { v ≠ u : v has rated item i }
2. For each v in Candidates:
     compute sim(u, v)        // e.g., cosine / Pearson over co-rated items
3. N ← top-k candidates by sim(u, v)        // the neighborhood N_i(u)
4. r̂_ui ← ( Σ_{v in N} sim(u,v) · r_vi ) / ( Σ_{v in N} |sim(u,v)| )
     // or the flat average  (1/|N|) Σ_{v in N} r_vi
5. return r̂_ui

(Item-based: swap users <-> items; neighbors = items similar to i
 that u has rated, aggregate u's own ratings on them.)
```

## Connections

- Subtype of: [[Collaborative Filtering]] (the neighborhood / memory-based branch)
- Contrasted with: [[Matrix Factorization]] and [[Neural Collaborative Filtering]] (model-based CF that learns latent factors)
- Operates on: [[User-Item Interaction Matrix]] (framed as matrix completion)
- Core mechanism: [[ANN|k-nearest neighbors]]; scaled with [[ANN Search|approximate nearest neighbor]] search
- Suffers from: [[Cold Start Problem]] and [[Data Sparsity]]
- Evaluated with: [[Recall]], [[MRR]], [[NDCG]] (ranking / accuracy metrics)
- Motivates: [[Sequential Recommendation]] (adds the interaction order CF ignores)
- Alternative paradigms: [[Content-Based Filtering]], [[Hybrid Recommendation]]

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L03a - Sequential Recommendation Models]]
