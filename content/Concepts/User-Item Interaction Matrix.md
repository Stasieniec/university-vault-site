---
type: concept
aliases: [Interaction Matrix, User-Item Interaction]
course: [RecSys]
tags: [collaborative-filtering, exam-topic]
status: complete
---

# User-Item Interaction Matrix

## Definition

> [!definition] User-Item Interaction Matrix
> The **user-item interaction matrix** (or **ratings matrix**) $R$ is an $m \times n$ matrix recording observed interactions between $m$ **users** and $n$ **items**. Each entry $R_{ui}$ holds a user $u$'s feedback on item $i$ — an explicit rating (e.g., $1$–$5$ stars), a signed preference ($+1/0/-1$), or a binary implicit signal (clicked / not clicked). It is the core data structure of [[Collaborative Filtering]]: predictions leverage the *collective* user-item interaction data of a large pool of users, rather than item content.
>
> Formally, given users $U = \{u_1, \ldots, u_m\}$ and items $I = \{i_1, \ldots, i_n\}$, the matrix tabulates which items each user interacted with so the recommender can find the unseen items most pertinent to a given user $u$.

## Intuition

> [!intuition] A mostly-empty grid we have to fill in
> Picture a spreadsheet with users down the rows and items across the columns. Most cells are **blank** — any one user has touched only a tiny fraction of the catalog. The recommendation task is exactly *predicting the missing entries*: estimate how much user $u$ would like the items they have not yet seen, then rank those by predicted value to produce a [[Top-N Recommendation]] list.
>
> The matrix is "collaborative" because the blanks are filled by borrowing signal from *other* users/items: if Lucy and Eric rate the same movies similarly, Lucy's known ratings predict Eric's unknown ones. The structure of the observed entries (who liked what) is the only input pure CF needs — no text, no images, no item metadata.

## Mathematical Formulation

The interaction matrix is the object on which all CF predictors operate. With explicit ratings $r_{ui}$:

$$
R \in \mathbb{R}^{m \times n}, \qquad
R_{ui} =
\begin{cases}
r_{ui} & \text{if } (u,i) \in \mathcal{K} \quad (\text{observed}) \\
? & \text{otherwise} \quad (\text{to be predicted})
\end{cases}
$$

where:
- $m$ — number of users; $n$ — number of items
- $\mathcal{K}$ — set of observed (user, item) pairs; $|\mathcal{K}| \ll m \cdot n$ (the matrix is **sparse**)
- $r_{ui}$ — observed feedback: a star rating, a signed like ($+1/0/-1$), or implicit $\{0,1\}$
- $?$ — a missing entry; recommendation = predicting $\hat{r}_{ui}$ for these cells

**Memory-based read of $R$** — [[User-based Rating Prediction]] fills a blank by averaging the column's ratings over $u$'s nearest neighbors $\mathcal{N}_i(u)$ (users who *did* rate $i$):

$$
\hat{r}_{ui} = \frac{1}{|\mathcal{N}_i(u)|} \sum_{v \in \mathcal{N}_i(u)} r_{vi}
$$

**Model-based read of $R$** — [[Matrix Factorization]] approximates the whole matrix by a low-rank product of latent user/item factors:

$$
R \approx U V^\top, \qquad \hat{r}_{ui} = \bar{u}_u \cdot \bar{v}_i
$$

where $U \in \mathbb{R}^{m \times k}$ (each row a user factor), $V \in \mathbb{R}^{n \times k}$ (each row an item factor), and $k$ is the number of latent concepts. A rating is reconstructed as the dot product of the corresponding row of $U$ and row of $V$; in the lecture's rank-2 toy example the two latent dimensions turn out interpretable ("history" vs. "romance").

## Key Properties / Variants

- **Sparsity.** The dominant practical feature: nearly all of the $m \cdot n$ cells are missing, which drives the choice of algorithm and causes the cold-start problem (a new user/item is an all-blank row/column). See [[Data Sparsity]] and [[Cold Start Problem]].
- **Feedback type.** Entries encode either [[Explicit Feedback]] (numeric ratings; a blank means "not rated") or [[Implicit Feedback]] (clicks/plays/purchases; a blank is ambiguous — disinterest *or* simply not-yet-seen). Implicit matrices are usually treated as binary positives + negative sampling.
- **Asymmetry of the two reads.** Slicing $R$ by **rows** gives [[User-based Collaborative Filtering]] (similar users); slicing by **columns** gives [[Item-based Collaborative Filtering]] (similar items).
- **Missing-not-at-random.** Observed entries are biased — popular items and active users are over-represented — so the blanks are not a random sample. This connects to [[Popularity Bias]] and to evaluating beyond accuracy.
- **Generalization by neural models.** [[Neural Collaborative Filtering]] also takes one-hot user/item indices into $R$ but replaces the dot product with a learned non-linear function; classic MF is a special case of it.
- **Filling-in procedure** (memory-based prediction for one blank cell):

```pseudo
Predict R[u, i] for a missing entry:
──────────────────────────────────────
  candidates ← { v : R[v, i] is observed, v ≠ u }
  for each v in candidates:
      s[v] ← similarity( row_u(R), row_v(R) )   # over co-rated items
  N ← top-k users v by s[v]                       # nearest neighbors N_i(u)
  R_hat[u, i] ← (1 / |N|) * Σ_{v in N} R[v, i]    # (optionally weight by s[v])
  return R_hat[u, i]
```

## Connections

- Core input to: [[Collaborative Filtering]], [[Matrix Factorization]], [[Neural Collaborative Filtering]]
- Read row-wise / column-wise: [[User-based Collaborative Filtering]], [[Item-based Collaborative Filtering]]
- Memory-based predictor over it: [[User-based Rating Prediction]], [[Neighborhood-based Collaborative Filtering]]
- Entry semantics: [[Explicit Feedback]], [[Implicit Feedback]]
- Pathologies of the matrix: [[Data Sparsity]], [[Cold Start Problem]], [[Popularity Bias]]
- Output produced from filled entries: [[Top-N Recommendation]]

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
