---
type: concept
aliases: [Top-K Recommendation, Top-N, Top-K Ranking]
course: [RecSys]
tags: [collaborative-filtering, evaluation, exam-topic]
status: complete
---

# Top-N Recommendation

## Definition

> [!definition] Top-N Recommendation
> **Top-N recommendation** (a.k.a. **top-K recommendation**) is the task of, for each user $u$, producing an **ordered list** of the $N$ items the user is most likely to find relevant, drawn from a large catalog $I$. The output is a **ranking**, not a calibrated rating: only the **relative order** of the returned items matters, and only the **first $N$** items are shown.
> 
> Formally, given users $U = \{u_1, \dots, u_n\}$ and items $I = \{i_1, \dots, i_m\}$, learn a scoring function $\hat{y}(u, i)$ and return the $N$ items with the highest scores (excluding items $u$ has already interacted with). This is the dominant setting for [[Implicit Feedback]] (clicks, plays, purchases), where exact ratings are unavailable.

## Intuition

> [!intuition] Ranking, not rating
> Classic [[Matrix Factorization]] was framed as **rating prediction**: minimize the squared error between $\hat{r}_{ui}$ and a known star rating. But a deployed [[Recommender System]] never shows a user a predicted "4.2 stars" — it shows a *list*. What matters is whether the relevant items land at the **top** of that list.
> 
> Top-N reframes the goal as: get relevant items into the small visible window and order them well. This is why top-N is evaluated with **rank-aware metrics** ([[NDCG]], [[MRR]], [[MAP]], [[HR@K]]) rather than RMSE, and why it is trained with **ranking losses** ([[BPR]]) rather than pointwise regression. A model can have great RMSE yet rank badly, and vice versa — so the objective should match the deployed task.

## Mathematical Formulation

The task is to produce, per user, the ranked top-$N$ set:

$$\text{Top-}N(u) = \operatorname*{arg\,topN}_{i \in I \setminus I_u^{+}} \; \hat{y}(u, i)$$

where:
- $\hat{y}(u, i)$ — relevance score of item $i$ for user $u$ (e.g. dot product $\mathbf{p}_u^\top \mathbf{q}_i$, an [[Neural Collaborative Filtering|NCF]] output, or an autoregressive next-item probability)
- $I_u^{+}$ — items $u$ has already interacted with (excluded so we recommend *new* items)
- $\operatorname{arg\,topN}$ — returns the $N$ highest-scoring items in **ranked order**

**Training objective (pairwise, the canonical top-N loss).** Because there are no negative labels in implicit feedback, [[Bayesian Personalized Ranking (BPR)]] optimizes the *relative order* of an observed (positive) item $i$ over a sampled un-observed item $j$:

$$\mathcal{L}_{\text{BPR}} = -\sum_{(u, i, j) \in D_S} \ln \sigma\!\big(\hat{y}(u,i) - \hat{y}(u,j)\big) + \lambda \lVert \Theta \rVert^2$$

where:
- $D_S = \{(u,i,j) : i \in I_u^{+}, \; j \notin I_u^{+}\}$ — training triples; $i$ is a positive item, $j$ a sampled negative
- $\sigma(x) = 1/(1+e^{-x})$ — sigmoid; pushes the positive score above the negative score
- $\hat{y}(u,i) - \hat{y}(u,j)$ — score margin the model is trained to make positive
- $\lambda \lVert \Theta \rVert^2$ — L2 [[Regularization]] on model parameters $\Theta$

**Pointwise alternative ([[Neural Collaborative Filtering|NCF]]).** Treat top-N as binary classification — label $y_{ui}=1$ for observed interactions, $0$ otherwise — and minimize binary cross-entropy with [[Negative Sampling]]:

$$\mathcal{L}_{\text{BCE}} = -\sum_{(u,i)\in \mathcal{Y}^{+}\cup\mathcal{Y}^{-}} \Big[ y_{ui}\log \hat{y}_{ui} + (1-y_{ui})\log(1-\hat{y}_{ui}) \Big]$$

where $\hat{y}_{ui}=\sigma(\phi(\mathbf{p}_u,\mathbf{q}_i))$ is the predicted relevance probability and $\mathcal{Y}^{-}$ is a set of sampled un-observed (negative) instances.

## Key Properties / Variants

- **Set-based vs rank-aware evaluation.** Set metrics ([[Recall]], [[Precision]], [[HR@K]], [[F1-Score]]) ignore *where* in the list a relevant item sits — two lists with the same relevant items at a cutoff get the same score. Rank-aware metrics ([[NDCG]] via the $1/\log_2(k+1)$ discount, [[MRR]], [[MAP]]) reward placing relevant items higher. The course's motivating example: relevant $=\{B,D\}$; list $A,B,C,D,\dots$ and list $B,D,A,C,\dots$ have equal recall but the second has higher MRR.
- **The "@N" / "@K" cutoff** is intrinsic. Because users see only the top of the list, metrics are reported at a cutoff (HR@10, NDCG@10) reflecting the visible window.
- **Negative sampling is essential.** Implicit feedback has no explicit negatives; un-observed items are sampled as negatives so the score margin (BPR) or the classifier (NCF) can be learned without scoring the full catalog every step.
- **Candidate generation + ranking (two-stage).** At scale, scoring all $m$ items per user is infeasible. A retrieval stage produces a candidate pool (via [[ANN Search|approximate nearest neighbor]] over learned embeddings), then a heavier ranker re-orders it into the final top-N — the cascaded pipeline used industrially.
- **Beyond-accuracy concerns surface at the list level.** Because the list is short, top-N *exacerbates* the [[Long-Tail Distribution]]: popular items dominate exposure ([[Popularity Bias]], [[Exposure Fairness]]). [[Diversity]], [[Novelty]], [[Serendipity]], and [[Coverage]] are list-level objectives that only make sense for a top-N output.
- **Position bias.** Exposure decays with rank position (logarithmic / geometric / cascade browsing models), so a top-N list both reflects and amplifies attention skew — central to [[Fairness in Recommendation]].
- **Model families that produce top-N:** memory-based [[Collaborative Filtering]] (rank by neighbor scores), [[Matrix Factorization]] / [[Neural Collaborative Filtering|NCF]] (rank by score), [[Sequential Recommendation]] ([[SASRec]], [[GRU4Rec]], [[BERT4Rec]]) where top-N = [[Next-Item Prediction]], and [[Generative Recommendation]] where the list is *decoded* token-by-token.

A generic two-stage top-N pipeline:

```pseudo
Algorithm: Serving a Top-N list for user u
──────────────────────────────────────────
Input: user u, catalog I, cutoff N
1. Retrieve candidate set C ⊆ I            # ANN over embeddings, |C| << |I|
2. For each i in C:  s_i ← ŷ(u, i)          # score with ranker
3. Remove items already seen: C ← C \ I_u⁺
4. Sort C by s_i descending
5. (optional) Re-rank C for diversity / fairness  # e.g. MMR, post-processing
6. return first N items of C
```

## Connections

- Output ranked/evaluated by: [[NDCG]], [[MRR]], [[MAP]], [[HR@K]], [[Recall]], [[Precision]]
- Trained with: [[Bayesian Personalized Ranking (BPR)]], [[Negative Sampling]], binary cross-entropy
- Produced by: [[Collaborative Filtering]], [[Matrix Factorization]], [[Neural Collaborative Filtering]], [[Sequential Recommendation]], [[Generative Recommendation]]
- Contrasted with: [[Explicit Feedback]] rating prediction (RMSE), [[Implicit Feedback]] ranking
- Special case: [[Next-Item Prediction]] (top-N with $N{=}1$ over the next step)
- List-level concerns: [[Diversity]], [[Novelty]], [[Serendipity]], [[Coverage]], [[Popularity Bias]], [[Long-Tail Distribution]], [[Position Bias]], [[Exposure Fairness]]
- Scaling: [[ANN Search]], [[Multi-Stage Ranking]]

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03b - From LLMs to LRMs]]
