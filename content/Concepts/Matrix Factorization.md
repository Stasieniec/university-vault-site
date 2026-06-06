---
type: concept
aliases: [MF, Rank-2 Matrix Factorization]
course: [RecSys]
tags: [collaborative-filtering, exam-topic]
status: complete
---

# Matrix Factorization

## Definition

> [!definition] Matrix Factorization
> **Matrix factorization (MF)** is a **model-based** [[Collaborative Filtering]] method that decomposes the (sparse) [[User-Item Interaction Matrix]] $R$ into two **lower-dimensional** factor matrices: an $m \times k$ user matrix $U$ and an $n \times k$ item matrix $V$, so that $R \approx U V^\top$. Each user and each item is represented by a $k$-dimensional **latent factor** vector, and a predicted rating is the **dot product** of the corresponding user and item factors. MF turns recommendation into a **matrix-completion** problem: fill in the missing entries of $R$ from a small number of shared latent dimensions.

## Intuition

> [!intuition] Compress users and items into shared latent concepts
> A full ratings matrix with $m$ users and $n$ items has up to $m \cdot n$ free entries — far more than we ever observe. MF assumes the matrix is really **low-rank**: a handful of hidden "concepts" (e.g. in the rank-2 toy example, a **history** axis and a **romance** axis) explains most of the preferences. A user is a point in this concept space (how much they like history vs. romance), an item is a point in the same space (how much it *is* history vs. romance), and how much a user likes an item is just how **aligned** their two vectors are (the dot product). Because all users and all items share the same $k$ axes, observing one user's ratings constrains the factors and lets us **generalize** to unseen user-item pairs — this is what makes inference efficient and beats raw neighbourhood lookup on sparse data.

## Mathematical Formulation

> [!formula] Low-Rank Factorization and Rating Prediction
> $$R \approx U V^\top, \qquad \hat{r}_{ij} = \bar{u}_i \cdot \bar{v}_j = \sum_{f=1}^{k} u_{if}\, v_{jf}$$
>
> where:
> - $R \in \mathbb{R}^{m \times n}$ — user-item ratings/interaction matrix ($m$ users, $n$ items), mostly missing
> - $U \in \mathbb{R}^{m \times k}$ — user factor matrix; row $\bar{u}_i$ is user $i$'s latent vector
> - $V \in \mathbb{R}^{n \times k}$ — item factor matrix; row $\bar{v}_j$ is item $j$'s latent vector
> - $k$ — number of latent factors/concepts ($k \ll m, n$); the chosen **rank**
> - $\hat{r}_{ij}$ — predicted rating, the dot product of the user and item factors

> [!formula] Regularized Squared-Error Objective
> $$\min_{U, V} \sum_{(i,j) \in \mathcal{K}} \left( r_{ij} - \bar{u}_i \cdot \bar{v}_j \right)^2 + \lambda \left( \lVert \bar{u}_i \rVert^2 + \lVert \bar{v}_j \rVert^2 \right)$$
>
> where:
> - $\mathcal{K}$ — set of **observed** (user, item) entries only (we never fit the missing ones)
> - $r_{ij}$ — the observed rating; $\bar{u}_i \cdot \bar{v}_j$ — the reconstruction
> - $\lambda$ — L2 [[Regularization]] strength, prevents overfitting the sparse observations
>
> The "general recipe" from lecture: **(1) define a model** ($\hat{r}_{ij} = \bar{u}_i \cdot \bar{v}_j$), **(2) define an objective** (the loss above), **(3) optimize** (typically [[Stochastic Gradient Descent|SGD]] over observed entries, or alternating least squares).

For the **rank-2** worked example, the reconstruction is a sum over the two interpretable concepts:

$$r_{ij} \approx \underbrace{(\text{user } i\text{'s affinity to } \textit{history}) \cdot (\text{item } j\text{'s } \textit{history})}_{\text{factor 1}} + \underbrace{(\text{user } i\text{'s affinity to } \textit{romance}) \cdot (\text{item } j\text{'s } \textit{romance})}_{\text{factor 2}}$$

## Key Properties / Variants

- **Latent factors can be interpretable.** In the rank-2 example, columns of $U$ correspond to *history* and *romance* affinity; rows of $V^\top$ score each film on those axes. The Google-style 2D embedding example uses *arthouse↔blockbuster* and *children's↔adult's* axes — but in general the axes are learned and not human-labelled.
- **Only observed entries are fit.** The sum runs over $\mathcal{K}$ (known cells); MF imputes the rest. This is the matrix-completion view.
- **Latent dimensionality is a capacity knob.** Larger $k$ = more expressive but more parameters and overfitting risk; performance typically rises with embedding dimensionality (seen in FPMC's F-measure-vs-dimensionality curves).
- **Linear by construction.** The dot product captures only **linear** user-item interactions; this is the motivation for neural extensions.
- **Generalization of MF — [[Neural Collaborative Filtering]] (NCF)** [He et al., 2017]: replaces the fixed dot product with non-linear neural layers over the user/item embeddings. MF is the **special case** of NCF where the "neural CF layers" become a single element-wise multiplication layer with a fixed all-ones output weight $J_{k\times1}$ and identity activation — recovering exactly $p_u \cdot q_i$. NCF is trained as binary classification (weighted square loss for [[Explicit Feedback]], binary cross-entropy for [[Implicit Feedback]]) with [[Negative Sampling]].
- **Deep MF (DMF)** [Xue et al., 2017]: optimizes the factorization with deep neural networks (a FairDiverse non-LLM baseline).
- **Pairwise ranking variant — [[Bayesian Personalized Ranking (BPR)]]**: instead of squared error on ratings, optimizes a pairwise ranking loss for [[Implicit Feedback]] (BPR-MF), pushing observed items above unobserved ones.
- **MF fails on order.** MF treats interactions as an **unordered set** — it ignores sequence, recency, repetition, and item-to-item transitions. This motivates [[Sequential Recommendation]].
- **MF lives inside FPMC.** [[Factorized Personalized Markov Chains (FPMC)]] combines a long-term MF term $P_u^\top Q_j$ (standard user-item factorization) with a factorized short-term item-to-item transition term $R_i^\top S_j$, so MF is literally one component of the FPMC score.

```pseudo
Algorithm: MF training via SGD (regularized squared error)
───────────────────────────────────────────────────────────
Initialize U (m×k), V (n×k) with small random values
Loop for each epoch:
  Shuffle observed entries K
  For each observed (i, j) with rating r_ij in K:
    e_ij ← r_ij − dot(u_i, v_j)          # prediction error
    u_i  ← u_i + α (e_ij · v_j − λ · u_i) # gradient step on user factor
    v_j  ← v_j + α (e_ij · u_i − λ · v_j) # gradient step on item factor
  until convergence
Predict unseen (i, j):  r_hat_ij ← dot(u_i, v_j)
```

## Connections

- Subtype of: [[Collaborative Filtering]] (specifically [[Memory-based Collaborative Filtering|model-based CF]], contrasted with [[Neighborhood-based Collaborative Filtering]])
- Operates on: [[User-Item Interaction Matrix]]; addresses [[Data Sparsity]] and [[Cold Start]] partially via shared latent factors
- Generalized by: [[Neural Collaborative Filtering]] (MF = special case of NCF)
- Optimized with: [[Stochastic Gradient Descent]], [[Regularization]]; ranking variant uses [[Bayesian Personalized Ranking (BPR)]]
- Component of: [[Factorized Personalized Markov Chains (FPMC)]]
- Limitation motivates: [[Sequential Recommendation]] (MF ignores interaction order)
- Evaluated with: [[Recall]], [[MRR]], [[NDCG]], [[Hit Ratio]] (top-K ranking metrics)
- Feedback types: [[Explicit Feedback]] vs [[Implicit Feedback]]

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03a - Sequential Recommendation Models]]
- [[RS-L04 - Generative Recommendation]]
