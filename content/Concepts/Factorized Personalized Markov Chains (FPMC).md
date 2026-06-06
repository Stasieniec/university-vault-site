---
type: concept
aliases: [FPMC]
course: [RecSys]
tags: [collaborative-filtering, sequential-rec, exam-topic]
status: complete
---

# Factorized Personalized Markov Chains (FPMC)

## Definition

> [!definition] FPMC
> **FPMC** [Rendle et al., 2010] is a model for **next-basket / next-item** [[Sequential Recommendation]] that combines two signals into a single score:
> - a **long-term** [[Matrix Factorization]] term capturing user *u*'s general affinity for a candidate item *j*, and
> - a **short-term** factorized **first-order** [[Markov Chain]] term capturing the likelihood of **transitioning** from the last item *i* to the candidate *j*.
>
> The key idea is **personalized** transitions: instead of one global [[Transition Matrix]] shared by everyone, FPMC gives each user their own transition behavior. Because per-user transition matrices are extremely sparse, FPMC **factorizes the user × item × item transition cube** so that parameters are shared across users and items.

## Intuition

> [!intuition] Why factorize the transition cube?
> A naive **personalized** Markov chain would store, for every user *u*, a full $|I| \times |I|$ transition matrix $T^{(u)}$ — "given user *u* just bought *i*, how likely is *j* next?". With millions of items and few interactions per user, almost every entry is unobserved (a "?"). Direct estimation is hopeless.
>
> [[Matrix Factorization]] solves this the same way it solves sparse rating prediction: it replaces explicit entries with **low-rank latent factors**. Each item gets a "previous-item" embedding $R_i$ and a "next-item" embedding $S_j$; a transition score is their dot product $R_i^\top S_j$. Observations from one user now inform transitions for *all* users, because they share the same item embeddings. FPMC then **adds** this short-term transition signal to a standard MF long-term preference signal $P_u^\top Q_j$, so the final recommendation reflects both *what the user generally likes* and *what naturally follows their last item*.

## Mathematical Formulation

FPMC scores a candidate **next item *j*** for **user *u*** who last interacted with **item *i***:

$$
\hat{x}_{u,i,j} \;=\; \underbrace{P_u^\top Q_j}_{\text{long-term (MF)}} \;+\; \underbrace{R_i^\top S_j}_{\text{short-term (transition)}}
$$

where:
- $P_u \in \mathbb{R}^{k}$ — latent factor vector for **user *u*** (long-term preference).
- $Q_j \in \mathbb{R}^{k}$ — latent factor vector for the **candidate item *j*** in the user term.
- $R_i \in \mathbb{R}^{k}$ — latent factor for the **previous item *i*** (the "from" side of the transition).
- $S_j \in \mathbb{R}^{k}$ — latent factor for the **candidate item *j*** in the transition term (the "to" side).
- $k$ — embedding dimensionality (model capacity; results improve as $k$ grows).

The two dot products are the **rank-$k$ factorizations** of two pairwise interactions extracted from the full personalized transition cube $\mathcal{X} \in \mathbb{R}^{|U| \times |I| \times |I|}$: the **user–next-item** matrix ($P,Q$) and the **prev-item–next-item** matrix ($R,S$). (Rendle's full model also includes a user–prev-item factor, but it is shown empirically to be dropped without loss, leaving the two terms above.)

**Training — S-BPR (pairwise ranking).** FPMC is fit with [[Bayesian Personalized Ranking (BPR)|BPR]] adapted to sequences (S-BPR), optimized by [[Stochastic Gradient Descent|SGD]]. For an observed transition (user *u*, last item *i*, true next item *j*) and a sampled negative item $j'$ the user did **not** take next, it maximizes the probability that the positive is ranked above the negative:

$$
\mathcal{L}_{\text{S-BPR}} \;=\; -\sum_{(u,i,j,\,j')} \ln \sigma\!\big(\hat{x}_{u,i,j} - \hat{x}_{u,i,j'}\big) \;+\; \lambda \,\lVert \Theta \rVert^2
$$

where:
- $\sigma(\cdot)$ — logistic sigmoid.
- $\hat{x}_{u,i,j}$ / $\hat{x}_{u,i,j'}$ — FPMC scores of the positive next item *j* and a sampled negative $j'$.
- $\lambda \lVert \Theta \rVert^2$ — L2 [[Regularization]] over all factor matrices $\Theta = \{P, Q, R, S\}$.

## Key Properties / Variants

- **First-order only.** FPMC conditions the transition on the single **last** item $i$; it captures no longer-range dependencies. This is its core limitation versus RNN/attention models ([[GRU4Rec]], [[SASRec]], [[BERT4Rec]]).
- **Personalization via factorization.** Sharing item embeddings $R, S$ across users is what makes per-user transitions estimable from sparse data — the cube is never materialized.
- **Two ablations recover known models.** Drop the transition term ($R_i^\top S_j$) and FPMC reduces to plain personalized [[Matrix Factorization]] (MF). Drop the user term ($P_u^\top Q_j$) and it reduces to a **factorized (non-personalized) Markov chain** (FMC). FPMC empirically beats both, especially on sparse data: **FPMC > FMC > MF > dense MC > most-popular**.
- **Lightweight and interpretable**, well suited to short interaction sequences and limited compute. When data is **sparse** or the budget is small, FPMC can **outperform** heavier deep models like [[GRU4Rec]]; with abundant data the deep models win.
- **Loss is model-agnostic.** S-BPR is the original choice but FPMC can be trained with other pairwise/listwise/contrastive objectives.

```pseudo
Algorithm: FPMC training (S-BPR + SGD)
──────────────────────────────────────────────
Initialize factor matrices P, Q, R, S ~ N(0, σ²)   # dims |U|×k, |I|×k, |I|×k, |I|×k

Loop until converged:
  Sample an observed transition (u, i, j)            # user u, last item i, true next item j
  Sample a negative item j' not taken next by u
  x_pos  = P_u·Q_j  + R_i·S_j
  x_neg  = P_u·Q_j' + R_i·S_j'
  δ      = 1 - σ(x_pos - x_neg)                       # gradient of -ln σ(·)
  # gradient ascent on the BPR objective (with L2 term λ)
  P_u  += α ( δ (Q_j - Q_j')  - λ P_u )
  Q_j  += α ( δ  P_u          - λ Q_j )
  Q_j' += α ( δ (-P_u)        - λ Q_j')
  R_i  += α ( δ (S_j - S_j')  - λ R_i )
  S_j  += α ( δ  R_i          - λ S_j )
  S_j' += α ( δ (-R_i)        - λ S_j')

Inference: score every candidate j by  x_{u,i,j} = P_u·Q_j + R_i·S_j ; recommend Top-K.
```

## Connections

- Builds on: [[Matrix Factorization]] (long-term term), [[Markov Chain]] / [[Transition Matrix]] (short-term term)
- Trained with: [[Bayesian Personalized Ranking (BPR)]], [[Stochastic Gradient Descent]], [[Regularization]]
- Solves: per-user [[Transition Matrix]] sparsity in Personalized Markov Chains
- Type of: [[Sequential Recommendation]] model (next-item / next-basket prediction)
- Superseded by: [[GRU4Rec]], [[SASRec]], [[BERT4Rec]] (capture higher-order dependencies)
- Contrasts with: [[Collaborative Filtering]] / plain [[Matrix Factorization]] (which ignore interaction order)

## Appears In

- [[RS-L03a - Sequential Recommendation Models]]
