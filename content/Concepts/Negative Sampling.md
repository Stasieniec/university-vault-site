---
type: concept
aliases: []
course: [RecSys]
tags: [collaborative-filtering, sequential-rec, evaluation, exam-topic]
status: complete
---

# Negative Sampling

## Definition

> [!definition] Negative Sampling
> **Negative sampling** is a training trick for [[Implicit Feedback]] recommenders: instead of treating *every* un-interacted user–item pair as a negative, the model draws a small random **subset** of unobserved (user, item) pairs to serve as negatives for each observed positive. It converts an intractable "score-all-items" objective into a cheap mini-batch objective while still teaching the model to rank positives above negatives.

## Intuition

> [!intuition] Why we sample negatives
> With [[Implicit Feedback]] (clicks, plays, purchases) we only observe **positives** — the items a user interacted with. The interaction matrix is enormous and almost entirely "missing," so naively labelling all $|I|$ items a user did *not* touch as 0 gives a hugely imbalanced classification problem with $O(|U|\cdot|I|)$ training instances. Computing a full-vocabulary softmax/loss over millions of items is too expensive.
>
> The trick: for each observed positive, sample only a handful of unobserved items as "negatives." The gradient still pushes the positive's score up and the sampled negatives' scores down, so on average the model learns the same ranking — at a tiny fraction of the cost. This is the move behind [[BPR]], [[GRU4Rec]], [[SASRec]], and [[Neural Collaborative Filtering]].

## Mathematical Formulation

For an observed positive interaction $(u, i)$, sample a set of negatives $j \sim P_n$ from the items $u$ has **not** interacted with, and optimize a contrast between the positive score $\hat{r}_{u,i}$ and the negative scores $\hat{r}_{u,j}$.

**Pointwise form — Binary Cross-Entropy ([[Implicit Feedback]], used by [[SASRec]] / [[Neural Collaborative Filtering]]):**

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N_S}\sum_{i=1}^{N_S}\Big[\, y_{u,i}\log\hat{y}_{u,i} + (1-y_{u,i})\log\!\big(1-\hat{y}_{u,i}\big)\,\Big]$$

where:
- $N_S$ — number of samples per positive (1 positive + several sampled negatives)
- $y_{u,i}\in\{0,1\}$ — label: $1$ for the observed positive, $0$ for each sampled negative
- $\hat{y}_{u,i}=\sigma(\hat{r}_{u,i})$ — predicted relevance probability (sigmoid of the model score)

**Pairwise form — [[BPR]] / [[GRU4Rec]] loss (rank positive above sampled negative):**

$$\mathcal{L}_{\text{BPR}} = -\frac{1}{N_S}\sum_{j=1}^{N_S}\log\sigma\!\big(\hat{r}_{u,i}-\hat{r}_{u,j}\big)$$

where:
- $i$ — observed positive (true next/relevant item), $j$ — sampled negative
- $\hat{r}_{u,i},\,\hat{r}_{u,j}$ — model scores for positive and negative
- $\sigma(\cdot)$ — sigmoid; the loss is minimized when $\hat{r}_{u,i}\gg\hat{r}_{u,j}$

**Cost contrast.** A full CE objective normalizes over the whole catalog, $P(i\mid u)=\frac{\exp(\hat{r}_{u,i})}{\sum_{k\in I}\exp(\hat{r}_{u,k})}$, costing $O(|I|)$ per example; negative sampling replaces the $\sum_{k\in I}$ with a sum over $N_S\ll|I|$ sampled items.

## Key Properties / Variants

- **Sampling distribution $P_n$:**
  - *Uniform* — every non-interacted item equally likely; simplest, cheap.
  - *Popularity-based* — sample proportional to item frequency (or frequency$^{0.75}$, the word2vec choice); harder, more informative negatives.
  - *In-batch negatives* — reuse other positives in the same mini-batch as negatives (free, common in two-tower / [[Dense Retrieval]] training).
- **[[Hard Negative Mining|Hard negatives]]:** deliberately sample high-scoring (currently "confusable") negatives to sharpen the decision boundary, rather than easy random ones.
- **Number of negatives matters a lot.** Too few negatives can cause **overconfidence** and weak ranking; reproducibility work on sequential models (Klenitskiy & Vasilev, 2023) showed that [[SASRec]] trained with **BCE + ~3000 negatives** (or full CE) beats BERT4Rec — i.e. the *loss + negative count*, not the architecture, drove the reported gap.
- **Bias trade-off:** sampled losses approximate the full softmax/CE objective; more (and harder) negatives reduce bias toward the full-catalog ranking at higher compute cost.
- **Where it appears in training:** used by [[Neural Collaborative Filtering]] (slide 45: "reduce the number of training unobserved instances"), [[BPR]], [[GRU4Rec]], and [[SASRec]] (one positive + several negatives per position with a causal mask).

```pseudo
Algorithm: Training with Negative Sampling (per positive interaction)
─────────────────────────────────────────────────────────────────────
For each observed positive (u, i):           # i = item u interacted with
  Draw N_S negatives {j_1, ..., j_{N_S}} ~ P_n,  j ∉ items(u)
  Compute scores  r_hat(u, i) and r_hat(u, j_k) for all k
  Pointwise:  loss = BCE(label=1, r(u,i)) + Σ_k BCE(label=0, r(u,j_k))
  -- or --
  Pairwise:   loss = -Σ_k log σ( r(u,i) - r(u,j_k) )
  Backpropagate; update embeddings of u, i, and sampled j_k only
```

## Connections

- Trains: [[BPR]], [[GRU4Rec]], [[SASRec]], [[Neural Collaborative Filtering]]
- Essential for: [[Implicit Feedback]] recommendation (only positives observed)
- Contrasts with: full-vocabulary CE / softmax (exact but $O(|I|)$)
- Refinement: [[Hard Negative Mining]] (informative negatives)
- Loss families it plugs into: [[BPR]] (pairwise), BCE (pointwise), [[Contrastive Learning]] (InfoNCE)
- Related task framing: [[Top-N Recommendation]], [[Next-Item Prediction]]

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L03a - Sequential Recommendation Models]]
