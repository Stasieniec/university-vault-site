---
type: concept
aliases: [Implicit Feedback, Explicit Feedback]
course: [RecSys]
tags: [collaborative-filtering, evaluation, exam-topic]
status: complete
---

# Implicit and Explicit Feedback

## Definition

> [!definition] Implicit and Explicit Feedback
> The two kinds of preference signal a [[Recommender System]] learns from.
> - **Explicit feedback** is a *deliberate, stated* judgement of preference: a 1–5 star rating, a thumbs up/down, a review score. The value is (roughly) on an interpretable preference scale, and a missing entry means "not rated", i.e. genuinely **unknown**.
> - **Implicit feedback** is an *indirect behavioural* signal interpreted as preference: a click, play, watch-duration, purchase, skip, add-to-playlist. It is observed as a by-product of using the system. The signal is (usually) **positive-only** — we see what the user *did*, never an explicit "I dislike this".
>
> Both are recorded in the user–item [[Interaction Matrix]] $R$ with users $U=\{u_1,\dots,u_n\}$ and items $I=\{i_1,\dots,i_m\}$, but the *semantics of an observed value* and of a *missing value* differ, which changes how we model and evaluate.

## Intuition

> [!intuition] What does a missing cell mean?
> The crux of the distinction is the meaning of an **unobserved** entry $r_{ui}$.
> - With **explicit** ratings, an unrated cell is **missing data**: the user simply has not told us. We should *not* treat it as a negative. Models are fit on the **observed** entries only.
> - With **implicit** feedback, there is no "negative" channel. A user who never clicked an item might (a) dislike it, or (b) never have *seen* it. Treating all non-interactions as negatives is wrong (most are unseen), but ignoring them entirely leaves only positive examples and the model collapses to predicting "everything is good".
>
> Practical reality (RS-L01 case studies): explicit ratings are **scarce, expensive, and biased** (few users rate; those who do are not a random sample). Implicit signals are **abundant and cheap** — every play, skip, and purchase is logged — which is why production systems (Spotify Daily Mix, bol.com deals, YouTube feed) run almost entirely on implicit feedback. The price is **noise and ambiguity**: a play could be a mis-click, a purchase could be a gift.

## Mathematical Formulation

The same architecture can be trained on either signal; the **loss function** is what differs. RS-L01 frames [[Neural Collaborative Filtering]] as binary classification with label $y_{ui}$ and prediction $\hat{y}_{ui}$, and gives two losses:

> [!formula] Explicit feedback — (weighted) squared loss
> $$\mathcal{L}_{\text{exp}} = \sum_{(u,i)\,\in\,\mathcal{O}} w_{ui}\,\bigl(r_{ui}-\hat{r}_{ui}\bigr)^2$$
> where:
> - $\mathcal{O}$ — set of **observed** (rated) user–item pairs; the sum runs over these *only*
> - $r_{ui}$ — the actual rating (e.g. on a 1–5 scale)
> - $\hat{r}_{ui}$ — predicted rating (e.g. dot product $\bar{u}_u\cdot\bar{v}_i$ in [[Matrix Factorization]], or NCF output)
> - $w_{ui}$ — optional per-entry weight (set $w_{ui}=1$ for plain squared error)

> [!formula] Implicit feedback — binary cross-entropy with negative sampling
> $$\mathcal{L}_{\text{imp}} = -\!\!\sum_{(u,i)\,\in\,\mathcal{O}^{+}}\!\log\hat{y}_{ui}\;-\!\!\sum_{(u,j)\,\in\,\mathcal{O}^{-}}\!\log\bigl(1-\hat{y}_{uj}\bigr)$$
> where:
> - $\mathcal{O}^{+}$ — observed interactions, treated as positives ($y_{ui}=1$)
> - $\mathcal{O}^{-}$ — **sampled** unobserved pairs treated as negatives ($y_{uj}=0$); drawn by [[Negative Sampling]] to avoid summing over the huge set of all non-interactions
> - $\hat{y}_{ui}\in(0,1)$ — predicted probability that item $i$ is relevant to $u$ (sigmoid output)

An alternative for implicit data is to **rank** a positive above a sampled negative rather than classify each. This is [[Bayesian Personalized Ranking (BPR)]] (Rendle et al., 2012), the canonical implicit-feedback objective:

> [!formula] BPR — pairwise ranking from implicit feedback
> $$\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j)\,\in\,D_S}\log\sigma\bigl(\hat{r}_{ui}-\hat{r}_{uj}\bigr)$$
> where:
> - $D_S$ — triples $(u,i,j)$ with $i$ an **observed** (positive) item for $u$ and $j$ a **sampled unobserved** item
> - $\hat{r}_{ui},\hat{r}_{uj}$ — model scores for the positive and negative item
> - $\sigma(\cdot)$ — sigmoid; the loss pushes the positive's score above the negative's, encoding "$i \succ_u j$" rather than fitting an absolute value

## Key Properties / Variants

- **Missing-data semantics** (the central exam point):
  - Explicit → missing = *unknown*; fit on observed entries only; a regression/rating-prediction task evaluated with error metrics (the implicit assumption being "missing at random", which is itself questionable).
  - Implicit → missing = *unlabeled* (mostly unseen, some disliked); a ranking / one-class task; you **must** synthesise negatives ([[Negative Sampling]]) because there is no negative channel.
- **Task framing.** Explicit feedback naturally suits **rating prediction** ($\hat{r}_{ui}$, evaluated by RMSE/MAE). Implicit feedback naturally suits **[[Top-K Recommendation]]** / [[Next-Item Prediction]], evaluated by ranking metrics — [[Recall]]/[[Hit Rate|HR@K]], [[MRR]], [[NDCG]] (RS-L01/RS-L02). The modern course default is implicit + ranking.
- **Confidence weighting.** Implicit signals carry *strength*: watching a film twice or purchasing is stronger evidence than a single click. A common trick is to weight the positive term by an interaction-derived confidence $c_{ui}$ (e.g. $c_{ui}=1+\alpha\,\text{count}_{ui}$), the implicit analogue of $w_{ui}$ above.
- **Bias.** Implicit logs are **not** an unbiased sample of preference: they are filtered through what the system already exposed and through [[Position Bias]] (RS-L02 motivates simulation and debiasing precisely because logged implicit data is biased). Explicit ratings carry **selection bias** (users rate what they feel strongly about).
- **Where each model sits:**
  - [[Matrix Factorization]] / user-based rating prediction — classically explicit (squared loss on observed ratings).
  - [[Neural Collaborative Filtering]] — either, via the loss switch above.
  - [[Bayesian Personalized Ranking (BPR)]], [[GRU4Rec]] (BPR loss), [[SASRec]] (BCE + negatives), [[BERT4Rec]] (masked CE over the catalogue) — implicit / sequential, learned from interaction order.
  - [[Generative Recommendation]] (TIGER, OneRec) — implicit by construction: trained on the *sequence of items the user actually interacted with*, decoding the next item's identifier.

```pseudo
Choosing the objective from the feedback type
─────────────────────────────────────────────
if feedback is EXPLICIT (ratings / scores):
    task   = rating prediction
    train  = squared loss over OBSERVED entries only      (do NOT impute 0)
    eval   = RMSE / MAE   (and/or rank the predicted ratings)
else feedback is IMPLICIT (clicks / plays / purchases):
    positives = observed interactions
    negatives = SAMPLE from unobserved pairs              (negative sampling)
    train  = BCE  (pointwise)   or   BPR  (pairwise ranking)
    weight positives by confidence c_ui if signal strength varies
    eval   = ranking metrics: Recall/HR@K, MRR, NDCG
    beware: logged implicit data is BIASED (exposure / position)
```

## Connections

- Recorded in: [[Interaction Matrix]] / [[User-Item Interaction]]
- Trained via: [[Negative Sampling]], [[Bayesian Personalized Ranking (BPR)]], [[Matrix Factorization]], [[Neural Collaborative Filtering]]
- Drives task choice: [[Top-K Recommendation]] vs rating prediction; [[Next-Item Prediction]] in [[Sequential Recommendation]]
- Evaluated with: [[Recall]], [[Hit Rate]], [[MRR]], [[NDCG]] (ranking) for implicit; error metrics for explicit
- Confounded by: [[Position Bias]], [[Popularity Bias]] (implicit logs are not unbiased preference samples)
- Underlies: [[Collaborative Filtering]], [[GRU4Rec]], [[SASRec]], [[BERT4Rec]], [[Generative Recommendation]]

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03a - Sequential Recommendation Models]]
- [[RS-L04 - Generative Recommendation]]
