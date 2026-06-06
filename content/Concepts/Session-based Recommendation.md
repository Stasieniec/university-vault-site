---
type: concept
aliases: [Session-based Recommender]
course: [RecSys]
tags: [sequential-rec, collaborative-filtering, exam-topic]
status: complete
---

# Session-based Recommendation

## Definition

> [!definition] Session-based Recommendation
> **Session-based recommendation** predicts the next item(s) a user will interact with based **only on the current session's short-term browsing behavior** — the ordered sequence of clicks/views/adds within one visit — rather than a long-lived user profile. It is a paradigm of [[Sequential Recommendation]]: items arrive in chronological order and the model conditions on that order.
>
> Canonical example: a user browsing a phone is shown a phone-case; the recommendation is driven by what is happening *now*, not by who the user is across all time.

## Intuition

> [!intuition] Why "session" and not "user"
> Classic [[Matrix Factorization|MF]] / [[Collaborative Filtering|CF]] treats interactions as an **unordered set** tied to a stable user identity. That fails in two situations session-based RecSys targets:
> - **Anonymous / logged-out users** — there is no persistent profile, only the events seen so far in this session.
> - **Short-term intent that overrides long-term taste** — a user who usually buys rock music is, *right now*, building a workout playlist. The session signal is more informative than the historical average.
>
> So we drop the user vector and instead build a representation of the **session itself** (a short ordered list of items) and decode the next item from it. The split is: [[Sequential Recommendation]] = use interaction **order**; session-based = the special case where the conditioning context is the **current session only** (often with no user ID), as opposed to user-based recommendation which spans a persistent history.

## Mathematical Formulation

Given the current session as an ordered sequence of interacted items $s = \langle i_1, i_2, \dots, i_t \rangle$, the task is **next-item prediction**: model the conditional distribution over the catalog $I$ for the next position,

$$P(i_{t+1} \mid i_1, i_2, \dots, i_t)$$

A simple first-order [[Markov Chain]] approximates this with only the last item, $P(i_{t+1}\mid i_t)$, estimated from a global transition matrix. Modern session-based models instead encode the whole prefix into a hidden state. The first deep model, GRU4Rec [Hidasi et al., 2015], uses a [[Gated Recurrent Unit (GRU)|GRU]] [[Recurrent Neural Network (RNN)|RNN]] that consumes the one-hot items left-to-right:

$$h_t = \mathrm{GRU}(h_{t-1}, \mathbf{e}_{i_t}), \qquad \hat{r}_{s} = h_t W_{\text{out}}$$

where:
- $h_{t-1}, h_t$ — recurrent hidden state before/after consuming item $i_t$; $h_t$ summarizes the session so far
- $\mathbf{e}_{i_t}$ — dense embedding of the current item (each item has its own learned embedding)
- $W_{\text{out}}$ — output projection producing a score $\hat{r}_{s,j}$ for every candidate item $j \in I$
- the highest-scoring items become the recommendation list

GRU4Rec is trained with the pairwise [[Bayesian Personalized Ranking (BPR)|BPR]] loss (a positive next item vs. sampled negatives):

$$\mathcal{L}_{\text{BPR}} = -\frac{1}{N_S} \sum_{j=1}^{N_S} \log \sigma\!\left(\hat{r}_{s,i} - \hat{r}_{s,j}\right)$$

where:
- $\hat{r}_{s,i}$ — score for the **true** next item $i$ in session $s$
- $\hat{r}_{s,j}$ — score for **negative sample** $j$
- $N_S$ — number of negative samples per positive instance
- $\sigma(\cdot)$ — sigmoid; the objective pushes the true next item above sampled negatives

## Key Properties / Variants

- **Conditioning context:** the *current session* only. Contrast with **user-based** sequential recommendation, which conditions on a persistent cross-session history. Both are sub-paradigms of [[Sequential Recommendation]].
- **No (or optional) user ID:** well-suited to anonymous traffic; user/item representations *can* still be augmented with side-information / content features when available.
- **Targets beyond single items:** the next-step output can be a single item, a basket, a bundle, or a playlist (next-basket recommendation).
- **Model lineage** (increasing capacity, all usable as session-based when fed the current session):
  - First-order [[Markov Chain]] / [[FPMC|Factorized Personalized Markov Chains (FPMC)]] — short, sparse sessions; first-order transitions only.
  - GRU4Rec ([[Gated Recurrent Unit (GRU)|GRU]]/[[Recurrent Neural Network (RNN)|RNN]]) — the original deep session-based model; captures short within-session temporal patterns; BPR / TOP1-max loss.
  - [[SASRec|Self-Attentive Sequential Recommendation (SASRec)]] — [[Self-Attention]] with item + positional embeddings and a **causal mask**; faster and stronger than RNN/CNN.
  - [[BERT4Rec]] — **bidirectional** [[Transformer Model|Transformer]] trained with a Cloze/masked-item objective; appends a [MASK] at the end at inference for next-item prediction.
- **Loss matters as much as architecture** (Klenitskiy & Vasilev, 2023): SASRec trained with full cross-entropy or BCE with many negatives ("SASRec+") beats BERT4Rec; too few negatives causes overconfidence. BPR / BCE / CE are model-agnostic.
- **Data sparsity** is the core difficulty for count-based session models; mitigated by skipping, clustering, and mixture-of-orders, or sidestepped by embedding-based deep models.

Generic next-item inference for an encoder-style session model:

```pseudo
Algorithm: Session-based Next-Item Recommendation (inference)
─────────────────────────────────────────────────────────────
Input: current session s = <i_1, ..., i_t>, item embedding table E
  h ← initial state
  for k = 1 ... t:                 # consume the session in order
    h ← Encoder(h, E[i_k])         # GRU step, or self-attention over prefix
  scores ← h · E^T                 # score ALL candidate items
  mask out items already in s (optional, domain-dependent)
  return Top-K items by score
```

## Connections

- Sub-paradigm of: [[Sequential Recommendation]] (uses interaction order); contrast with user-based history
- Departs from: [[Matrix Factorization]] / [[Collaborative Filtering]] (which ignore order)
- Probabilistic basis: [[Markov Chain]], [[FPMC]]
- Deep models: [[Gated Recurrent Unit (GRU)]], [[Recurrent Neural Network (RNN)]], [[SASRec]], [[BERT4Rec]], [[Self-Attention]], [[Transformer Model]]
- Trained with: [[Bayesian Personalized Ranking (BPR)]], [[Negative Sampling]]
- Output framing: [[Next-Item Prediction]], [[Top-K Recommendation]]
- Evaluated with: [[Recall]], [[MRR]], [[NDCG]], [[Hit Rate]] (and [[Beyond-Accuracy Metrics]] like [[Diversity]])

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03a - Sequential Recommendation Models]]
