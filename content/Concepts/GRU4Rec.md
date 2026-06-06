---
type: concept
aliases: []
course: [RecSys]
tags: [sequential-rec, collaborative-filtering, exam-topic]
status: complete
---

# GRU4Rec

## Definition

> [!definition] GRU4Rec
> **GRU4Rec** [Hidasi et al., 2015] was among the **first deep learning models** for [[Sequential Recommendation]], originally designed for **[[Session-based Recommendation|session-based]]** recommendation. It uses a [[Gated Recurrent Unit (GRU)]] — a type of [[Recurrent Neural Network (RNN)]] — to encode a **chronologically ordered sequence** of item interactions into a hidden state, then scores all catalogue items for the **next item**. Unlike [[FPMC]] (which captures only first-order item→item transitions), the GRU's recurrent hidden state can in principle summarise the **entire session** so far.

## Intuition

> [!intuition] Why an RNN for sequences?
> [[Matrix Factorization|MF]] / [[Collaborative Filtering|CF]] treat a user's interactions as an **unordered set** — they ignore order, recency, and transitions. But order matters: you buy a phone case *after* the phone, watch Star Wars IV → V → VI in order, and your taste drifts over years. A Markov chain only looks back $k$ steps; an RNN instead carries a **running summary** (the hidden state $h_t$) that is updated at every step and, through the GRU's gates, can keep relevant signal alive across many steps. So at each position the model has a single vector encoding "everything that happened in the session up to now," which it turns into a score for every candidate next item.

## Mathematical Formulation

A session is a sequence of items $\langle i_1, i_2, \dots, i_t \rangle$ fed in one at a time. Each step’s item is encoded as a **1-of-N (one-hot)** vector, mapped through an embedding, and processed by the GRU, whose hidden state is updated by the standard gating equations:

$$
\begin{aligned}
z_t &= \sigma\!\left(W_z x_t + U_z h_{t-1}\right) \\
r_t &= \sigma\!\left(W_r x_t + U_r h_{t-1}\right) \\
\hat{h}_t &= \tanh\!\left(W x_t + U\,(r_t \odot h_{t-1})\right) \\
h_t &= (1 - z_t)\odot h_{t-1} + z_t \odot \hat{h}_t
\end{aligned}
$$

The final hidden state is passed through feedforward layer(s) to produce a **score per candidate item**, $\hat{r}_{s,i}$. Training uses the pairwise **[[Bayesian Personalized Ranking (BPR)]]** loss (the original paper also introduces the TOP1 / TOP1-max alternative):

$$
\mathcal{L}_{\text{BPR}} = -\frac{1}{N_S}\sum_{j=1}^{N_S} \log \sigma\!\left(\hat{r}_{s,i} - \hat{r}_{s,j}\right)
$$

where:
- $x_t$ — embedding of the current input item (from the one-hot encoding of $i_t$)
- $h_{t-1}, h_t$ — GRU hidden state before/after step $t$ (the session summary)
- $z_t$ — **update gate**: how much of the new candidate state $\hat{h}_t$ to write into $h_t$
- $r_t$ — **reset gate**: how much past state to forget when forming $\hat{h}_t$
- $\hat{h}_t$ — candidate (proposed) hidden state at step $t$
- $\sigma(\cdot)$ — sigmoid; $\odot$ — element-wise product
- $\hat{r}_{s,i}$ — score of the **true next item** $i$ for session $s$
- $\hat{r}_{s,j}$ — score of a **negative sample** $j$
- $N_S$ — number of negative samples per positive instance
- **Interpretation of the loss:** push the score of the true next item above the scores of sampled negatives (pairwise ranking).

## Key Properties / Variants

- **Architecture (bottom→top):** one-hot item → **embedding layer** (each item has its own dense embedding) → one or more **stacked GRU layers** (recurrent self-connections capture sequential dependencies) → feedforward layer(s) → **scores over items**.
- **Next-item prediction** by **scoring catalogue items directly** — same score-and-rank skeleton as [[SASRec]] and [[BERT4Rec]]; the models differ only in how they encode the history $\mathcal{H}_t$.
- **Left-to-right / unidirectional**, like SASRec but unlike the bidirectional BERT4Rec.
- **When to use it:** outperforms [[FPMC]] especially when **more data is available**, and handles **longer sequences** and more complex modelling than FPMC. When data is **sparse**, the model must be simple, or the compute budget is limited, **FPMC can win**.
- **Limitations (lecture comparison table):** effective at **short temporal patterns within a session**, but requires a lot of **training time** and **struggles with long sequences**.
- **Losses are model-agnostic:** BPR/BCE/CE can all be applied to GRU4Rec; alternatives include TOP1-max (GRU4Rec paper), listwise LambdaRank-style losses, and contrastive InfoNCE. Reproducibility studies stress that the **loss and the number of negatives** matter as much as the architecture (e.g., "GRU4Rec (our)" with a strong loss is competitive with BERT4Rec on ML-1M).
- It appears as a standard sequential baseline throughout the course (e.g., compared against in SASRec/BERT4Rec/TIGER experiments and used as a baseline in the FairDiverse toolkit).

```pseudo
Algorithm: GRU4Rec next-item scoring (one session)
────────────────────────────────────────────────────
Input: session sequence ⟨i_1, ..., i_t⟩
h_0 ← 0
for k = 1 .. t:
    x_k ← Embed(one_hot(i_k))          # item embedding
    z_k ← σ(W_z x_k + U_z h_{k-1})     # update gate
    r_k ← σ(W_r x_k + U_r h_{k-1})     # reset gate
    h~_k ← tanh(W x_k + U (r_k ⊙ h_{k-1}))
    h_k ← (1 - z_k) ⊙ h_{k-1} + z_k ⊙ h~_k
scores ← FeedForward(h_t)              # one score per candidate item
return top-K items by score
# Train with pairwise BPR (or TOP1-max) over sampled negatives
```

## Connections

- Is a: [[Sequential Recommendation]] model / [[Session-based Recommendation|session-based recommender]]
- Built on: [[Gated Recurrent Unit (GRU)]], a [[Recurrent Neural Network (RNN)]]
- Input via: [[Embedding Layer]] over [[Atomic Item IDs|atomic item ids]]
- Trained with: [[Bayesian Personalized Ranking (BPR)]] loss (pairwise), with [[Negative Sampling]]
- Improves on: [[FPMC]] (first-order [[Markov Chain]]), [[Matrix Factorization]] / [[Collaborative Filtering]]
- Compared with: [[SASRec]] (self-attention), [[BERT4Rec]] (bidirectional) — both also do [[Next-Item Prediction]] by scoring
- Evaluated with: [[Recall]], [[MRR]], [[NDCG]], [[HR@K]]; appears as a baseline in [[Fairness in Recommendation|fairness]] studies

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03a - Sequential Recommendation Models]]
- [[RS-L04 - Generative Recommendation]]
