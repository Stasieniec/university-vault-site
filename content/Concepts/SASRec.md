---
type: concept
aliases: [Self-Attentive Sequential Recommendation]
course: [RecSys]
tags: [sequential-rec, collaborative-filtering, exam-topic]
status: complete
---

# SASRec

## Definition

> [!definition] SASRec (Self-Attentive Sequential Recommendation)
> **SASRec** [Kang and McAuley, 2018] is a [[Sequential Recommendation|sequential recommender]] that predicts the **next item** in a user's chronologically ordered interaction history using a **unidirectional (causal) [[Transformer Model|Transformer]]** built from [[Self-Attention|self-attention]] blocks. It was the **first sequential recommender to rely solely on self-attention** (no recurrence, no convolution). Each input position is an **item embedding + positional embedding**; a **causal mask** lets each position attend only to itself and earlier items, so the representation at position $t$ is used to predict the item at $t+1$.

## Intuition

> [!intuition] Attention picks out the *relevant* part of the history
> An [[RNN]] like [[GRU4Rec]] compresses the whole history into a single hidden state and must "remember" old items through many recurrent steps, which is slow and forgets long-range signals. SASRec instead lets the model **directly look at every past item** and learn, via attention weights, **which** past interactions matter for the next prediction (e.g. the phone you bought three steps ago is what makes a phone-case relevant now). Because all positions are processed in parallel and the causal mask reuses every prefix as a training example, SASRec is **far faster to train** than RNN/CNN baselines (about an order of magnitude faster per epoch on MovieLens-1M) while reaching higher NDCG@10.
>
> It sits between [[FPMC]] (only first-order transitions) and [[BERT4Rec]] (bidirectional): SASRec captures long-range dependencies but, being **left-to-right**, never conditions on future context.

## Mathematical Formulation

SASRec processes a fixed-length item sequence $s = (s_1, \dots, s_n)$ (left-padded/truncated). The input to the first block is the sum of a learned item embedding and a learned positional embedding:

$$\hat{\mathbf{E}}_t = \mathbf{M}_{s_t} + \mathbf{P}_t$$

A self-attention block applies **scaled dot-product attention with a causal mask**, followed by a **point-wise feed-forward network (FFN)**:

$$\mathbf{S} = \text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} + \mathbf{Mask}\right)\mathbf{V}, \qquad \mathbf{F}_t = \text{FFN}(\mathbf{S}_t)$$

where:
- $\mathbf{M}_{s_t}$ — row of the shared item embedding table $\mathbf{M} \in \mathbb{R}^{|I| \times d}$ for item $s_t$
- $\mathbf{P}_t$ — learned positional embedding for position $t$ ($d$ = latent dimension)
- $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ — query/key/value projections of the (embedded) sequence; SASRec is self-attention, so all three come from the same input
- $\mathbf{Mask}$ — causal mask forcing entry $(t, t')$ to $-\infty$ for $t' > t$, so position $t$ **cannot attend to future items**
- $\mathbf{F}_t$ — block output at position $t$; blocks can be stacked ($b$ layers, with residual connections, layer norm and dropout)

**Scoring.** The relevance of candidate item $i$ at step $t$ is the dot product of the final-layer state with that item's embedding:

$$r_{i,t} = \mathbf{F}_t^{(b)} \, \mathbf{M}_i^\top$$

where:
- $\mathbf{F}_t^{(b)}$ — output state of the last block at position $t$ (the encoded history $s_1, \dots, s_t$)
- $\mathbf{M}_i$ — embedding of candidate item $i$ from the **shared** table $\mathbf{M}$ (input and output embeddings are tied)
- At inference, only the **last** position $\mathbf{F}_n^{(b)}$ is scored against all items $\mathbf{M}$ to rank the next item.

**Training objective.** SASRec is trained with the **binary cross-entropy (BCE) loss** over a true next item (positive) and **negative-sampled** items, applied at every position of the sequence:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N_S}\sum_{i=1}^{N_S}\Big[ y_{s,i}\log \hat{y}_{s,i} + (1 - y_{s,i})\log(1 - \hat{y}_{s,i}) \Big]$$

where:
- $N_S$ — number of samples (positive + sampled negatives) per sequence
- $y_{s,i} \in \{0,1\}$ — ground-truth label (1 for the true next item, 0 for a sampled negative)
- $\hat{y}_{s,i} = \sigma(r_{i,t})$ — predicted score (sigmoid of the dot product)

## Key Properties / Variants

- **Unidirectional / causal**: each position attends only leftward; one forward pass produces a next-item prediction for *every* prefix simultaneously (efficient training via the shared causal mask).
- **Shared, tied item embeddings**: the table $\mathbf{M}$ serves both as input embeddings and as the output projection — scoring is just $\mathbf{F}_t^{(b)}\mathbf{M}_i^\top$. This keeps it a **score-and-rank** model over atomic item ids (contrast with [[Generative Recommendation]], which decodes a [[Semantic IDs|semantic id]] token-by-token instead of scoring a fixed catalogue).
- **Strengths**: balances complexity and efficiency, captures long-range dependencies, outperforms RNN/CNN baselines (e.g. GRU4Rec, Caser) and trains roughly an order of magnitude faster per epoch.
- **Limitation**: ignores **bidirectional context** (cannot use items after position $t$); the original few-negative BCE training can cause weak ranking on full-catalogue evaluation.
- **The loss matters more than the architecture** (Klenitskiy and Vasilev, 2023, "Turning Dross Into Gold"): vanilla SASRec with few negatives *underperforms* [[BERT4Rec]], but **SASRec+** — SASRec trained with a **full cross-entropy loss** or BCE with many (~3000) negatives — **beats BERT4Rec** on HR@K and NDCG@K on ML-1M. Too few negatives causes overconfidence; BPR/BCE/CE are model-agnostic choices.
- **Mechanism (pseudo-code):**

```pseudo
Algorithm: SASRec forward pass + scoring
─────────────────────────────────────────────
Input: item sequence s = (s_1, ..., s_n)   # left-padded to length n
Params: item table M ∈ R^{|I|×d}, position table P, b self-attention blocks

# 1. Embedding layer
for t = 1..n:
    E_t ← M[s_t] + P[t]                     # item + positional embedding
E ← dropout(E)

# 2. Stacked causal self-attention blocks
for layer = 1..b:
    Q,K,V ← project(E)
    A ← softmax( (Q Kᵀ)/sqrt(d) + causal_mask ) V   # mask future positions
    S ← LayerNorm(E + A)                    # residual
    E ← LayerNorm(S + FFN(S))               # point-wise FFN + residual
F ← E                                       # F_t encodes prefix s_1..s_t

# 3. Train (per position) or rank (last position)
#   training: BCE over true next item s_{t+1} (pos) + sampled negatives
#   inference: scores r_i = F_n · M[i]ᵀ  for all items i → rank top-K
```

## Connections

- Is a: [[Sequential Recommendation]] model / [[Next-Item Prediction|next-item predictor]] over a [[User-Item Interaction|user history]]
- Built from: [[Self-Attention]] + [[Transformer Model|Transformer]] blocks (causal mask), [[Embedding Layer|item + positional embeddings]]
- Trained with: [[Negative Sampling]] + BCE loss; can also use full [[NDCG|CE]] / [[BPR|Bayesian Personalized Ranking]] losses
- Improves on: [[FPMC]] (first-order only) and [[GRU4Rec]] (RNN, slower, weaker on long sequences)
- Contrast with: [[BERT4Rec]] (bidirectional, Cloze/masked-item training) — both stack Transformer blocks but BERT4Rec is bidirectional while SASRec is left-to-right
- Evaluated with: [[NDCG]], [[Hit Rate|HR@K]], [[Recall]] under [[Top-K Recommendation|top-K]] [[Offline Evaluation]]
- Precursor to: [[Generative Recommendation]] (TIGER, OneRec) — replaces score-and-rank over atomic ids with autoregressive decoding of [[Semantic IDs]]

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03a - Sequential Recommendation Models]]
- [[RS-L04 - Generative Recommendation]]
