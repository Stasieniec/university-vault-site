---
type: concept
aliases: [SeqRec, Sequential Recommender Systems]
course: [RecSys]
tags: [collaborative-filtering, sequential-rec, exam-topic]
status: complete
---

# Sequential Recommendation

> Models the order of a user's interactions to predict the next item (covered in Lecture 3).

## Definition

> [!definition] Sequential Recommendation
> **Sequential recommendation** is the task of predicting the **next item** $i_{n+1}$ in a **chronologically ordered** sequence of (historical) user-item interactions $\langle i_1, i_2, \dots, i_n \rangle$. Unlike plain [[Matrix Factorization|MF]] / [[Collaborative Filtering|CF]], which treats a user's interactions as an **unordered set**, sequential models exploit temporal order, recency, repetition, and item-to-item transitions. The recommended target can be a single item, or a basket / bundle / playlist; the setting splits into **session-based** (anonymous within-session) and **user-based** (persistent profile) paradigms.

## Intuition

> [!intuition] Why order matters
> Standard CF answers "which items does this user like?" but ignores *when* and *in what sequence*. Many real signals are inherently ordered:
> - **Natural sequence patterns:** buy a phone case *after* the phone, not before.
> - **Series:** Star Wars IV → V → VI; Harry Potter book 1 → 2 → 3 → (predict book 4).
> - **Evolving interests:** a user who listened to pop 10 years ago now prefers rock.
> - **Repetition:** a user re-buys coffee every month.
>
> [[Matrix Factorization]] collapses all of these into a single static preference and loses them. Sequential recommendation keeps the timeline, so the **most recent context** drives the prediction.

## Mathematical Formulation

The general objective is to model the conditional probability of the next item given the history:

$$P(i_{n+1} \mid i_1, i_2, \dots, i_n)$$

A **k-order [[Markov Chain]]** truncates the history to the last $k$ interactions:

$$P(i_{n+1} \mid i_{n-k}, \dots, i_n)$$

where:
- $i_t$ — item interacted with at position $t$ in the chronological sequence
- $n$ — current sequence length (history seen so far)
- $k$ — Markov order (how many previous items condition the prediction; $k=1$ is a first-order chain)

[[Factorized Personalized Markov Chains (FPMC)|FPMC]] (Rendle et al., 2010) combines a long-term preference term with a factorized first-order transition term, scoring a transition from previous item $i$ to candidate next item $j$ for user $u$:

$$\hat{x}_{u,i,j} = P_u^\top Q_j + R_i^\top S_j$$

where:
- $P_u^\top Q_j$ — **long-term preference**: user $u$'s general affinity for item $j$ (standard MF dot product of latent factors)
- $R_i^\top S_j$ — **short-term transition**: factorized likelihood of moving from item $i$ to item $j$
- $P_u$ — user latent vector; $Q_j$ — item-as-target factor (preference term)
- $R_i$ — previous-item factor; $S_j$ — next-item factor (transition term)

Factorization is needed because per-user transition matrices $T^{(u)} \in \mathbb{R}^{|I| \times |I|}$ are far too sparse to estimate directly. FPMC is trained with a pairwise ranking loss (S-[[BPR]]) via [[SGD]].

## Key Properties / Variants

The lecture traces a progression from counting-based Markov models to deep sequence models. All deep models share the recipe **item embeddings → sequence encoder → next-item scores**, and the loss is largely interchangeable across them.

- **Naive [[Markov Chain]] (Shani et al., 2005):** a single **global** transition matrix $T \in \mathbb{R}^{|I| \times |I|}$ estimated by counting consecutive co-occurrences (n-gram model). No personalization; severe **data sparsity**, mitigated by *skipping* (allow gaps), *clustering* (group similar items), and *mixture modeling* (interpolate Markov orders).
- **[[Factorized Personalized Markov Chains (FPMC)|FPMC]]:** personalized MF (long-term) + factorized first-order transition (short-term). Lightweight and interpretable; captures only **first-order** dependencies. Wins on sparse data / short sequences.
- **[[GRU4Rec]] (Hidasi et al., 2015):** first deep model; a [[Gated Recurrent Unit (GRU)|GRU]] [[Recurrent Neural Network (RNN)|RNN]] over the session. One-hot item → embedding → stacked GRU layers → feedforward → per-item scores. Trained with pairwise [[BPR]] loss. Handles longer sequences than FPMC but is slow and struggles with very long histories.
- **[[SASRec]] (Self-Attentive Sequential Recommendation; Kang & McAuley, 2018):** first recommender built purely on [[Self-Attention]]. Input = item embedding + positional embedding; a **causal mask** prevents attending to future items (left-to-right). Faster and stronger than RNN/CNN; trained with binary cross-entropy + negative sampling. At inference, scores all items by multiplying the **last token's** representation with the item-embedding matrix.
- **[[BERT4Rec]] (Sun et al., 2019):** **bidirectional** [[Self-Attention]] trained with a Cloze (masked-item) task — randomly mask items and predict them using both left and right context. At inference, a `[MASK]` token is appended to the end of the history; adding this "mask-at-the-end" as a second training stage closes the train/inference gap and boosts performance.

The three losses discussed are model-agnostic (any model can use any of them):

> [!formula] Pairwise BPR loss (GRU4Rec)
> $$\mathcal{L}_{\text{BPR}} = -\frac{1}{N_S} \sum_{j=1}^{N_S} \log \sigma\!\left(\hat{r}_{s,i} - \hat{r}_{s,j}\right)$$
> where: $N_S$ = negatives per positive; $\hat{r}_{s,i}$ = score of true next item $i$; $\hat{r}_{s,j}$ = score of negative $j$; $\sigma$ = sigmoid. Pushes the true item above sampled negatives.

> [!formula] BCE loss (SASRec)
> $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N_S} \sum_{i=1}^{N_S} \left[ y_{s,i} \log \hat{y}_{s,i} + (1 - y_{s,i}) \log(1 - \hat{y}_{s,i}) \right]$$
> where: $y_{s,i} \in \{0,1\}$ = ground-truth label; $\hat{y}_{s,i}$ = predicted score; $N_S$ = positives + negatives per sequence.

> [!formula] Masked LM / CE loss (BERT4Rec)
> $$\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log P(v_i \mid S_{\text{masked}}; \theta)$$
> where: $\mathcal{M}$ = masked positions; $v_i$ = true item at $i$; $P(\cdot)$ = full-vocabulary softmax over items; $\theta$ = parameters.

```pseudo
Algorithm: Generic deep sequential recommender (training)
──────────────────────────────────────────────────────────
Given: ordered interaction sequences per user/session
Initialize: item embedding table E, positional embeddings, encoder θ

Loop over training sequences s = [v_1, ..., v_n]:
  x ← Embed(s)                       # item emb (+ positional emb)
  h ← Encoder(x)                     # GRU (RNN) | causal self-attn (SASRec)
                                     #   | bidirectional self-attn w/ masking (BERT4Rec)
  for each prediction position t:
    sample negatives; compute scores ⟨h_t, E⟩
    update θ, E by minimizing L  (BPR | BCE | CE/MLM)

Inference:
  encode the user history; score the LAST position against all items;
  return Top-K items
```

> [!warning] Loss matters more than architecture
> Klenitskiy & Vasilev (2023), "Turning Dross into Gold," show BERT4Rec's reported edge over SASRec is mostly due to its **loss and number of negatives**, not bidirectionality. A "SASRec+" trained with full cross-entropy, or BCE with ~3000 negatives, **beats** BERT4Rec on HR@K and NDCG@K. Too few negatives causes **overconfidence**. Alternatives beyond BPR/BCE/CE include TOP1-max, listwise LambdaRank, and contrastive InfoNCE.

> [!warning] Open challenges
> Scaling to catalogs of **millions of items** (motivates [[Approximate Nearest Neighbor|ANN]] search and sub-item [[Semantic IDs]]); **cold-start** for new users/items (motivates [[Generative Recommendation]] / [[LLM]]-based models); and learning from **very long histories**. These directly bridge into Lecture 3b (LRMs) and Lecture 4.

## Connections

- Contrasts with: [[Matrix Factorization]], [[Collaborative Filtering]] (order-agnostic, treat interactions as an unordered set)
- Task framing: [[Next-Item Prediction]], [[Top-K Recommendation]], [[Session-based Recommendation]]
- Markov models: [[Markov Chain]], [[Factorized Personalized Markov Chains (FPMC)|FPMC]], [[Transition Matrix]]
- Deep models: [[GRU4Rec]] ([[Recurrent Neural Network (RNN)|RNN]]/[[Gated Recurrent Unit (GRU)|GRU]]), [[SASRec]] ([[Self-Attention]]), [[BERT4Rec]] (bidirectional [[Transformers]])
- Losses: [[BPR]], [[Negative Sampling]]
- Evaluated with: [[NDCG]], [[Hit Rate|HR@K]]
- Scales into: [[Generative Recommendation]], [[Large Recommendation Models (LRM)|LRM]], [[Semantic IDs]], [[Approximate Nearest Neighbor|ANN]]

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03a - Sequential Recommendation Models]]
- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
