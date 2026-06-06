---
type: concept
aliases: [Next Item Prediction]
course: [RecSys]
tags: [sequential-rec, generative-rec, collaborative-filtering, exam-topic]
status: complete
---

# Next-Item Prediction

## Definition

> [!definition] Next-Item Prediction
> **Next-item prediction** is the core objective of [[Sequential Recommendation]]: given a chronologically ordered history of a user's past interactions $\mathcal{H}_t = (i_1, i_2, \ldots, i_t)$, predict the next item $i_{t+1}$ the user will interact with (click, watch, purchase, listen to).
>
> It is the recommendation analogue of **next-token prediction** in language modelling: use the past sequence to predict the next output. Two solution families share this objective:
> - **Discriminative (score-and-rank):** learn a score $s(\mathcal{H}_t, i)$ for each candidate item and rank — e.g. [[SASRec]], [[BERT4Rec]], [[GRU4Rec]].
> - **Generative:** decode an item *identifier* token-by-token and look it up in the catalogue — e.g. P5, TIGER, OneRec.

## Intuition

> [!intuition] User behaviour is already a sequence
> A user's interaction log $i_1 \to i_2 \to \cdots \to i_t \to ?$ is a temporal sequence, exactly like a sentence is a sequence of tokens. So next-item prediction inherits the entire sequence-modelling toolkit:
> - the **classical** view encodes the history into a state and dots it against every candidate item embedding (a softmax over the whole catalogue);
> - the **generative** view instead *produces* the answer directly, sidestepping the per-candidate scan.
>
> The objective is the same; what differs is the **output space**. Classical models output a distribution over catalogue items; generative models output a sequence of identifier tokens that must map back to a real item.

## Mathematical Formulation

The classical (discriminative) formulation scores each catalogue item against the encoded history. For [[SASRec]], the encoder produces a state $F_t^{(b)}$ and the score for item $i$ is the inner product with its embedding row $M_i$:

$$r_{i,t} = F_t^{(b)} M_i^{\top}, \qquad p(i_{t+1} = i \mid \mathcal{H}_t) = \mathrm{softmax}_i\!\left(F_t^{(b)} M_i^{\top}\right)$$

where:
- $\mathcal{H}_t = (i_1, \ldots, i_t)$ — chronological user interaction history
- $F_t^{(b)}$ — encoded history state from a causal self-attention encoder
- $M_i$ — learned embedding (row of the shared item table $M$) for candidate item $i$; vocabulary $= |\mathcal{I}|$ (one [[Atomic Item IDs|atomic id]] per item)
- $r_{i,t}$ — score; higher means $i$ is a more likely next interaction

The **generative** formulation replaces direct scoring with autoregressive decoding of an item identifier $\mathbf{z}_i = (z_{i,1}, \ldots, z_{i,L})$ (e.g. a [[Semantic IDs|semantic id]]):

$$p_\theta(\mathbf{z}_i \mid \mathbf{x}) = \prod_{\ell=1}^{L} p_\theta\!\left(z_{i,\ell} \mid \mathbf{x},\, z_{i,<\ell}\right), \qquad s_\theta(\mathbf{x}, i) = \log p_\theta(\mathbf{z}_i \mid \mathbf{x})$$

where:
- $\mathbf{x} = (x_1, \ldots, x_t)$ — user history (each $x_j \in \mathcal{I}$, expanded to its id tokens)
- $\mathbf{z}_i = (z_{i,1}, \ldots, z_{i,L})$ — fixed-length identifier of item $i$ ($L$ tokens; atomic ids are the special case $L=1$)
- $z_{i,<\ell}$ — identifier tokens already generated (left context)
- $s_\theta(\mathbf{x}, i)$ — the identifier's log-likelihood, reused as the item score for ranking

Both families are trained the same way: **next-token cross-entropy** with teacher forcing over the target id (for atomic ids this is a single position, $L=1$):

$$\mathcal{L} = -\sum_{\ell=1}^{L} \log p_\theta\!\left(z_\ell \mid \text{history},\, z_{<\ell}\right)$$

where the loss is averaged over all $L$ positions and all items in the batch. The tokens are item codes from a small learned codebook ($K \sim 256$–$4096$), not a natural-language vocabulary.

## Key Properties / Variants

- **Discriminative skeleton (encode → score → rank).** [[SASRec]] (causal self-attention), [[BERT4Rec]] (masked, bidirectional), [[GRU4Rec]] ([[Gated Recurrent Unit (GRU)|GRU]]) differ only in *how* they encode $\mathcal{H}_t$; all keep the score-and-rank step. Output space = catalogue size, so the final layer is a softmax over millions of items.
- **Why move beyond atomic ids.** An [[Atomic Item IDs|atomic id]] like `item_3487` carries no semantics, the embedding table grows linearly with the catalogue, and every new item needs a fresh id *and* a trained embedding (**strict [[Cold Start]]**).
- **Generative variant — decode an identifier.** Reframes next-item prediction as [[Autoregressive Generation|autoregressive]] id generation. Encoder–decoder (T5-style: TIGER, LETTER) reads the history fully then writes the id; decoder-only (GPT-style: [[HSTU]], OneRec) treats `[history || target]` as one stream. Scales to long histories and unifies retrieval + ranking.
- **Item tokenization is a modelling choice.** [[Semantic IDs]] built by [[RQ-VAE|residual-quantized VAE]] give $K^L$ ids from only $K \cdot L$ tokens (e.g. $256^4 \approx 4.3\times10^9$), share prefixes for related items (cold-start generalisation), and decouple capacity from vocabulary size.
- **Validity constraint (generative only).** Most of the $K^L$ codes are not real items, so decoding is **trie-constrained**: at each step a logit mask permits only tokens lying on a valid catalogue path. Alternatively, validity is rewarded during RL fine-tuning ([[GRPO]]).
- **Inference = beam search.** Greedy decoding yields one item; **beam search** keeps $B$ partial ids to produce a *ranked list*. Pathologies: popularity-prefix amplification, homogeneous (look-alike) lists, local optima from greedy first tokens, and $L$-step latency.
- **Optional reward fine-tuning.** Cross-entropy only rewards copying the exact next click; RL ([[GRPO]] / [[Direct Preference Optimization (DPO)|DPO]]) lets the model generate a group of candidates, score them for validity / relevance / diversity, and push above-average ones up.

```pseudo
Algorithm: Generative Next-Item Prediction (SID-based, inference)
─────────────────────────────────────────────────────────────────
Given: trained model p_θ, frozen tokenizer, trie of valid catalogue SIDs
Input: user history H = (i_1, ..., i_t)

1. Look up each item's SID:  x ← flatten[ SID(i_1), ..., SID(i_t) ]
2. Initialize beam B with the empty prefix
3. For ℓ = 1 .. L:                      # L codebook positions per item
     For each partial SID b in beam:
       allowed ← trie.children(b)        # validity constraint
       score next tokens with p_θ(z_ℓ | x, b), masking ∉ allowed
     beam ← top-B partial SIDs by cumulative log-prob
4. Map each completed SID → catalogue item (id-to-item lookup)
5. Filter (drop items already in H), dedup, apply business rules
6. Return ranked list
```

## Connections

- Core task of: [[Sequential Recommendation]] · [[Session-based Recommendation]]
- Classical encoders: [[SASRec]] · [[BERT4Rec]] · [[GRU4Rec]]
- Generative route: [[Generative Recommendation]] · [[Generative Retrieval]] · [[Autoregressive Generation]]
- Identifier design: [[Atomic Item IDs]] · [[Semantic IDs]] · [[Item Tokenization]] · [[RQ-VAE]]
- Decoding: [[Beam Search]] · [[Trie-Constrained Decoding]]
- Objectives & tuning: [[Supervised Fine-Tuning (SFT)]] · [[GRPO]] · [[Direct Preference Optimization (DPO)]]
- Scaling-native architectures: [[HSTU]] · [[Large Recommendation Models (LRM)]]
- Failure mode for new items: [[Cold Start]]
- Evaluation: [[Top-K Recommendation]] · [[NDCG]] · [[Recall]]

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
