---
type: concept
aliases: [Atomic IDs, Atomic Item ID]
course: [RecSys]
tags: [generative-rec, sequential-rec, exam-topic]
status: complete
---

# Atomic Item IDs

## Definition

> [!definition] Atomic Item ID
> An **atomic item ID** assigns **one unique, indivisible token per catalogue item**: each item $i \in \mathcal{I}$ gets its own integer index and its own learned embedding row. It is the simplest item-tokenization scheme (level **L1** of the tokenization ladder) and the implicit choice in classical sequential recommenders. In generative recommendation it is the **special case of a semantic ID with length $L=1$ and codebook = the entire catalogue**. The tokens are *arbitrary*: `item_3487` carries no information about what the item is, and similar items share no structure.

## Intuition

> [!intuition] One token per item — simple, but it does not scale or generalize
> Atomic IDs are the "natural" identifier: like giving every product a unique barcode. They are trivial to look up and a recommendation is exactly **one** autoregressive step. But two problems follow directly from "one token per item":
> 1. **The vocabulary equals the catalogue.** A platform with $10^6$ items needs $10^6$ output tokens (and a softmax over all of them); $10^9$ items is intractable. Capacity is tied 1-to-1 to vocabulary size.
> 2. **No shared structure.** Inception → `item_3487` and Interstellar → `item_124` are unrelated symbols even though the films are similar. There is no reusable "subword" the model can generalize over, so a brand-new item needs a brand-new token *and* a freshly trained embedding before it can ever be recommended — strict **cold start**.
>
> This is precisely the pain that **semantic IDs** (a few shared codebook tokens per item) were designed to remove.

## Mathematical Formulation

In classical sequential recommendation each item maps to a single token with a learned embedding, and the next item is produced by **scoring**, not generating. Given a SASRec-style history encoding $F_t^{(b)}$ and the shared item embedding table $M$ (one row $M_i$ per atomic ID), the score and the next-item objective are:

$$r_{i,t} = F_t^{(b)} M_i^{\top}, \qquad p(i_{t+1} \mid i_1,\dots,i_t) = \mathrm{softmax}_{i \in \mathcal{I}}\!\left(F_t^{(b)} M_i^{\top}\right)$$

where:
- $i_j$ — the $j$-th interacted item, encoded as a single atomic token $\langle i_j \rangle$
- $M \in \mathbb{R}^{|\mathcal{I}| \times d}$ — embedding table; **grows linearly with the catalogue size $|\mathcal{I}|$**
- $F_t^{(b)} M_i^{\top}$ — inner-product score of candidate $i$ against the encoded history
- softmax denominator runs over **all** $|\mathcal{I}|$ items — the output space *is* the catalogue

In the **generative** view, an item identifier is a length-$L$ token sequence decoded autoregressively. Atomic IDs are the degenerate case:

$$p_\theta(\mathbf{z}_i \mid \mathbf{x}) = \prod_{\ell=1}^{L} p_\theta(z_{i,\ell} \mid \mathbf{x}, z_{i,<\ell}) \;\;\xrightarrow{\;L=1\;}\;\; p_\theta(z_{i,1} \mid \mathbf{x})$$

where:
- $\mathbf{x} = (x_1,\dots,x_t)$ — user history; $\mathbf{z}_i$ — item identifier
- $L = 1$ — a single decoding step per item (vs. $L$ steps for a semantic ID)
- codebook at position 1 has $K = |\mathcal{I}|$ entries — so $K^L = |\mathcal{I}|$, capacity is **not** decoupled from vocabulary (contrast: semantic IDs reach $K^L \gg K\cdot L$ tokens, e.g. $256^4 \approx 4.3\times10^9$ items from only $4\times256$ tokens)

## Key Properties / Variants

- **Strengths:** simplest possible scheme; no tokenizer stage; direct, collision-free item↔token lookup; one autoregressive step per recommendation; works fine for **finite, stable catalogues**.
- **Weaknesses:** vocabulary explodes with catalogue size; tokens are semantically arbitrary; similar items share nothing; **no cold-start generalization**; popularity bias enters through the large output softmax; embedding table memory grows linearly.
- **Cold start (atomic vs semantic):** with atomic IDs a new item $i^\star$ has *no token in the vocabulary* — an embedding row must be added and learned from interactions, and until then $i^\star$ is unrecommendable. With semantic IDs the new item is run through a frozen tokenizer, its sub-tokens already exist, and similar items share prefixes, giving warm-start generalization "for free".
- **Where atomic IDs still appear in GenRec:** GPTRec and P5 use atomic IDs; there is even a counter-current paper "Atomic IDs are enough" (arXiv:2508.10478) arguing they remain competitive in some regimes.
- **Other ladder rungs (plain text, not the focus here):** L2 text/description-based IDs (very long, no collaborative info); L3 codebook-based semantic IDs from RQ-VAE (compact + semantic); L4 codebook + collaborative signal (LETTER, TokenRec); L5 adaptive IDs.

```pseudo
Building training examples — Atomic IDs (next-item prediction)
──────────────────────────────────────────────────────────────
Input : chronological user sequence (i_1, i_2, ..., i_t, i_{t+1})
Map   : each item -> ONE atomic token,  i_j -> <i_j>
        history = [<i_1>, <i_2>, ..., <i_t>]   (t tokens)
        target  = <i_{t+1}>                    (1 token)
Learn : p(i_{t+1} | i_1, ..., i_t)             (single AR step)
Vocab : exactly |I|  (one token per catalogue item)

Cold start (new item i*):
  token <i*> does NOT exist in vocab
  -> add embedding row, train from interactions  (strict cold start)
```

## Connections

- Level L1 / special case ($L=1$) of: [[Item Tokenization]], [[Semantic IDs]]
- The scheme [[Semantic IDs]] and [[RQ-VAE]] were designed to replace
- Used by score-and-rank classical models: [[SASRec]], [[Next-Item Prediction]] over a [[Sequential Recommendation]] history
- Contrasts with: [[Generative Retrieval]], [[Trie-Constrained Decoding]] (semantic-ID decoding must stay valid; atomic IDs are trivially valid)
- Drives the [[Cold Start Problem]] in generative recommenders
- Feeds the [[Top-K Recommendation]] softmax in discriminative [[Collaborative Filtering]] / [[Matrix Factorization]]

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
