---
type: concept
aliases: [Semantic ID, Hierarchical Semantic IDs]
course: [RecSys]
tags: [generative-rec, sequential-rec, llm, exam-topic]
status: complete
---

# Semantic IDs

## Definition

> [!definition] Semantic ID
> A **Semantic ID (SID)** is a **short, fixed-length sequence of discrete tokens** $(z_1, z_2, \ldots, z_L)$ that identifies one catalogue item, where each token is drawn from a small learned **codebook** rather than from a per-item vocabulary. Unlike an **atomic item ID** (one unique token per item), SID tokens are **shared across items**: related items share leading tokens, while the full tuple still resolves to a single item. SIDs are derived from **item content/embeddings** (and optionally collaborative signal), which makes items **generable** by an autoregressive model — they are the token space that turns recommendation into a [[Generative Recommendation|sequence-generation]] problem.

## Intuition

> [!intuition] The Middle Ground Between Atomic IDs and Full Text
> There are two extremes for naming an item. An **atomic ID** (`item_3487`) is compact and easy to look up, but the vocabulary grows linearly with the catalogue ($10^6$ items need $10^6$ tokens), the IDs are arbitrary (similar films get unrelated tokens), and every new item needs a freshly trained embedding. The other extreme — using the item's **full text description** as the ID — is meaningful but produces very long, hard-to-constrain sequences that may not map uniquely to one item.
>
> A Semantic ID sits in between: a **few reusable tokens** that are shorter than text and more structured than one atomic token. In *hierarchical* SIDs the **shared prefix** describes a coarse group and later tokens refine toward the specific item, e.g. movies under $(12, 48, *, *)$ are coarsely similar and only diverge at deeper positions. This is the same trick subword tokenization (BPE) plays for language — except items have no natural subwords, so we have to *learn* the codebook.

## Mathematical Formulation

The defining property of a SID is that a small codebook generates an enormous item space. With $L$ token positions and a codebook of size $K$ per position, the number of distinct codes is

$$\underbrace{K \times K \times \cdots \times K}_{L \text{ positions}} = K^{L}, \qquad \text{e.g. } 256^{4} \approx 4.3 \times 10^{9},$$

where:
- $L$ — identifier length (number of codebook levels / token positions), typically $3$–$4$
- $K$ — codebook size per position, typically $256$–$4096$
- only $L \times K$ learned code embeddings are stored, yet they index up to $K^L$ items — SIDs separate **capacity** from **vocabulary size**

The canonical construction (TIGER, Rajput et al. NeurIPS 2023) builds SIDs with a **Residual-Quantized VAE ([[RQ-VAE]])**. An item content embedding $\mathbf{x}_i$ (e.g. a Sentence-T5 vector) is encoded to a latent vector, then quantized over $L$ codebooks: at each level $d$ pick the **nearest codeword**, subtract it, and pass the **residual** to the next level. The chosen indices form the SID:

$$\text{id}(i) = (c_{i,1}, c_{i,2}, \ldots, c_{i,L}), \qquad \hat{\mathbf{z}}_i = \sum_{d=1}^{L} \mathbf{e}_{d,\, c_{i,d}},$$

where:
- $c_{i,d}$ — index of the codeword chosen at level $d$ for item $i$ (one SID token)
- $\mathbf{e}_{d,\,c_{i,d}}$ — the corresponding codeword vector in codebook $d$
- $\hat{\mathbf{z}}_i$ — quantized latent (sum of selected codewords), which a decoder reconstructs back to $\mathbf{x}_i$

The tokenizer is trained with a reconstruction + quantization objective:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{rqvae}}, \qquad \mathcal{L}_{\text{recon}} = \lVert \mathbf{x}_i - \hat{\mathbf{x}}_i \rVert_2^2,$$

where $\mathcal{L}_{\text{rqvae}}$ is the codebook/commitment loss that pulls residuals toward their nearest codewords. After training the **decoder is discarded for serving**: the downstream recommender predicts the **discrete indices**, not the continuous vector.

Once items are SIDs, a generative recommender decodes the next item's identifier **autoregressively**, one token at a time:

$$p_\theta(\mathbf{z}_i \mid \mathbf{x}) = \prod_{\ell=1}^{L} p_\theta\!\left(z_{i,\ell} \mid \mathbf{x},\, z_{i,<\ell}\right), \qquad s_\theta(\mathbf{x}, i) = \log p_\theta(\mathbf{z}_i \mid \mathbf{x}),$$

where $\mathbf{x}$ is the user history (itself a flat sequence of SID tokens), $z_{i,<\ell}$ are the already-generated tokens, and the identifier log-likelihood doubles as the item score for ranking.

## Key Properties / Variants

- **Hierarchical / coarse-to-fine:** earlier indices are coarser, later ones refine the residual. Shared prefixes form a **category hierarchy** (e.g. root "Sports" $(1722,*,*)$ → "Outdoor sports" $(1723, 541, *)$ → "Surfing" $(1723, 541, 1129)$), which is what enables generating a coarse prefix then refining toward a specific item.
- **Collision handling:** distinct items can quantize to the same tuple, so an extra disambiguating token is appended, e.g. $(12,24,52) \to (12,24,52,0), (12,24,52,1)$, guaranteeing each final SID maps to exactly one catalogue item.
- **Validity is not automatic:** of the $K^L \approx 10^9$ possible codes only a tiny fraction ($\sim 10^7$) are real items. Generation must be **constrained** so the model emits only existing IDs (see pseudo-code below).
- **One item = $L$ positions:** with SIDs, predicting the next item takes $L$ autoregressive steps (vs $1$ for atomic IDs). Atomic IDs are the special case $L=1$ with codebook = full catalogue.
- **Construction families:** **Residual Quantization** (RQ-VAE, RQ-KMeans, R-VQ — ordered coarse→fine); **Product Quantization** (split the embedding, quantize subspaces — VQ-Rec); **Hierarchical Clustering** (tree-path IDs — P5-CID, RecForest); **LM/textual IDs** (LMIndexer, IDGenRec).
- **Beyond reconstruction (behaviour-aware SIDs):** reconstruction-only codes capture *what items are*, not *how users use them*. CoST adds a **contrastive** objective to preserve neighbourhood structure; LETTER adds **semantic + collaborative + diversity** regularizers ($\mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{sem}} + \mathcal{L}_{\text{CF}} + \mathcal{L}_{\text{div}}$); ActionPiece makes tokenization **context-aware** (same action → different tokens depending on surrounding sequence). "A Semantic ID is only as good as the representation it quantizes."
- **Cold start:** a new item is run through the *frozen* tokenizer to get a SID whose sub-tokens already exist in the codebook, so it becomes **decodable without retraining** and inherits prefixes from similar items — but being decodable is not the same as being recommended (see warning).

```pseudo
Inference: Trie-Constrained Beam Search over SIDs
──────────────────────────────────────────────────
Precompute: store all valid catalogue SIDs in a trie T
Input: user history h (flat sequence of SID tokens), beam size B
Initialize beams ← { empty prefix }
For step ℓ = 1 .. L:
  candidates ← {}
  For each partial SID p in beams:
    allowed ← children of prefix p in trie T   # only valid next tokens
    For each token z in allowed:
      score(p · z) ← score(p) + log p_θ(z | h, p)   # renormalize over allowed
      add (p · z) to candidates
  beams ← top-B candidates by score
Return the B complete SIDs  →  map each back to its catalogue item
Post-process: drop items already in history, dedup, apply business rules
```

> [!warning] Cold-Start Is Fragile and Decoding Is Biased
> A fresh item gets a valid SID the moment it is tokenized, but the generator was **trained only on SIDs of items people actually clicked**, so a new item's SID has almost no probability mass and **beam search prunes it before it is considered**. Production systems therefore stay **hybrid** (e.g. LIGER): generate warm candidates, inject cold items by hand, then re-rank with **dense embeddings**. Decoding also has pathologies: **popularity amplification** (popular prefixes dominate the beam), **homogeneity** (top-$B$ items share a long prefix → near-duplicate lists), and **latency** ($L$ sequential steps per item plus trie upkeep as the catalogue changes).

## Connections

- Enables: [[Generative Recommendation]] / [[Generative Retrieval]] (SIDs are the generable token space)
- Built with: [[RQ-VAE]] / [[Residual Quantization]], also [[Product Quantization]], [[Codebook]]
- Contrast with: [[Atomic Item IDs]] (one token per item, vocabulary grows with catalogue)
- Generated by: [[TIGER]] (encoder–decoder), [[OneRec]], [[HSTU]] (decoder-only)
- Grounded via: [[Trie-Constrained Decoding]] + [[Beam Search]] for valid-item generation
- Trained with: [[Next-Item Prediction]] cross-entropy, optionally [[GRPO]] / [[DPO]] reward fine-tuning
- Parallel idea in IR: [[Differentiable Search Index]] / [[DSI]] (generate a document identifier)
- Item-tokenization paradigm within: [[Item Tokenization]], one of the LLM-alignment routes alongside [[LLM-based Recommendation]]
- Diversity remedies: [[Maximal Marginal Relevance (MMR)]] re-ranking, diverse beam search

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L03a - Sequential Recommendation Models]]
- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
