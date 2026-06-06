---
type: concept
aliases: []
course: [RecSys]
tags: [generative-rec, sequential-rec, exam-topic]
status: complete
---

# TIGER

## Definition

> [!definition] TIGER (Transformer Index for GEnerative Recommenders)
> **TIGER** (Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023) is the canonical [[Generative Retrieval]] model for [[Next-Item Prediction|next-item recommendation]]. It works in **two stages**: (1) an **offline tokenizer** ([[RQ-VAE]]) maps each item's content embedding to a short tuple of discrete codeword indices — its [[Semantic ID]]; (2) a **seq2seq Transformer** ([[Generative Recommendation|encoder–decoder]]) reads the user's history of semantic IDs and **autoregressively generates** the semantic ID of the next item, token by token. The generated ID is then looked up in the catalogue. This replaces "score every candidate over a fixed pool" with "decode the target identifier."

## Intuition

> [!intuition] Generate the address, don't scan the warehouse
> Classical [[Sequential Recommendation]] (e.g. [[SASRec]]) encodes the history into a state $F_t$ and scores **every** catalogue item via $r_{i,t}=F_t M_i^\top$ — a softmax over millions of [[Atomic Item IDs|atomic IDs]] whose table grows linearly with the catalogue. TIGER instead gives each item a structured "address" of $L$ codes (e.g. $(7,1,4)$), where related items share a coarse prefix. The model then *spells out* the next item's address one code at a time, exactly like a [[LLM|language model]] predicting the next token. Because only $K \cdot L$ code tokens exist (e.g. $4\times256$), they combine into $K^L \approx 4.3\times10^9$ possible IDs — capacity is decoupled from vocabulary size, the embedding table stays tiny, and a brand-new item gets a decodable address from its content alone (warm cold-start), without ever having been clicked.

## Mathematical Formulation

TIGER has two distinct objectives, one per stage.

> [!formula] Stage 1 — RQ-VAE tokenizer (build the Semantic ID)
> Encode item content embedding $\mathbf{x}_i$ to a latent $\mathbf{z}_i$, then **residual-quantize** it over $L$ codebooks. Starting from $\mathbf{r}_0 = \mathbf{z}_i$, at each level $d=1,\dots,L$ pick the nearest codeword, record its index, and pass the residual on:
> $$c_{i,d} = \arg\min_{k} \lVert \mathbf{r}_{d-1} - \mathbf{e}_{d,k}\rVert_2^2, \qquad \mathbf{r}_d = \mathbf{r}_{d-1} - \mathbf{e}_{d,\,c_{i,d}}$$
> The **Semantic ID** is the index tuple $\text{id}(i)=(c_{i,1},\dots,c_{i,L})$; the quantized latent is $\hat{\mathbf{z}}_i=\sum_{d=1}^{L}\mathbf{e}_{d,\,c_{i,d}}$, decoded back to $\hat{\mathbf{x}}_i$. The tokenizer is trained with
> $$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{rqvae}}, \qquad \mathcal{L}_{\text{recon}} = \lVert \mathbf{x}_i - \hat{\mathbf{x}}_i \rVert_2^2$$
>
> where:
> - $\mathbf{x}_i$ — item content embedding (TIGER uses a Sentence-T5 vector over title/brand/category)
> - $\mathbf{e}_{d,k}$ — the $k$-th codeword vector in codebook $d$ ($K$ codewords per level)
> - $c_{i,d}$ — selected index at level $d$ (coarse $\to$ fine as $d$ grows)
> - $\mathcal{L}_{\text{recon}}$ — reconstruction loss; $\mathcal{L}_{\text{rqvae}}$ — codebook/commitment loss with the stop-gradient straight-through estimator (VQ-style)
> - $L$ — ID length (toy figure uses $L=3$, $K=8$); a trailing token is appended for **collision handling** so each tuple maps to one item

> [!formula] Stage 2 — Autoregressive generation (the recommender)
> Given user history $\mathbf{x}$ (a flat sequence of the history items' semantic-ID tokens), the next item's ID $\mathbf{z}_i=(z_{i,1},\dots,z_{i,L})$ is decoded one code at a time:
> $$p_\theta(\mathbf{z}_i \mid \mathbf{x}) = \prod_{\ell=1}^{L} p_\theta\!\left(z_{i,\ell}\mid \mathbf{x},\, z_{i,<\ell}\right)$$
> trained with **next-token cross-entropy** (teacher forcing):
> $$\mathcal{L} = -\sum_{\ell=1}^{L} \log p_\theta\!\left(z_{i,\ell}\mid \mathbf{x},\, z_{i,<\ell}\right)$$
> At inference the likelihood doubles as a ranking score, $s_\theta(\mathbf{x},i)=\log p_\theta(\mathbf{z}_i\mid\mathbf{x})$, and a ranked list is produced by **beam search** over the $L$ decoding steps.
>
> where:
> - $z_{i,\ell}$ — the $\ell$-th codebook token of item $i$ (so each item spans $L$ decoder positions, not one)
> - $\theta$ — parameters of the T5-style [[Transformer Model]] (bidirectional encoder + autoregressive decoder)
> - the codebook ($\sim 256$–$4096$ entries) replaces the BPE vocabulary; output is item identifiers, **not** natural language

## Key Properties / Variants

- **Two-stage, frozen tokenizer:** semantic-ID construction is an *offline* preprocessing step; once trained the RQ-VAE is frozen and the generator predicts **indices**, never the continuous embeddings. The decoder of the RQ-VAE is discarded for serving.
- **Hierarchical prefixes:** earlier codes are coarse (e.g. broad category), later codes refine the residual. Items sharing $(12,48,\ast)$ are coarsely similar; the full tuple disambiguates. This prefix tree is what enables cold-start generalization and controllable/diverse retrieval.
- **Validity is not free:** of the $K^L$ possible code sequences only a tiny fraction are real items, so naive decoding can emit non-existent IDs. TIGER-style systems use [[Trie-Constrained Decoding]]: store all valid catalogue IDs in a trie and apply a logit mask so only on-path tokens are allowed. (Complementary fix: reward validity during RL.)
- **Decoding pathologies:** popular prefixes dominate beam search (popularity amplification), top-$B$ items share prefixes (homogeneity), and each recommendation costs $L$ sequential steps + trie lookup (latency).
- **Empirical result:** on Amazon Sports/Beauty/Toys, RQ-VAE semantic IDs beat Random IDs and LSH-based IDs, and TIGER outperforms SASRec / S³-Rec / P5 baselines on Recall@K and [[NDCG]]@K — establishing item tokenization as a *modelling choice*, not mere preprocessing.
- **Architecture variants:** TIGER uses an encoder–decoder (T5-style, "read history fully, then write"); decoder-only successors (HSTU, OneRec, GPTRec) treat `[history || target SID]` as one stream and scale to longer histories.
- **Tokenizer variants / successors:** beyond reconstruction-only RQ-VAE — CoST adds a contrastive objective, LETTER adds semantic + collaborative + diversity regularizers, ActionPiece makes tokens context-aware, and LC-Rec tunes an LLM over the semantic IDs. RQ-KMeans, R-VQ, and product-quantization (VQ-Rec) are competing tokenizers; the best choice depends on embedding space and task.

```pseudo
Algorithm: TIGER (two-stage generative retrieval)
──────────────────────────────────────────────────
STAGE 1 — Offline tokenization (per catalogue item i)
  x_i  ← ContentEncoder(title, brand, category)      # Sentence-T5 embedding
  z_i  ← RQVAE_Encoder(x_i)
  r    ← z_i
  for d = 1 .. L:
     c[d] ← argmin_k || r - e[d,k] ||^2               # nearest codeword
     r    ← r - e[d, c[d]]                             # pass residual on
  SID(i) ← (c[1], ..., c[L])  (+ extra token if collision)
  train RQ-VAE by min  ||x_i - Decoder(sum_d e[d,c[d]])||^2 + L_rqvae
  freeze tokenizer; build trie of all valid SIDs

STAGE 2 — Train the seq2seq recommender
  for each user history (i_1,...,i_t -> i_{t+1}):
     input  ← flatten(SID(i_1) ... SID(i_t))           # t * L tokens
     target ← SID(i_{t+1})                             # L tokens
     minimize  - sum_l log p_theta(z_l | input, z_<l)   # teacher forcing

INFERENCE — recommend next items
  encode user history -> context
  beam search (size B) over L steps, masked by trie    # valid SIDs only
  emit B complete SIDs -> map back to items
  filter (drop already-seen, dedup, business rules) -> ranked list
```

## Connections

- Special case of: [[Generative Recommendation]] / [[Generative Retrieval]] (SID-based, generate the identifier)
- Tokenizer: [[RQ-VAE]] (residual quantization producing [[Semantic IDs]]); contrast with [[Atomic Item IDs]] and full-text IDs
- Decoder objective: [[Autoregressive Generation]] with next-token cross-entropy, [[Beam Search]], [[Trie-Constrained Decoding]]
- Replaces the score-and-rank skeleton of: [[SASRec]], [[Sequential Recommendation]]
- Recommendation analogue of generative IR: [[DSI]], [[GENRE]] (generate a document identifier)
- Item-tokenization ladder context: see [[Item Tokenization]] / [[Item ID Tokenization]]
- Scaling sibling route (native, not borrowed): [[HSTU]], [[Large Recommendation Models (LRM)]]

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
