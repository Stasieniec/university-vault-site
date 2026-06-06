---
type: concept
aliases: [Item ID Tokenization]
course: [RecSys]
tags: [generative-rec, llm, exam-topic]
status: complete
---

# Item Tokenization

> [!info] Lecture context
> Turning items into discrete identifiers/tokens an LLM (or any autoregressive decoder) can read and generate.

## Definition

> [!definition] Item Tokenization
> **Item tokenization** is the design choice of how each catalogue item $i \in \mathcal{I}$ is mapped to a **fixed-length sequence of discrete tokens** $\mathbf{z}_i = (z_{i,1}, \ldots, z_{i,L})$, so that a generative model can produce items the same way a language model produces text. It is the recommendation analogue of subword tokenization: language models generate **text tokens**, [[Generative Recommendation|generative recommenders]] generate **item tokens**.
> 
> It is the third of the three alignment paradigms for adapting an LLM to recommendation (after text prompting and injecting collaborative signal), and it answers the central question of [[Generative Recommendation]]: *how do items become tokens an LLM can generate?* The tokenizer defines the **output space** the model must learn to decode, so it is a **modelling choice, not mere preprocessing**.

## Intuition

> [!intuition] Why not just keep the item ID?
> Classical sequential recommenders ([[SASRec]], [[BERT4Rec]], [[GRU4Rec]]) treat each item as one **atomic ID** with its own learned embedding and *score* it: $r_{i,t} = F_t^{(b)} M_i^\top$. That breaks down for generation:
> - **Scale:** the output space equals the catalogue, so a softmax over $10^6$–$10^9$ items.
> - **Arbitrary:** `item_3487` carries no information; two similar films get two unrelated tokens.
> - **Cold start:** every new item needs a brand-new token *and* a freshly trained embedding before it is recommendable.
>
> The opposite extreme — use the item's **full text/description** as the ID — is meaningful but produces very long sequences that are expensive to decode and hard to constrain to real items.
>
> **Semantic IDs** are the middle ground: a *short* tuple of tokens drawn from *small shared codebooks*, where related items share a prefix. This separates **capacity** ($K^L$ possible codes) from **vocabulary size** ($K \cdot L$ tokens), giving compact, structured, generable identifiers.

## Mathematical Formulation

A semantic ID uses $L$ token positions, each chosen from a codebook of size $K$. The representational capacity is exponential in length while the vocabulary stays tiny:

$$\#\text{codes} = K^L, \qquad \#\text{tokens} = K \cdot L \qquad \text{(e.g. } 256^4 \approx 4.3\times10^9 \text{ from } 4\times256 = 1024 \text{ tokens)}$$

where:
- $L$ — identifier length (number of codebook levels), e.g. $3$–$4$
- $K$ — codebook size per position, e.g. $256$–$4096$
- $\mathbf{z}_i = (z_{i,1}, \ldots, z_{i,L})$ — the semantic ID of item $i$; $z_{i,\ell} \in \{1,\ldots,K\}$

The canonical learned tokenizer (TIGER) is **RQ-VAE** — a residual-quantized VAE. An encoder maps the item content embedding $\mathbf{x}_i$ (e.g. a Sentence-T5 vector of title/brand/category) to a latent $\mathbf{z}_i^{(0)}$, then **residual quantization** runs over $L$ codebooks: at each level it picks the nearest codeword, subtracts it, and passes the **residual** to the next level:

$$c_{i,\ell} = \arg\min_{k}\; \big\lVert \mathbf{r}_i^{(\ell-1)} - \mathbf{e}_{\ell,k} \big\rVert_2^2, \qquad \mathbf{r}_i^{(\ell)} = \mathbf{r}_i^{(\ell-1)} - \mathbf{e}_{\ell,\,c_{i,\ell}}, \qquad \mathbf{r}_i^{(0)} = \mathbf{z}_i^{(0)}$$

The selected codewords sum to the quantized latent $\hat{\mathbf{z}}_i = \sum_{\ell=1}^{L} \mathbf{e}_{\ell,\,c_{i,\ell}}$, which a decoder reconstructs back to $\hat{\mathbf{x}}_i$. The **semantic ID is the tuple of chosen indices** $\text{id}(i) = (c_{i,1}, \ldots, c_{i,L})$. The tokenizer is trained with reconstruction plus a quantization (commitment + codebook) term:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{rqvae}}, \qquad \mathcal{L}_{\text{recon}} = \lVert \mathbf{x}_i - \hat{\mathbf{x}}_i \rVert_2^2$$

where:
- $\mathbf{e}_{\ell,k}$ — codeword $k$ in the codebook at level $\ell$
- $\mathbf{r}_i^{(\ell)}$ — residual after subtracting the level-$\ell$ codeword (drives the coarse→fine hierarchy)
- $\mathcal{L}_{\text{recon}}$ — squared reconstruction error of the item embedding
- $\mathcal{L}_{\text{rqvae}}$ — quantization loss pulling residuals toward their nearest codewords

After tokenization the IDs are usually **frozen**: the downstream generator predicts the **indices**, not the continuous vectors, autoregressively:

$$p_\theta(\mathbf{z}_i \mid \mathbf{x}) = \prod_{\ell=1}^{L} p_\theta\big(z_{i,\ell} \mid \mathbf{x},\, z_{i,<\ell}\big), \qquad s_\theta(\mathbf{x}, i) = \log p_\theta(\mathbf{z}_i \mid \mathbf{x})$$

where $\mathbf{x}$ is the user history and the identifier likelihood doubles as the item's recommendation score.

## Key Properties / Variants

- **The L1–L5 ladder of item identifiers** (RS-L03b):
  - **L1 — Atomic ID** (P5, CLLM4Rec): one special token per item. Simple lookup, but vocabulary blows up and tokens carry no semantics.
  - **L2 — Text-based** (BIGRec, M6): use the item title/description. Meaningful but very long sequences, no collaborative info.
  - **L3 — Codebook-based** (TIGER, LC-Rec): discrete semantic IDs from RQ-VAE. **Compact + semantic** — the canonical sweet spot.
  - **L4 — Codebook + CF** (LETTER, TokenRec, CCFRec): inject collaborative signal into the quantizer so one ID carries both language and behaviour.
  - **L5 — Adaptive** (SIIT): the LLM refines identifiers during training; tokens evolve with the model.
- **Hierarchical prefixes = coarse-to-fine semantics.** Earlier indices are broad (e.g. a category like "Sports"), later ones refine the residual; items sharing $(12, 48, *)$ are coarsely similar and diverge later. This prefix structure is what enables cold-start generalization and controllable/diverse decoding.
- **Collision handling.** Distinct items can map to the same tuple, so an extra disambiguating token is appended: $(12,24,52)\!\to\!(12,24,52,0),(12,24,52,1)$, guaranteeing each final ID maps to exactly one item.
- **Construction families** (RS-L04): Residual Quantization (RQ-VAE, RQ-KMeans, R-VQ — ordered coarse→fine); Product Quantization (split the embedding, quantize subspaces — VQ-Rec); Hierarchical Clustering (tree-path IDs — P5-CID, RecForest); LM/Textual IDs (language tokens — LMIndexer, IDGenRec). No family is universally best — it depends on the embedding space, catalogue, and downstream task.
- **What shapes a semantic ID.** Two axes: *what representation we quantize* (text, multimodal, categorical, or raw) and *what objective we learn it with*. The field is moving from static content-only IDs toward **behaviour-aware, context-aware, task-aware** IDs:
  - **CoST** adds a contrastive objective so quantized codes preserve neighbourhood structure, not just reconstruction.
  - **LETTER** adds three regularizers — semantic hierarchy, collaborative (CF) alignment, and diversity for balanced code usage.
  - **ActionPiece** makes tokens *context-dependent*: the same action receives different tokens depending on surrounding actions (a subword-style merge over feature sets).
- **Content vs collaborative signal.** Content IDs capture what items *are* (title, brand, image); collaborative signal captures how users *use* items together (co-consumption). "A semantic ID is only as good as the representation it quantizes."
- **Why it is harder than text tokenization:** no natural reusable subwords, millions of long-tail items, sparse supervision per item, and a hard **validity constraint** — every generated ID must map to a real catalogue item (handled downstream by [[Trie-Constrained Decoding]]).

RQ-VAE residual-quantization tokenizer (offline, then frozen):

```pseudo
Algorithm: RQ-VAE Item Tokenization (build item→SID lookup)
──────────────────────────────────────────────────────────
Train phase (over item content embeddings x_i):
  for each item i:
    z ← Encoder(x_i)                      # latent vector
    r ← z                                 # residual r^(0)
    for level ℓ = 1 .. L:
      c[ℓ] ← argmin_k ‖ r - e[ℓ][k] ‖²    # nearest codeword index
      r    ← r - e[ℓ][c[ℓ]]               # subtract → next residual
    ẑ ← Σ_ℓ e[ℓ][c[ℓ]]                    # quantized latent
    x̂ ← Decoder(ẑ)
    minimize  ‖x_i - x̂‖²  +  L_rqvae       # update encoder, decoder, codebooks
  resolve collisions: append a unique suffix token to duplicate tuples
  SID(i) ← (c[1], ..., c[L][, suffix])    # store frozen item→SID table

Inference (downstream generator):
  generator decodes SID tokens autoregressively, one codebook level at a time
  constrain each step to valid catalogue paths (trie); map SID back to item i
```

## Connections

- Is one of the three alignment paradigms in [[LLM-based Generative Recommendation]] (with text prompting and injecting [[Collaborative Filtering|collaborative signal]])
- Core enabling step for [[Generative Recommendation]] / [[Generative Retrieval]] (and parallels document-ID generation in generative IR via the [[Differentiable Search Index]])
- Builds on [[Residual-Quantized VAE]] / [[Product Quantization]] to produce [[Semantic IDs]]
- Alternative to [[Atomic Item IDs]] (the $L=1$, codebook = catalogue special case)
- Feeds [[Autoregressive Decoding]] with [[Beam Search]] and [[Trie-Constrained Decoding]] for [[Next-Item Prediction]]
- Trained downstream with [[Supervised Fine-Tuning (SFT)]] (next-token cross-entropy) and optionally [[Group Relative Policy Optimization|GRPO]] / [[Direct Preference Optimization (DPO)]]
- Contrasts with the score-and-rank skeleton of [[Sequential Recommendation]] models ([[SASRec]], [[BERT4Rec]], [[GRU4Rec]])
- Affects [[Cold Start]] handling, [[Diversity]], and [[Popularity Bias]] in the generated list

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
