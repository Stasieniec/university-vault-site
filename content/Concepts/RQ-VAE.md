---
type: concept
aliases: [Residual-Quantized VAE, Residual Quantization, Codebook]
course: [RecSys]
tags: [generative-rec, sequential-rec, exam-topic]
status: complete
---

# RQ-VAE

Residual-Quantized Variational Autoencoder used to produce discrete codes ("Semantic IDs") for items.

## Definition

> [!definition] RQ-VAE
> **RQ-VAE (Residual-Quantized VAE)** is an autoencoder that maps a continuous item embedding to a **short tuple of discrete codeword indices** — a [[Semantic ID]] — by quantizing it **residually** across $L$ ordered codebooks. A DNN encoder produces a latent vector; at each level the nearest codeword is selected, **subtracted off**, and the leftover residual is passed to the next codebook. The selected indices $(c_1, \ldots, c_L)$ are the item's identifier; their codeword vectors **sum** to the quantized latent, which a DNN decoder reconstructs back to the original embedding.
>
> In [[Generative Recommendation]] (TIGER, Rajput et al., NeurIPS 2023) RQ-VAE is the **offline tokenizer**: it is trained once, then frozen, and the [[Sequential Recommendation]] model autoregressively generates the *indices*, never the continuous vectors.

## Intuition

> [!intuition] One codebook is too coarse; stack the leftovers
> A single codebook (plain vector quantization, $K$ codes) can only place an item in one of $K$ buckets — too coarse to identify millions of items. RQ-VAE instead **refines a guess**: codebook 1 gives a rough position, then you look at *what's left over* (the residual) and refine it with codebook 2, then refine *that* leftover with codebook 3, and so on.
>
> This makes the code **hierarchical**: the first index $c_1$ is a coarse category, later indices split it finer (slide 26's tree: `Sports → Outdoor → Surfing`). Related items end up sharing a **prefix** $(12, 48, *)$ — coarse semantic similarity — while the full tuple pins down one item. It also **separates capacity from vocabulary size**: $L=4$ levels of $K=256$ codes each address $256^4 \approx 4.3 \times 10^9$ items using only $4 \times 256 = 1024$ learned vectors, instead of one token per item.

## Mathematical Formulation

> [!formula] Residual quantization (encode an item to a Semantic ID)
> Encode item embedding $\mathbf{x}_i$ to a latent $\mathbf{z}_i = \text{Enc}(\mathbf{x}_i)$. Initialize the residual $\mathbf{r}_0 = \mathbf{z}_i$. For each level $d = 1, \ldots, L$ with codebook $\mathcal{C}_d = \{\mathbf{e}_{d,k}\}_{k=1}^{K}$:
> $$c_{i,d} = \arg\min_{k} \; \lVert \mathbf{r}_{d-1} - \mathbf{e}_{d,k} \rVert_2^2, \qquad \mathbf{r}_d = \mathbf{r}_{d-1} - \mathbf{e}_{d,\, c_{i,d}}$$
> The Semantic ID and the quantized reconstruction of the latent are:
> $$\text{id}(i) = (c_{i,1}, c_{i,2}, \ldots, c_{i,L}), \qquad \hat{\mathbf{z}}_i = \sum_{d=1}^{L} \mathbf{e}_{d,\, c_{i,d}}$$
>
> where:
> - $\mathbf{x}_i$ — input item embedding (e.g., a Sentence-T5 vector of title/brand/category text)
> - $\text{Enc}, \text{Dec}$ — DNN encoder / decoder; $\hat{\mathbf{x}}_i = \text{Dec}(\hat{\mathbf{z}}_i)$
> - $\mathbf{r}_{d-1}$ — residual entering level $d$ (what the previous codebooks failed to capture)
> - $\mathbf{e}_{d,k}$ — the $k$-th codeword vector of codebook $d$; $K$ = codebook size, $L$ = number of levels (ID length)
> - $c_{i,d}$ — index of the nearest codeword at level $d$ (one entry of the Semantic ID)

> [!formula] Training loss
> Train encoder, decoder, and all codebooks jointly:
> $$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{rqvae}}, \qquad \mathcal{L}_{\text{recon}} = \lVert \mathbf{x}_i - \hat{\mathbf{x}}_i \rVert_2^2$$
> $$\mathcal{L}_{\text{rqvae}} = \sum_{d=1}^{L} \Big( \lVert \,\text{sg}[\mathbf{r}_{d-1}] - \mathbf{e}_{d,c_{i,d}} \rVert_2^2 \;+\; \beta\,\lVert \mathbf{r}_{d-1} - \text{sg}[\mathbf{e}_{d,c_{i,d}}] \rVert_2^2 \Big)$$
>
> where:
> - $\mathcal{L}_{\text{recon}}$ — reconstruction: the decoder must recover the original embedding from the quantized latent
> - $\mathcal{L}_{\text{rqvae}}$ — quantization loss: a **codebook term** (pull each codeword toward its assigned residuals) plus a $\beta$-weighted **commitment term** (pull the encoder output toward the chosen codeword)
> - $\text{sg}[\cdot]$ — stop-gradient (the $\arg\min$ is non-differentiable, so gradients flow via a straight-through estimator)
> - $\beta$ — commitment-loss weight

## Key Properties / Variants

- **Offline + frozen**: the tokenizer is learned *before* the generator and held fixed; the recommender predicts indices $(c_1,\ldots,c_L)$, not embeddings (RS-L04 slides 23–25).
- **Hierarchical / coarse-to-fine**: $c_1$ is broad, later codes refine; shared prefixes encode similarity and enable [[Cold Start]] generalization (a new item is tokenized by the frozen encoder and reuses existing codes).
- **Capacity decoupling**: $K^L$ addressable items from $L \cdot K$ codewords — small vocabulary, huge item space (RS-L04 slide 22).
- **Collision handling**: distinct items may hash to the same tuple, so TIGER appends a uniquifying token: $(12,24,52) \to (12,24,52,0), (12,24,52,1)$ — guaranteeing each Semantic ID maps to one catalogue item.
- **Validity**: most of the $K^L$ codes are unused, so decoding the generator must be constrained (a [[Trie]] / [[Trie-Constrained Decoding]]) to emit only real items.
- **Reconstruction-only weakness**: RQ-VAE optimizes recovery, not neighborhood structure — later work adds a contrastive objective (CoST) or collaborative/diversity regularizers (LETTER) so codes also reflect how users *use* items, not just item content.
- **Related quantizers** (RS-L04 slides 28–29): RQ-KMeans / RK-Means / R-VQ are simpler residual-quantization variants and can be competitive; [[Product Quantization]] (VQ-Rec) splits the embedding into subspaces instead of taking residuals; hierarchical clustering gives root-to-leaf path IDs. RQ-VAE is a canonical starting point, not a universal best.

```pseudo
Algorithm: RQ-VAE encode item -> Semantic ID
──────────────────────────────────────────────
Input: item embedding x_i ; codebooks C_1..C_L (each K codewords)
  z  ← Enc(x_i)            # latent vector
  r  ← z                   # initial residual
  id ← ()                  # empty Semantic ID
  for d = 1 .. L:
      c ← argmin_k || r - C_d[k] ||^2     # nearest codeword index
      id ← id ++ (c)                       # append index
      r  ← r - C_d[c]                      # subtract; pass residual on
  z_hat ← sum over d of C_d[id[d]]         # quantized latent
  return id                                 # ( decode z_hat -> x_hat only at train time )
```

## Connections

- Produces: [[Semantic ID]] / [[Semantic IDs]] (the L3 codebook-based level of [[Item Tokenization]])
- Special case of / contrasted with: [[Atomic Item IDs]] (one token per item; the degenerate $L=1$, codebook = catalogue case)
- Alternative quantizers: [[Product Quantization]], and plain vector quantization (one codebook)
- Consumed by: [[Generative Recommendation]] / [[Generative Retrieval]] models that decode IDs autoregressively
- Encoder input often from: [[Word Embeddings]] / content encoders (Sentence-T5 in TIGER)
- Generation must be grounded via: [[Trie-Constrained Decoding]] / [[Beam Search]]
- Compare to text tokenization: [[Tokenization]] (BPE subwords vs. learned item codes)

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
