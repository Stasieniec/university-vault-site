---
type: concept
aliases: [TK]
course: [IR]
tags: [neural-ir, reranking, exam-topic]
status: complete
---

# Transformer Kernel (TK)

## Definition

> [!definition] Transformer Kernel (TK)
> **TK** (Transformer-Kernel) is an interaction-focused neural ranking model that sits *between* the cheap [[Bi-Encoder]] and the expensive [[Cross-Encoder]] on the efficiency–effectiveness curve. It first **contextualizes** query and document tokens *independently* with a small stack of [[Transformer Model|Transformer]] layers, then scores them with **RBF kernel-pooling** applied to the query–document cosine-similarity matrix (the same kernel idea as KNRM). Because the two sides are encoded separately, document contextualization can be **pre-computed**, making TK far cheaper than a full all-to-all cross-encoder while still capturing soft, term-level interactions.

## Intuition

> [!intuition] "Cheap contextualization, then count the matches by strength"
> A [[Cross-Encoder]] like [[MonoBERT]] is accurate because every query token attends to every document token — but that all-to-all attention must be recomputed for every candidate at query time, so it costs $O(L^2)$ per document and cannot be cached.
>
> TK splits the work. The expensive part — contextualizing tokens with attention — is done on each text *on its own* (so the document side is reusable/indexable). The cheap part — comparing query and document — is reduced to a single similarity matrix that is summarized by a bank of Gaussian (RBF) kernels. Each kernel acts as a soft histogram bin: one kernel counts "near-exact" matches (similarity $\approx 1$), another counts "loosely related" matches (similarity $\approx 0.5$), and so on. The model learns how much each match-strength band should contribute to relevance. This is the [[Neural Reranking|interaction-focused]] philosophy (like DRMM/KNRM) but with learned contextual embeddings instead of static word vectors.

## Mathematical Formulation

TK has three stages: contextualization, kernel-pooling over the match matrix, and a linear scoring layer.

**(1) Contextualization (independent per text).** Embed query $q = (q_1,\dots,q_{|q|})$ and document $d = (d_1,\dots,d_{|d|})$, then pass each *separately* through a shallow Transformer stack. A residual mixing weight $\alpha$ blends the raw embedding $t_i$ with its contextualized version:

$$
\hat{t}_i = \alpha\, t_i + (1-\alpha)\, \mathrm{Transformer}(t)_i
$$

**(2) Match matrix + kernel-pooling.** Build the cosine-similarity matrix $M$ between every contextualized query and document token, then apply $K$ Gaussian RBF kernels:

$$
M_{ij} = \cos\!\big(\hat{q}_i,\, \hat{d}_j\big),
\qquad
K_k(M_{ij}) = \exp\!\left(-\frac{(M_{ij}-\mu_k)^2}{2\sigma_k^2}\right)
$$

Each query row is pooled over document tokens, then log-summed over query tokens to give one soft-count feature per kernel:

$$
\phi_k(q,d) = \sum_{i=1}^{|q|} \log \left( \sum_{j=1}^{|d|} K_k(M_{ij}) \right)
$$

**(3) Scoring.** The $K$ kernel features are combined by a learned linear layer into the final relevance score:

$$
s(q,d) = \sum_{k=1}^{K} w_k\, \phi_k(q,d) + b
$$

where:
- $t_i,\ \hat{t}_i$ — the raw and contextualized embedding of token $i$ (applied to both $q$ and $d$)
- $\alpha \in [0,1]$ — learned residual weight trading off raw vs. contextual signal (small Transformer means context is added, not replacing the embedding)
- $M_{ij}$ — cosine similarity between contextualized query token $i$ and document token $j$ (the **match / interaction matrix**)
- $\mu_k,\ \sigma_k$ — center and width of kernel $k$; $\mu_k$ selects a similarity band, $\sigma_k$ its tolerance. A kernel with $\mu \approx 1,\ \sigma \to 0$ approximates exact matching.
- $\phi_k(q,d)$ — soft match-count for band $k$ (log-sum-exp pooling gives length normalization and gradient stability)
- $w_k, b$ — learned weights/bias of the final linear scorer
- $K$ — number of kernels (e.g. 11, evenly spaced $\mu_k$ over $[-1, 1]$)

TK is trained as a [[Pairwise Learning to Rank|pairwise]] reranker with a margin / RankNet-style loss on $(q, d^+, d^-)$ triples, minimizing $\max(0,\ 1 - s(q,d^+) + s(q,d^-))$ (see [[Hard Negatives]] for negative selection).

## Key Properties / Variants

- **Position on the efficiency curve:** more effective than a single-vector [[Bi-Encoder]] (it keeps per-token interactions) but much cheaper than a [[Cross-Encoder]] (no all-to-all attention across the concatenated pair).
- **Pre-computable document side:** because contextualization is independent, document representations and even partial match structures can be cached/indexed, cutting query-time cost.
- **Shallow Transformer:** typically only 2–3 layers (not a full 12-layer [[BERT for IR|BERT]]); the residual gate $\alpha$ lets it add contextual signal without destroying the lexical signal in the raw embeddings.
- **Interpretable kernels:** each $\mu_k$ corresponds to a human-readable match strength, so $w_k$ shows which match bands the model relies on.
- **Variants:**
  - **TK-Sparse / TKL (Transformer-Kernel for Long documents):** scans long documents in overlapping windows and aggregates per-region saliency, extending TK beyond the 512-token limit.
  - **CK (Conv-Kernel):** replaces the Transformer contextualizer with CNNs over n-grams.
  - Conceptual ancestor: **KNRM / Conv-KNRM** (kernel-pooling over *static* embeddings); TK = KNRM kernels + learned contextual embeddings.

```pseudo
Algorithm: TK Scoring (query q, document d)
─────────────────────────────────────────────
# 1. Independent contextualization (document side cacheable)
Q ← Embed(q);  D ← Embed(d)
Q̂ ← α·Q + (1-α)·Transformer(Q)
D̂ ← α·D + (1-α)·Transformer(D)        # can be precomputed/indexed

# 2. Match matrix
for each query token i, doc token j:
    M[i,j] ← cosine(Q̂[i], D̂[j])

# 3. Kernel-pooling (K Gaussian kernels with centers μ_k, widths σ_k)
for k in 1..K:
    for i in 1..|q|:
        soft_count[i] ← Σ_j exp( -(M[i,j] - μ_k)^2 / (2 σ_k^2) )
    φ[k] ← Σ_i log( soft_count[i] )    # log-sum-exp pooling

# 4. Linear scoring
return  Σ_k w_k · φ[k] + b
```

## Connections

- Interpolates between: [[Bi-Encoder]] (independent encoding) and [[Cross-Encoder]] (full interaction)
- Family: [[Neural Reranking]], interaction-focused models; contrast with [[MonoBERT]] (cross-encoder) and [[ColBERT]] (late-interaction MaxSim instead of kernel-pooling)
- Building block: [[Transformer Model]] / [[Attention architecture]] for contextualization
- Used in: [[Multi-Stage Ranking]] pipelines as a second-stage reranker over [[BM25]] candidates
- Trained with: [[Pairwise Learning to Rank]] losses and [[Hard Negatives]]
- Tackles the same vocabulary-mismatch problem as [[Dense Retrieval]]

## Appears In

- [[IR-PTR Ch3 - Multi-Stage Architectures for Reranking]]
