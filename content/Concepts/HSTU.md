---
type: concept
aliases: [Hierarchical Sequential Transduction Unit]
course: [RecSys]
tags: [sequential-rec, generative-rec, llm, exam-topic]
status: complete
---

# HSTU

## Definition

> [!definition] HSTU (Hierarchical Sequential Transduction Unit)
> **HSTU** is Meta's attention-based encoder block (Zhai et al., "Actions Speak Louder than Words," ICML 2024) that reframes **discriminative CTR prediction as a generative sequential-transduction problem**. Instead of collecting many pointwise $(\text{user}, \text{item})$ samples and learning a scoring function $f(\text{user}, \text{item})$, HSTU merges a user's whole history into **one chronological sequence of interleaved items and actions** and predicts the next action/item **autoregressively**. It is the founding [[Large Recommendation Models (LRM)|LRM]]: a recommendation-native architecture in which a **scaling law** emerges from behavior data rather than being borrowed from language.

## Intuition

> [!intuition] Generative Recommendation (GR) vs. DLRM
> A classical Deep Learning Recommendation Model (DLRM) treats every impression as an independent example: extract numerical/categorical features, embed them, cross them (FMs, DCN), and score. Stacking more of this gives diminishing returns — industrial Model FLOPs Utilization (MFU) sits at only **0.1%–1%** versus ~70% for LLM inference.
>
> HSTU instead says: **a user's behavior already IS a sequence**, just like text is a sequence of tokens. So pose it as next-token prediction over (item, action) tokens. This single move (1) unifies **retrieval and ranking** in one model, (2) lets the model exploit the *full* long history causally instead of a retrieved short slice, and (3) — crucially — makes performance **keep climbing with compute**, the way LLMs do, instead of plateauing.

## Mathematical Formulation

HSTU stacks blocks of a modified [[Self-Attention]]. Within one block, given an input sequence representation $X$ (one row per interaction token):

> [!formula] One HSTU Block (read bottom-up)
> $$U, Q, K, V = \phi_1\!\big(f_1(X)\big)$$
> $$A(X) = \phi_2\!\big(Q K^{\top} + \mathrm{rab}^{p,t}\big)$$
> $$Y(X) = f_2\!\Big(\mathrm{Norm}\big(A(X)\,V(X)\big) \odot U(X)\Big)$$
>
> where:
> - $X$ — sequentialized, unified interaction features (items + actions + side features) at the block input
> - $f_1$ — a linear projection splitting into four streams; $f_2$ — output projection
> - $U, Q, K, V$ — the **gate**, **query**, **key**, **value** streams (HSTU computes an extra gate $U$ that standard attention lacks)
> - $\phi_1, \phi_2$ — **pointwise nonlinearities (SiLU)**, replacing the row-wise softmax of standard attention
> - $\mathrm{rab}^{p,t}$ — **relative attention bias** over relative **position** $p$ and **time** $t$ (encodes recency/ordering)
> - $A(X)$ — the **pointwise aggregated attention** matrix (NOT normalized over keys: no softmax)
> - $\odot$ — elementwise **gating** by the gate stream $U(X)$
> - $\mathrm{Norm}$ — layer normalization of the attended values
>
> Blocks are stacked with Add&Norm residual connections, mirroring a Transformer.

The block plugs into the **generative objective**. With items mapped to tokens and the history flattened to $(x_1, \dots, x_t)$, HSTU is trained causally to predict the next token:

> [!formula] Generative (Autoregressive) Training Target
> $$p_\theta(x_{t+1} \mid x_1, \dots, x_t), \qquad \mathcal{L} = -\sum_t \log p_\theta(x_{t+1} \mid x_{\le t})$$
>
> where actions and items are **interleaved** in one stream, so the same model produces both retrieval candidates (which item next) and ranking signals (which action on it). This is the same next-token cross-entropy used for [[Sequential Recommendation]] and for [[Large Language Models (LLM)|LLM]]s — the difference is only that the tokens are recommendation events, not words.

## Key Properties / Variants

- **Non-softmax (pointwise) attention.** $\phi_2 = \mathrm{SiLU}$ applied elementwise replaces softmax. This drops the row-normalization constraint, which (a) lets the model represent the *intensity* of many interactions (high-cardinality, streaming vocabulary) rather than a probability distribution over a fixed set, and (b) is far cheaper to fuse on GPU.
- **Extra gating stream $U$.** Beyond $Q,K,V$, HSTU computes a gate $U(X)$ and multiplies it elementwise into the attended output — a lean, learned filter that increases expressivity at low cost.
- **Relative position + time bias** $\mathrm{rab}^{p,t}$ injects ordering and recency directly into the attention scores, important because recommendation streams are irregularly timed.
- **Ragged fused-GEMM kernels.** Because user sequences vary in length, HSTU uses *ragged* (jagged) tensors and fuses the attention math into custom kernels → **5–15× faster than FlashAttention-2** at sequence length 8192. This is how it "stays inside the latency budget" while scaling.
- **Unifies retrieval + ranking** in a single model, collapsing part of the multi-stage cascade (Retrieval/[[Reranking|rank]]) that classical industrial pipelines use.
- **Recommendation scaling law.** Plotted on the LLM compute-scaling chart, HSTU-style **GR** models follow the same upward trend as LLMs and beat a heavily-tuned production DLRM — the headline empirical claim of the paper.
- **Architecture family.** HSTU is the **attention-oriented, model-scaling** branch of LRMs (alongside Meta's KunLun). Contrast with FFN-oriented scaling (RankMixer, UniMixer, which drop self-attention for parameter-free token-mixing), data-scaling along sequence length (LONGER, TWIN-V2) or feature dimension (Wukong), and unified single-backbone designs (OneTrans).
- **Usage in GenRec.** HSTU is the canonical **decoder-only**, industrial-scale generative recommender, contrasted with the encoder–decoder, [[Semantic ID]]-based TIGER. It scales to very long histories (~1000+) where TIGER assumes short ones (~50).

```pseudo
HSTU — Generative Sequential Transduction (forward, one user)
─────────────────────────────────────────────────────────────
Input: chronological interaction stream X = (item/action tokens + side features)
Sequentialize & preprocess  →  per-token embeddings
for each stacked HSTU block:
    U, Q, K, V = SiLU(Linear(X))            # 4 streams incl. gate U
    A = SiLU(Q Kᵀ + rab^{p,t})              # pointwise attn, NO softmax, causal mask
    Z = Norm(A · V) ⊙ U                     # gate by U elementwise
    X = X + Linear(Z)                        # Add & Norm residual
predict next token p(x_{t+1} | x_≤t)         # retrieval + ranking jointly, autoregressive
```

## Connections

- Founding instance of: [[Large Recommendation Models (LRM)]] (attention-oriented, model-scaling branch)
- Reformulates: discriminative CTR into [[Generative Recommendation]]
- Built on / modifies: [[Self-Attention]], [[Transformer Model]] (replaces softmax with pointwise SiLU + gating)
- Trained with: [[Next-Item Prediction]] / next-token [[Supervised Fine-Tuning (SFT)|cross-entropy]] objective
- Scaling story: exhibits a recommendation-native [[Scaling Law]]; raises [[GPU Architecture|MFU]] via fused kernels
- Decoder-only contrast: [[TIGER]] (encoder–decoder, [[Semantic ID]]-based, short histories)
- Sibling LRM lines: RankMixer (FFN-oriented), Wukong (feature-dimension scaling), LONGER (long-sequence scaling), OneTrans (unified backbone) — plain text, not vault notes
- Builds on long-sequence lineage: [[Sequential Recommendation]] (SASRec, BERT4Rec, DIN/DIEN/SIM/TWIN)

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
