---
type: concept
aliases: [LRM]
course: [RecSys]
tags: [generative-rec, sequential-rec, llm, exam-topic]
status: complete
---

# Large Recommendation Models (LRM)

> Recommendation-native architectures designed for behavior data so a scaling law emerges natively.

## Definition

> [!definition] Large Recommendation Models (LRM)
> An **LRM** is a recommendation-native architecture that scales the *recommendation* problem directly, rather than translating it into a language task. Where [[LLM-based Recommendation|LLM-based Generative Recommendation]] *borrows* the [[Scaling Law]] from language (forcing behavior data into text), an LRM **designs native architectures for recommendation/behavior data and lets a recommendation-native scaling law emerge there** (Hou et al. 2025, §4.2).
>
> Concretely: take a user's raw interactions (items, actions, side features), express them as a single **sequence of tokens**, and train one large model end-to-end so that *accuracy keeps improving as compute grows* — the same compute → quality relationship LLMs enjoy, but now intrinsic to behavior data.

## Intuition

> [!intuition] Borrow scaling vs. grow your own
> Over the last decade RecSys gains came from raising compute (rule-based → linear → deep → long-sequence), tracing a **generalized scaling law** (sigmoid of accuracy vs. FLOPs/item). But the classic [[Multi-Stage Ranking|cascaded pipeline]] (retrieval ~$10^8$ → pre-rank ~$10^4$ → rank ~$10^3$) wastes most of that compute:
> - **Fragmented computing** — industrial RS [[GPU Architecture|MFU]] (Model FLOPs Utilization) sits at **0.1%–1%**, while LLM inference reaches **~70%**.
> - **Inconsistent objectives** — balancing *hundreds* of objectives degrades consistency.
> - **Technological gap** — architectural disconnect from validated LLM tech.
>
> The LRM bet: **reclaim wasted compute** (cheaper/approximate ops, cache & reuse, raise MFU) and **spend it on scale**, while staying inside the serving latency budget. Unlike LLM-based GR, an LRM does *not* depend on web-text world knowledge — its fuel is the platform's massive behavior log.

## Mathematical Formulation

The canonical LRM mechanism is **HSTU** (Hierarchical Sequential Transduction Unit, Zhai et al. ICML 2024), which reframes discriminative CTR scoring as **generative sequential modeling**: collapse the many pointwise samples per user into one chronological sequence $X$ of interleaved items, actions, and features, and predict the next action/item **causally** — unifying retrieval and ranking in one model.

The HSTU block (stacked with Add&Norm residuals) replaces softmax attention with **pointwise** attention:

$$
\begin{aligned}
U, Q, K, V &= \phi_1\!\big(f_1(X)\big) \\
A(X) &= \phi_2\!\left(Q K^{\top} + \mathrm{rab}^{p,t}\right) \\
\text{out} &= \mathrm{Norm}\!\big(A(X)\,V(X)\big) \odot U(X) \\
Y(X) &= f_2(\text{out})
\end{aligned}
$$

where:
- $X$ — the sequentialized, unified feature stream (items + actions + user/item features, in time order)
- $f_1, f_2$ — linear projections (input split / output transform)
- $\phi_1, \phi_2$ — **pointwise nonlinearities (SiLU)**, *not* a row-wise softmax; this is the key departure from [[Self-Attention|standard attention]]
- $Q, K, V$ — query / key / value; $U$ — a gating branch
- $\mathrm{rab}^{p,t}$ — **relative position + time attention bias** (recommendation has both ordering and timestamps)
- $\odot$ — elementwise gating by $U(X)$ (a learned multiplicative gate, replacing the softmax-FFN combo)
- $\mathrm{Norm}$ — normalization applied after the value aggregation

Because $\phi_2$ is pointwise, the operation maps onto **ragged fused-GEMM kernels** giving **5–15× speed-up over FlashAttention-2** at sequence length 8192 — i.e., the architecture is co-designed with the hardware so that scaling stays affordable.

## Key Properties / Variants

LRMs are organized along two scaling axes — **DATA** (sequence length / feature dimension) × **MODEL** (attention-oriented / FFN-oriented) — plus a **unified** route that scales both.

- **DATA · sequence length — LONGER** (ByteDance, RecSys 2025): first to model the *full* ultra-long history end-to-end (to **10,000 tokens**) instead of retrieving a short slice (DIN/DIEN/SIM/TWIN). Uses **global tokens** (target/user/CLS) with full receptive field as an *attention sink*, **token merge**, and **KV-cache serving** (encode the user once, reuse across candidates) — cutting online throughput loss from −40% to −6.8%; clean power-law scaling; +6.5% GMV/user live.
- **DATA · feature dimension — Wukong** (Meta, ICML 2024): a scaling law for *feature interactions* (not sequences). Stacks **Factorization-Machine blocks** so interaction order grows exponentially with depth — layer $i$ covers all orders up to $2^i$ — via parallel FMB (FM+MLP, raises order) and LCB (linear compress, preserves low orders), concat + Add&Norm.
- **MODEL · attention-oriented — HSTU** (Meta, ICML 2024): the equation above; generative reformulation + pointwise (non-softmax) attention + fused kernels.
- **MODEL · FFN-oriented — RankMixer** (ByteDance, 2025): drops self-attention entirely (rec features are *heterogeneous*, attention assumes one shared space + quadratic cost). Uses a **parameter-free Multi-Head Token Mixing** (shuffle feature subspaces across tokens) + **Per-Token FFN** (each feature token its own MLP) + optional Sparse-MoE. Result: MFU **4.5% → 45%** with flat latency.
- **Unified scaling — OneTrans** (ByteDance, WWW 2026): one Transformer backbone for *both* axes. **Unified tokenizer** (sequential + non-sequential attributes → one token stream), **mixed parameterization** (shared params for similar sequential tokens, token-specific for non-sequential), **causal attention + cross-request KV-cache + FlashAttention** — "scales like an LLM and serves like one."

> [!warning] LRM ≠ end-to-end generative ≠ LLM-based GR
> Most LRMs above (HSTU, Wukong, RankMixer, LONGER, OneTrans) still output a **score/next-action** and serve inside the cascaded pipeline — they scale the architecture, not the output format. A *separate* route is **end-to-end generative** (OneRec / GPR / PLUM), which generates [[Semantic IDs|semantic-ID]] item tokens autoregressively. Do not conflate: (a) LRM = native architecture + native scaling law; (b) LLM-based GR = borrow language scaling, squeeze behavior into text; (c) end-to-end generative recommender = generate the item tokens directly.

> [!formula] The unifying engineering trick
> $$\text{reclaimed compute} = \underbrace{\text{cheaper / approximate ops}}_{\text{e.g. pointwise attn, token mixing}} + \underbrace{\text{cache \& reuse}}_{\text{KV-cache}} + \underbrace{\text{raise MFU}}_{0.1\text{–}1\% \;\to\; \sim 70\%}$$
> Spend the reclaimed compute on **scale** (longer sequences, more interaction order, bigger model) while holding the **serving latency budget** fixed.

## Connections

- Contrasts with: [[LLM-based Recommendation]] (borrows language scaling) — this is the crux of [[RS-L03b - From LLMs to LRMs]]
- Realizes: [[Scaling Law]] natively for behavior data; reframes [[Sequential Recommendation]] as generative [[Next-Item Prediction]]
- Built on: [[Transformers|Transformer]] / [[Self-Attention]] (HSTU, OneTrans) and [[Matrix Factorization|Factorization Machines]] (Wukong)
- Replaces / augments: the [[Multi-Stage Ranking]] cascaded pipeline; unifies retrieval and ranking in one model
- Hardware co-design: [[GPU Architecture]], [[Kernel Fusion]] (fused-GEMM), KV-cache
- Sibling route: [[Generative Recommendation]] / [[Generative Retrieval]] via [[Semantic IDs]] and [[Item Tokenization]] (end-to-end generative, next lecture)

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
