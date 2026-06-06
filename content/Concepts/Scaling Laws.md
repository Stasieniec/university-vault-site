---
type: concept
aliases: [Scaling Law]
course: [RecSys]
tags: [generative-rec, llm, exam-topic]
status: complete
---

# Scaling Laws

## Definition

> [!definition] Scaling Law
> A **scaling law** is the empirical regularity that model quality (e.g. loss, accuracy) improves **predictably and monotonically** as you increase three resources — **data, compute (FLOPs), and parameters** — with *no early plateau*. For [[Large Language Models (LLM)|LLMs]] this is the central reason "bigger = reliably better." Classical recommenders historically did **not** exhibit this: past a point, enlarging them barely helped. The thesis of modern [[Generative Recommendation]] (and especially [[Large Recommendation Models (LRM)|LRMs]]) is that recommendation *can* be made to follow a scaling law — by reframing the stream of user actions as a token stream and using architectures (notably [[HSTU]]) built for very long histories.

## Intuition

> [!intuition] Why recommenders used to plateau
> A classical discriminative recommender learns a scoring function $f(\text{user}, \text{item})$ over a *fixed candidate pool*. Stacking more layers or features gives **diminishing returns**: the supervision signal (a single clicked item per example) and the fixed output space cap how much extra capacity can help. LLMs avoid this because next-token prediction provides dense supervision over an effectively unbounded sequence space, so more parameters keep finding structure to model.
>
> The key move ([[HSTU]], "Actions Speak Louder than Words", Zhai et al. ICML 2024) is to make recommendation *look like* language modelling: collapse the many pointwise samples per user into **one chronological behaviour sequence** of interleaved items and actions, and predict the next one autoregressively. Once recommendation is next-token prediction over actions, the same compute-scaling behaviour that drives LLMs reappears — performance keeps climbing with compute and beats a heavily-tuned production DLRM.

## Mathematical Formulation

A scaling law states that test loss $L$ falls as a **power law** in each scaled resource, holding the others non-bottlenecking:

$$L(N) \approx L_\infty + \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad L(D) \approx L_\infty + \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad L(C) \approx L_\infty + \left(\frac{C_c}{C}\right)^{\alpha_C}$$

where:

- $N$ — number of model **parameters**
- $D$ — number of training tokens / interactions (**data**)
- $C$ — total training **compute** (FLOPs); roughly $C \approx 6 N D$ for a transformer
- $L_\infty$ — irreducible loss (the floor set by data entropy)
- $N_c, D_c, C_c$ — characteristic constants (the resource scale at which the gap to $L_\infty$ is $\sim 1$)
- $\alpha_N, \alpha_D, \alpha_C > 0$ — power-law exponents; on a log-log plot $\log(L - L_\infty)$ is **linear** in $\log N$, $\log D$, $\log C$

For recommendation, the dependent variable is typically reframed as **accuracy / recall / NDCG vs. compute per item**, which traces a sigmoid-shaped rising curve (the "generalized scaling law" the course plots over the eras rule-based → linear → deep → long-sequence). The underlying mechanism enabling it (HSTU): instead of a softmax-attention transformer, use a **pointwise aggregated attention** block so very long histories stay affordable:

$$U, Q, K, V = \phi_1\!\big(f_1(X)\big), \qquad A(X) = \phi_2\!\big(Q K^{\top} + \mathrm{rab}^{p,t}\big), \qquad Y(X) = f_2\!\Big(\mathrm{Norm}\big(A(X)\,V(X)\big) \odot U(X)\Big)$$

where:

- $X$ — the sequentialized, unified feature/behaviour input
- $\phi_1, \phi_2$ — pointwise nonlinearities (**SiLU**, *not* softmax) — this is what makes the attention "pointwise" and removes the global normalization cost
- $\mathrm{rab}^{p,t}$ — **r**elative **a**ttention **b**ias over position $p$ and time $t$
- $\odot U(X)$ — elementwise **gating** by the learned $U$ branch
- $f_1, f_2$ — linear projections; blocks stack with Add & Norm residuals

The payoff is efficiency that frees compute to *spend on scale*: ragged fused-GEMM kernels make HSTU **5–15× faster** than FlashAttention-2 at length 8192, and the resulting model exhibits clean power-law scaling.

## Key Properties / Variants

- **Two routes to a recommendation scaling law:**
  - *Borrowed scaling (LLM-based GR):* squeeze behaviour into text and inherit the **language** model's scaling law. Strong for cold-start, cross-domain, explainable; bottlenecked by aligning to collaborative signal and grounding generated items.
  - *Native scaling ([[Large Recommendation Models (LRM)|LRM]]):* design architectures for behaviour data directly so a *recommendation-native* scaling law emerges. Strong for industrial main-feed ranking.
- **What you scale (the LRM landscape, two axes):**
  - *Data scaling* — sequence length (LONGER, TWIN-V2, SIM) and feature dimension / interaction order (Wukong).
  - *Model scaling* — attention-oriented ([[HSTU]], KunLun) and FFN-oriented (RankMixer, UniMixer); plus unified backbones (OneTrans, MTGR).
- **The unifying engineering trick — "scale up, stay inside the latency budget":** industrial recommenders run at **0.1%–1% Model FLOPs Utilization (MFU)** vs ~70% for LLM inference. Reclaim that compute via *cheaper/approximate ops · cache & reuse · raise MFU*, then spend it on scale. Concrete instances:
  - **LONGER** — scales sequence length to 10,000 tokens end-to-end; KV-cache (encode user once, reuse across candidates) cuts throughput loss −40% → −6.8%.
  - **Wukong** — scales feature-interaction order by stacking FM blocks: layer $i$ covers all orders up to $2^i$.
  - **HSTU** — scales attention via pointwise (non-softmax) attention + ragged fused-GEMM (5–15× over FlashAttention-2).
  - **RankMixer** — drops self-attention for parameter-free token-mixing + per-token FFN; MFU 4.5% → 45% at flat latency.
  - **OneTrans** — one backbone for both axes; causal attention + cross-request KV-cache so it "scales like an LLM and serves like one."
- **Empirical evidence (course figure):** plotting training PetaFLOP/s-days vs year places GenRec models (GR-23, GR-24) on the **same compute-scaling trend** as AlexNet → BERT → GPT-3 → LLaMA-2, while DLRMs sit off the trend.
- **When the scaling law pays off (generative > discriminative):** it requires **sufficient training compute** — generative models keep gaining where discriminative models saturate. Combined with world knowledge, NL understanding, reasoning, and creative generation, scaling law is one of the five enduring advantages of generative recommendation.

## Connections

- Realized in recommendation by: [[HSTU]], [[Large Recommendation Models (LRM)]], [[Generative Recommendation]]
- Borrowed from: [[Large Language Models (LLM)]] (language-model scaling)
- Contrasted with: classical [[Recommender System]] / discriminative scoring $f(\text{user}, \text{item})$ that plateaus
- Enabled by item-as-token machinery: [[Semantic IDs]], [[Item Tokenization]], [[RQ-VAE]]
- Architectural primitive: [[Self-Attention]] / [[Transformer Model]] (HSTU replaces softmax with pointwise SiLU attention)
- Efficiency lever: GPU [[Kernel Fusion]] (ragged fused-GEMM), KV-cache reuse, raising MFU

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
