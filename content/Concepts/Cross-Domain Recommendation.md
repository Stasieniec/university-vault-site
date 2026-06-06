---
type: concept
aliases: []
course: [RecSys]
tags: [collaborative-filtering, llm, exam-topic]
status: complete
---

# Cross-Domain Recommendation

> *Lecture context: transferring recommendation knowledge across domains, aided by LLM world knowledge.*

## Definition

> [!definition] Cross-Domain Recommendation (CDR)
> **Cross-domain recommendation** transfers preference knowledge learned in a **source domain** $\mathcal{D}^{S}$ (e.g., books) to improve recommendation in a **target domain** $\mathcal{D}^{T}$ (e.g., movies), typically to combat [[Data Sparsity]] and the [[Cold Start Problem]] in the target. A "domain" is any partition of items (and sometimes users) by catalogue, platform, or modality, where the two domains share an **overlap** — overlapping users (the usual case), overlapping items, or shared semantic content.
>
> Classical CDR builds an explicit **bridge** that maps source-domain user representations into the target-domain space. The generative / LLM era reframes this: a pretrained [[LLM]] already encodes cross-domain **world knowledge**, so transfer can happen *implicitly* through shared semantics rather than a hand-built mapping.

## Intuition

> [!intuition] Borrow signal where you have it, spend it where you don't
> A new user on a movie service has almost no clicks (cold start), but the same user has a rich history on a books service. Their taste — "likes dark, character-driven narratives" — is **domain-invariant**. If we can express that latent taste once and project it into the movie space, we recommend well from day one.
>
> The hard part is the **semantic gap**: a book embedding and a movie embedding live in different spaces with different ID vocabularies, so you cannot just reuse the vector. Three escalating answers:
> 1. **Learn a bridge** $f^{S\to T}$ that maps source user factors to target user factors (classic embedding-and-mapping CDR).
> 2. **Share a backbone** so both domains are trained jointly and the representation is forced to be shared.
> 3. **Use language as the universal interface** — describe users and items in text, and let an [[LLM]]'s world knowledge supply the cross-domain prior for free. This is why the lecture lists *cross-domain transfer* as a core advantage of [[Generative Recommendation]].

## Mathematical Formulation

The canonical **embedding-and-mapping** formulation (e.g., the EMCDR scheme): learn user/item latent factors **separately** in each domain via [[Matrix Factorization]], then learn a mapping that aligns the spaces using the **overlapping users** $\mathcal{U}^{S\cap T}$ as supervision.

$$
\mathcal{L}_{\text{map}}
= \sum_{u \in \mathcal{U}^{S\cap T}}
\big\lVert\, f_{\theta}\!\big(\mathbf{p}^{S}_{u}\big) - \mathbf{p}^{T}_{u} \,\big\rVert_{2}^{2}
\qquad
\hat{r}^{T}_{u,i} = f_{\theta}\!\big(\mathbf{p}^{S}_{u}\big)^{\top}\, \mathbf{q}^{T}_{i}
$$

where:

- $\mathbf{p}^{S}_{u} \in \mathbb{R}^{d}$ — user $u$'s latent factor learned in the **source** domain
- $\mathbf{p}^{T}_{u} \in \mathbb{R}^{d}$ — user $u$'s latent factor learned in the **target** domain (only available for overlapping users)
- $\mathbf{q}^{T}_{i} \in \mathbb{R}^{d}$ — target-domain item $i$'s latent factor
- $f_{\theta}: \mathbb{R}^{d} \to \mathbb{R}^{d}$ — the **bridge / mapping network** (linear or MLP), trained on overlapping users
- $\mathcal{U}^{S\cap T}$ — set of users present in **both** domains (the supervision signal)
- $\hat{r}^{T}_{u,i}$ — predicted target-domain score for a **cold-start** user, obtained by mapping their source factor into the target space

At inference, a user with no target-domain history but a known source factor $\mathbf{p}^{S}_{u}$ is scored entirely through $f_{\theta}(\mathbf{p}^{S}_{u})$ — knowledge has been transferred across the domain boundary.

**LLM-based transfer (the lecture's framing).** Instead of an explicit $f_\theta$, both domains are verbalized into a shared text space and a single LLM scores or generates:

$$
\hat{y}^{T}_{u} = \text{LLM}\big(\text{prompt}(\,h^{S}_{u},\, h^{T}_{u},\, c^{T}\,)\big)
$$

where $h^{S}_{u}, h^{T}_{u}$ are the user's natural-language histories in each domain and $c^{T}$ describes the target item/candidate. The cross-domain prior is no longer learned from overlap data — it is **baked into the pretrained weights** (e.g., the model already "knows" that fans of a certain author enjoy a certain director). The mapping $f_\theta$ is replaced by the LLM's internal world knowledge.

## Key Properties / Variants

- **Overlap regimes:** user-overlap (most common; enables EMCDR-style bridges), item-overlap, and *fully non-overlapping* (hardest — requires content/semantic anchors, exactly where LLMs help).
- **Direction:** single-direction transfer ($\mathcal{D}^S \to \mathcal{D}^T$) vs. **dual/joint** transfer where both domains improve each other.
- **What gets transferred:** (i) latent factors via a bridge (EMCDR); (ii) a **shared embedding space** trained jointly; (iii) **semantic/world knowledge** via text or a shared codebook of [[Semantic IDs]].
- **Primary motivation:** mitigate [[Data Sparsity]] and target-domain [[Cold Start]] — the slide explicitly pairs cross-domain transfer with cold-start as the regime where generative beats discriminative.
- **Discriminative limitation:** a [[Collaborative Filtering]] scorer $f(\text{user},\text{item})$ is tied to a **fixed candidate pool** in one domain and cannot reason about an unseen domain's items; generation over a shared semantic vocabulary can.
- **LLM-as-Recommender route:** zero-shot prompting transfers across scenarios with **no fine-tuning**, because the model's pretraining already spans domains; the cost is prompt sensitivity and item **hallucination** (it may "recommend" a target item that does not exist), motivating generation grounding.
- **LLM-as-Enhancer route:** the LLM rewrites cross-domain histories into enriched text features that a downstream [[Recommender System]] consumes — knowledge transfer as feature augmentation.

Classic embedding-and-mapping pipeline:

```pseudo
Algorithm: Embedding-and-Mapping CDR (EMCDR-style)
──────────────────────────────────────────────────
Input: source interactions R^S, target interactions R^T,
       overlapping users U^{S∩T}

1. Learn source factors  {p^S_u}, {q^S_i}  via MF on R^S
2. Learn target factors  {p^T_u}, {q^T_i}  via MF on R^T
3. Train bridge f_θ on overlapping users:
     minimize  Σ_{u∈U^{S∩T}} || f_θ(p^S_u) − p^T_u ||²
4. For a cold-start target user u (has p^S_u, no p^T_u):
     p̂^T_u ← f_θ(p^S_u)
     score item i:  r̂^T_{u,i} ← (p̂^T_u)ᵀ q^T_i
5. Recommend Top-K items by r̂^T_{u,i}
```

> [!warning] Negative transfer and overlap dependence
> Transfer is **not always helpful**. If domains share little true preference structure, forcing a shared space causes **negative transfer** (source noise degrades the target). Classic bridges also depend on a sufficiently large overlapping-user set $\mathcal{U}^{S\cap T}$ for supervision — when overlap is tiny or absent, embedding-and-mapping collapses, which is the gap content-based and LLM-based methods fill. LLM transfer dodges the overlap requirement but inherits **hallucination** and **prompt sensitivity**, and its world knowledge can be stale or biased for niche/long-tail catalogues.

## Connections

- Mitigates: [[Data Sparsity]], [[Cold Start Problem]], [[Long Tail]]
- Built on: [[Matrix Factorization]], [[Collaborative Filtering]], [[Content-Based Recommendation]]
- Enabled in the generative era by: [[Generative Recommendation]], [[LLM-based Recommendation]], [[LLM-as-Enhancer]], [[LLM-as-Recommender]]
- Shared vocabulary mechanism: [[Semantic IDs]], [[Item Tokenization]]
- Contrasts with: discriminative single-domain scoring ([[Top-K Recommendation]] over a fixed pool)
- Knowledge source: [[Large Language Models (LLM)]] world knowledge; complements [[Hybrid Recommendation]]

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
