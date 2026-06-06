---
type: concept
aliases: [Cold Start]
course: [RecSys]
tags: [collaborative-filtering, generative-rec, llm, exam-topic]
status: complete
---

# Cold Start Problem

## Definition

> [!definition] Cold Start Problem
> The **cold start problem** is the difficulty of producing good recommendations for **new users** or **new items** that have **little or no interaction history**. Recommenders learn from the user–item interaction matrix; when a row (new user) or column (new item) is (nearly) empty, [[Collaborative Filtering]] and [[Matrix Factorization]] have no signal to exploit and fall back to popularity or fail outright. It is especially salient in **news recommendation**, where items (articles) are constantly created, are recency-sensitive, and accumulate clicks only after they have already been shown.

## Intuition

> [!intuition] No interactions, no embedding
> CF answers "users similar to you liked X" and "items similar to what you liked are Y" — both require *prior interactions*. A new item has been clicked by nobody, so no similar-user or co-consumption signal exists; a new user has clicked nothing, so there are no neighbours. In [[Matrix Factorization]] this manifests concretely: the latent factor $\bar{u}_i$ or $\bar{v}_j$ for a cold entity is initialized but never (or barely) updated by gradient descent, so its dot product with any other factor is meaningless. The embedding table also grows *linearly* with the catalogue — every new item needs a brand-new id **and** a freshly trained embedding before it can be scored.

## Mathematical Formulation

Cold start is exactly the failure mode of the standard scoring recipe. In MF the predicted affinity is

$$r_{ij} \approx \bar{u}_i \cdot \bar{v}_j$$

where:
- $\bar{u}_i$ — latent factor vector of user $i$ (learned from $i$'s past interactions)
- $\bar{v}_j$ — latent factor vector of item $j$ (learned from interactions on $j$)
- $\cdot$ — dot product modelling the user–item match

For a cold entity the relevant factor has **no informative gradient**: if user $i$ has rated nothing, $\bar{u}_i$ stays at its (random / prior) initialization and $r_{ij}$ is uninformative for every $j$. The same holds for a cold item $j$.

In a discriminative recommender this is the limitation of a learned scoring function

$$s_\theta(\text{user}, \text{item}) \quad \text{over a fixed candidate pool},$$

which by construction cannot score an item that is not in the trained candidate set. **Generative recommendation** reframes the problem so the cold item never needs its own learned embedding: items are assigned a hierarchical **Semantic ID** $\mathbf{z}_i = (z_{i,1}, \dots, z_{i,L})$ built from *content* by a frozen tokenizer (e.g., RQ-VAE in TIGER), and the model decodes the next-item id autoregressively:

$$p_\theta(\mathbf{z}_i \mid \mathbf{x}) = \prod_{\ell=1}^{L} p_\theta\!\left(z_{i,\ell} \mid \mathbf{x},\, z_{i,<\ell}\right)$$

where:
- $\mathbf{x} = (x_1, \dots, x_t)$ — the user's interaction history
- $\mathbf{z}_i$ — the item's fixed-length code (a few shared codebook tokens, $K$ choices per position, $L$ positions ⇒ $K^L$ possible ids)
- $z_{i,<\ell}$ — codebook tokens already decoded

Because a new item is mapped to a Semantic ID **from its content** through the frozen tokenizer, all its sub-tokens already exist in the codebook and it shares **prefixes** with similar items — so it becomes *decodable* the moment it is tokenized, without retraining.

## Key Properties / Variants

- **Two flavours:** *user cold start* (new user, empty profile row) and *item cold start* (new item, empty interaction column). News recommendation suffers acute item cold start (new articles every minute) and recency makes it worse.
- **Why atomic ids break:** with one [[Atomic Item ID]] per item, the output is a softmax over the whole catalogue; a new item needs a new vocabulary token **and** a learned embedding row — until trained, it is strictly unrecommendable (popularity bias dominates the softmax).
- **Generative promise vs. fragile reality (RS-L04):** a Semantic ID makes a cold item *decodable*, but being decodable is **not** the same as being recommended — the generator was trained only on Semantic IDs of items people actually clicked, so a fresh item's id has near-zero probability and beam search **prunes it early** ("cold start becomes fragile").
- **Practical fix — hybrid (LIGER):** do not rely on the generator to "think of" cold items. (1) generate the usual (warm) candidates, (2) **inject** cold items into the candidate pool by hand, (3) **re-rank** with dense content embeddings so cold items get a fair, content-based score. The future of GenRec may be hybrid, not purely generative.
- **LLM route:** using a pretrained [[LLM]] directly (prompting / in-context learning, no fine-tuning) is strong on cold start and cross-domain because it leans on world knowledge and semantics rather than interaction counts — one of the headline advantages of generative recommendation.
- **Mitigations summary:**

```pseudo
Mitigating cold start
─────────────────────────────────────────────
Content-based / hybrid:
  use item content (text, metadata, image) → embedding
  → score by content similarity (no interactions needed)

Generative (Semantic IDs):
  new_item → frozen content tokenizer (RQ-VAE) → (z1,...,zL)
  add path to trie  →  item now decodable
  BUT bias toward warm ids → pair with dense re-ranker (LIGER)

LLM-as-recommender:
  prompt pretrained LLM (world knowledge) → zero-shot
  → good cold-start / cross-domain generalization
```

## Connections

- Failure mode of: [[Collaborative Filtering]], [[Matrix Factorization]], [[Content-Based Filtering]] (content helps the item side)
- Related sparsity issue: [[Data Sparsity]], [[Long-Tail Distribution]]
- Addressed by: [[Hybrid Recommendation]], [[Generative Recommendation]], [[Semantic ID]], [[LLM-as-Recommender]]
- Item representation choice: [[Atomic Item ID]] (worsens it) vs [[Semantic IDs]] (eases it via shared prefixes)
- Decoding interaction: [[Beam Search]] / [[Trie-Constrained Decoding]] (where the generative cold item gets pruned)

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L03a - Sequential Recommendation Models]]
- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
