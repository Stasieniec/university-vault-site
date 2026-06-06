---
type: concept
aliases: []
course: [RecSys]
tags: [generative-rec, sequential-rec, llm, exam-topic]
status: complete
---

# OneRec

## Definition

> [!definition] OneRec
> **OneRec** (Deng et al., Kuaishou, arXiv:2502.18965, 2025) is an **end-to-end generative recommender** that replaces the entire cascaded retrieve → pre-rank → rank pipeline with a **single encoder–decoder model**. It encodes a user's interaction history and **autoregressively generates the next item's identifier as a sequence of hierarchical [[Semantic IDs|semantic-ID]] tokens** (multi-codeword codes of the form `<item_a><item_b><item_c>`), which are then mapped back to real catalogue items (short videos). Candidate generation and ranking are learned **jointly** rather than as separate stages.

## Intuition

> [!intuition] Generate the item, don't score the catalogue
> A classical cascaded recommender funnels a corpus of $\sim 10^{10}$ videos down through Retrieval ($\sim 10^5$) → coarse ranking ($\sim 10^3$) → fine ranking ($\sim 10^2$) → a handful of recommendations. Each handoff loses information: **a recall error in an early stage can never be recovered by a later ranker**, and the overall Model FLOPs Utilization (MFU) of such systems is only ~0.1–1%.
>
> OneRec instead treats recommendation like a [[LLM|language model]] task: the user history is a "prompt", and the answer is the **identifier of the next item, decoded token by token**. Because items are tokenized into a small shared codebook, the model never needs a softmax over millions of atomic items, and it can be scaled (data + compute + parameters) like an LLM so a recommendation-native [[Scaling Laws|scaling law]] emerges. This is the "LLM-style end-to-end scaling" route to the [[Large Recommendation Models (LRM)|LRM]] goal, as opposed to merely scaling one stage of the old pipeline.

## Mathematical Formulation

The core mechanism is **autoregressive decoding over a fixed-length semantic identifier**. Each catalogue item $i$ is mapped (offline, by a quantization tokenizer such as [[RQ-VAE]]) to an $L$-level semantic ID $\mathbf{z}_i = (z_{i,1}, \dots, z_{i,L})$ drawn from per-level codebooks. Given a user history $\mathbf{x} = (x_1, \dots, x_t)$, the encoder produces a context and the decoder factorizes the next-item identifier:

$$p_\theta(\mathbf{z}_i \mid \mathbf{x}) = \prod_{\ell=1}^{L} p_\theta\!\left(z_{i,\ell} \mid \mathbf{x},\, z_{i,<\ell}\right)$$

where:
- $\mathbf{x} = (x_1, \dots, x_t)$ — the user's chronological interaction history (each $x_j$ an item, expanded into its $L$ SID tokens, so the decoder reads an $L\times$-longer flat token stream)
- $\mathbf{z}_i = (z_{i,1}, \dots, z_{i,L})$ — the hierarchical semantic ID of the target item; earlier codes are coarse (broad category), later codes refine the residual
- $z_{i,<\ell}$ — the SID tokens already generated for this item (teacher-forced at training time)
- $L$ — identifier length (number of codebook levels); $K$ — codebook size per level, so $K^L$ codes can index billions of items with only $K\cdot L$ token embeddings

Training uses standard **next-token cross-entropy** over the SID positions:

$$\mathcal{L}_{\text{CE}} = -\sum_{\ell=1}^{L} \log p_\theta\!\left(z_\ell \mid \mathbf{x},\, z_{<\ell}\right)$$

The identifier likelihood doubles as a relevance **score** for the item, $s_\theta(\mathbf{x}, i) = \log p_\theta(\mathbf{z}_i \mid \mathbf{x})$, so recommendation is "decode valid identifiers" rather than "score a fixed candidate set". On top of CE, OneRec adds a **reward / preference fine-tuning** stage (e.g. [[GRPO]] / [[DPO]]) to reward whole-list quality — validity, relevance, freshness, diversity — that pure next-token CE cannot express.

## Key Properties / Variants

- **Three-phase LLM-style pipeline:** Data ⇒ Pre-Training ⇒ (Mid-Training) ⇒ Post-Training ⇒ Test-Time. In the "Open OneRec" framing a [[LLM|Qwen LLM]] decoder drives all phases; in evaluation the generated itemic tokens are **mapped** back to actual short-video items.
- **Itemic + text tokens interleaved:** each short video is passed through a tokenizer that emits itemic tokens `<item_a_1><item_b_1><item_c_1>` (a hierarchical multi-codeword [[Semantic IDs|semantic ID]], three codebook levels a/b/c) interleaved with text tokens describing user behaviour.
- **Architecture:** an [[Generative Recommendation|encoder–decoder]] Transformer (the variant OneRec-V2 moves toward decoder-only); supports **long histories** (~1000+) and is multi-modal.
- **Unifies retrieve + rank:** one model does candidate generation and ranking jointly, removing the cascade's recall-error bottleneck and raising MFU toward LLM-level utilization.
- **Decoding must stay valid:** the $K^L$ SID space is far larger than the catalogue, so most code combinations correspond to no real item. Generation is paired with **[[Trie-Constrained Decoding|trie-constrained beam search]]** (a per-step logit mask over valid catalogue paths) and/or a validity reward in GRPO so emitted IDs are grounded in the catalogue.
- **Cold-start path:** a new item is run through the frozen tokenizer to obtain its SID and its path is added to the trie — it becomes *decodable* immediately (sub-tokens already exist in the codebook), though being decodable is not the same as being *recommended* (the generator was trained on clicked items).
- **Variants:** OneRec-V2 (decoder-only scaling) and OneRec-Think (test-time reasoning before recommending). Kuaishou reports it serving the production main feed end-to-end.

Sketch of the inference flow (constrained beam search over SIDs):

```pseudo
Algorithm: OneRec inference (constrained generation)
─────────────────────────────────────────────────────
Input: user history H = (i_1,...,i_t); beam width B; SID length L; valid-SID trie T
  Look up SID(i_j) for each i_j in H  →  flat token stream
  context ← Encoder(token stream)
  beams ← { empty prefix }
  for level ℓ = 1..L:
    candidates ← ∅
    for each prefix p in beams:
      allowed ← children of p in trie T          # validity mask
      for each token z in allowed:
        score ← logp(p) + log p_θ(z | context, p) # autoregressive step
        candidates ← candidates ∪ {(p·z, score)}
    beams ← top-B candidates by score             # prune the rest
  SIDs ← B complete identifiers in beams
  items ← trie/catalogue lookup of each SID
  return filter(items)   # drop history items, dedup, business rules → ranked list
```

## Connections

- Route to: [[Large Recommendation Models (LRM)]] — the "LLM-style end-to-end scaling" branch, contrasted with stage-wise cascade scaling
- Instance of: [[Generative Recommendation]] / [[Generative Retrieval]] — generate the item identifier instead of scoring candidates
- Item tokenization via: [[Semantic IDs]] built with [[RQ-VAE]] / [[Residual Quantization]] (the [[TIGER]] lineage)
- Decoding grounded by: [[Trie-Constrained Decoding]] / [[Constrained Decoding]] + [[Beam Search]]
- Fine-tuned with: [[GRPO]] / [[DPO]] reward optimization on top of next-token cross-entropy
- Builds on: [[Next-Item Prediction]] from [[Sequential Recommendation]] ([[SASRec]], [[BERT4Rec]], [[GRU4Rec]])
- Related LRM backbones: [[HSTU]] (decoder-only, industrial scale), [[Scaling Laws]] for recommendation

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
