---
type: concept
aliases: [Trie-Constrained Decoding, Constrained Generation, Trie]
course: [RecSys]
tags: [generative-rec, llm, exam-topic]
status: complete
---

# Constrained Decoding

## Definition

> [!definition] Constrained Decoding
> **Constrained decoding** restricts an [[Autoregressive Decoding|autoregressive]] generative recommender so that it can only emit token sequences corresponding to **real catalogue items**. In [[Generative Recommendation]], an item is represented by a [[Semantic ID]] $\mathbf{z}_i = (z_{i,1}, \ldots, z_{i,L})$, and the model decodes one code token at a time. The semantic-ID space has $K^L$ possible sequences (e.g. $\sim 10^9$) but only a tiny fraction (e.g. $\sim 10^7$) are valid items, so unconstrained generation overwhelmingly produces **invalid IDs**. The standard solution stores all valid catalogue IDs in a **trie** and, at each step, masks the logits of every token that does not continue a valid prefix path.

## Intuition

> [!intuition] You Cannot Hallucinate an Item
> In language generation any token sequence is a valid (if weird) sentence. In recommendation, most semantic-ID sequences point to **no item at all** — a code like $(5, 99, 13)$ may simply not exist in the catalogue. A free-decoding model would happily generate such phantom IDs, and the lookup would fail.
>
> The trie is a road map of every legal path. Sitting at prefix $(5, 23)$, the trie says "from here you may only go to $\{55, 18, 91\}$" — every other token is a dead end and is forbidden. Walking the trie from root to a leaf is therefore *guaranteed* to spell out a real item. Validity is enforced by construction, not hoped for.

## Mathematical Formulation

> [!formula] Logit Masking over Trie-Allowed Tokens
> At decoding step $\ell$, let $z_{<\ell}$ be the codes generated so far and let $\mathcal{A}(z_{<\ell}) \subseteq \{1, \ldots, K\}$ be the set of tokens that extend this prefix along *some* valid path in the trie. The model's distribution is masked and renormalized over only the allowed tokens:
> $$\tilde{p}_\theta(z_\ell = k \mid \mathbf{x}, z_{<\ell}) = \frac{\mathbb{1}[\,k \in \mathcal{A}(z_{<\ell})\,]\; \exp(o_k)}{\sum_{k' \in \mathcal{A}(z_{<\ell})} \exp(o_{k'})}$$
>
> where:
> - $o_k$ — the raw logit (pre-softmax score) the model assigns to code token $k$ at this step
> - $\mathcal{A}(z_{<\ell})$ — children of the trie node reached by following prefix $z_{<\ell}$ (the valid continuations)
> - $\mathbb{1}[\cdot]$ — indicator; tokens not in $\mathcal{A}$ get probability $0$ (logit set to $-\infty$ before softmax)
> - $\mathbf{x}$ — the user history conditioning the generation
>
> Equivalently the full autoregressive item probability is the masked product
> $$p_\theta(\mathbf{z}_i \mid \mathbf{x}) = \prod_{\ell=1}^{L} \tilde{p}_\theta(z_{i,\ell} \mid \mathbf{x}, z_{i,<\ell}),$$
> which is non-zero **only** when $\mathbf{z}_i$ is a complete root-to-leaf path. The mask is the only change versus standard next-token decoding; the model weights are untouched.

## Key Properties / Variants

- **Guaranteed validity:** every generated sequence is a real item — there is no post-hoc filtering of phantom IDs needed.
- **Implementation = logit mask:** the trie produces a per-step allowed-token set; tokens outside it have their logits set to $-\infty$, then softmax renormalizes. Cheap to apply on top of any decoder.
- **Composes with [[Beam Search]]:** the standard inference path is *constrained beam search* — maintain $B$ partial candidates, but only expand children that the trie permits, yielding $B$ valid ranked items after $L$ steps.
- **Catalogue-sync cost:** the trie must be rebuilt/updated whenever items are added or removed. In fast-moving systems this upkeep is non-trivial, and stale paths can keep surfacing dead items.
- **Per-request filtering for safety:** the trie is the only hard safety net in [[Generative Recommendation]] — it can be masked per user/locale to block NSFW, region-locked, recalled, or already-seen items before generation.
- **Reward-based alternative (complementary):** instead of masking, make "is this a real item?" part of an RL reward ([[GRPO]]): generate freely, reward valid IDs, penalize invalid ones. This needs no live trie but only makes validity *likely*, not guaranteed. Often the two are **combined** (trie at inference, validity reward in training).
- **Used by:** TIGER and most semantic-ID generative recommenders; the same idea drives generative IR document-ID decoding (GENRE, DSI).

```pseudo
Algorithm: Trie-Constrained Beam Search Decoding
─────────────────────────────────────────────────
Build trie T from all valid catalogue semantic IDs   (offline / on catalogue change)
Input: user history x, beam size B, ID length L

beams ← [ (prefix=[], score=0, node=root(T)) ]
for ℓ = 1 .. L:
    candidates ← []
    for (prefix, score, node) in beams:
        allowed ← children(node)              # A(z_<ℓ): valid next codes
        logits  ← model(x, prefix)            # raw scores over K codes
        mask logits[k] ← -∞   for all k ∉ allowed
        logp ← log_softmax(logits)            # renormalize over allowed only
        for k in allowed:
            candidates.append( (prefix+[k], score+logp[k], child(node,k)) )
    beams ← top-B candidates by score
return B complete semantic IDs  →  id-to-item lookup  →  ranked item list
```

## Connections

- Mechanism for: [[Generative Recommendation]], [[Generative Retrieval]] (item/document-ID decoding)
- Operates over: [[Semantic ID]]s produced by the (frozen) [[RQ-VAE]] tokenizer
- Combined with: [[Beam Search]] (constrained beam search), [[Autoregressive Decoding]]
- Trained objective it sits on top of: next-token cross-entropy; optionally [[GRPO]] / [[DPO]] for a validity reward
- Contrast: [[Atomic Item ID]]s make every token trivially valid (no trie needed), but lose cold-start generalization
- Downside it does *not* fix: decoding pathologies like popularity amplification and homogeneity (see [[Beam Search]], [[Maximal Marginal Relevance (MMR)]] for diversity)

## Appears In

- [[RS-L04 - Generative Recommendation]]
