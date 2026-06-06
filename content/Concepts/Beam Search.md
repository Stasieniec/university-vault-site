---
type: concept
aliases: []
course: [RecSys]
tags: [generative-rec, sequential-rec, exam-topic]
status: complete
---

# Beam Search

## Definition

> [!definition] Beam Search
> **Beam search** is an approximate decoding algorithm for [[Autoregressive Generation|autoregressive]] sequence models that keeps the $B$ highest-scoring **partial** sequences (the *beam*) at every generation step, rather than committing to the single best token like greedy decoding. In [[Generative Recommendation|GenRec]], the model decodes an item's [[Semantic IDs|Semantic ID]] $\mathbf{z} = (z_1, \ldots, z_L)$ one code at a time; beam search expands all $B$ live prefixes by every candidate next code, scores the resulting continuations by cumulative log-probability, and prunes back to the top $B$. After $L$ steps it emits $B$ complete SIDs, which become a **ranked top-$B$ list** of recommended items.

## Intuition

> [!intuition] A Ranked List, Not Just One Answer
> Greedy decoding keeps only the top-1 next code at each step. That is fine if you want *the* single next item, but recommendation needs a *ranked list*. Beam search is the cheap middle ground between greedy (track 1 path) and an exhaustive search over all $K^L$ possible code sequences (intractable). By carrying $B$ hypotheses forward, an item whose first code was only the 2nd- or 3rd-best can still survive to the end and end up ranked highly — something greedy can never recover. The width $B$ trades compute and list size against the risk of pruning a good item too early.

## Mathematical Formulation

The generator factorizes the SID likelihood autoregressively (the same chain rule as a language model):

$$p_\theta(\mathbf{z} \mid h) = p_\theta(z_1 \mid h) \prod_{\ell=2}^{L} p_\theta(z_\ell \mid h, z_{<\ell})$$

A partial hypothesis $z_{1:\ell}$ is scored by its cumulative log-probability, and at each step the beam $\mathcal{B}_\ell$ is the arg-top-$B$ over all one-code extensions of the previous beam:

$$s(z_{1:\ell}) = \sum_{k=1}^{\ell} \log p_\theta(z_k \mid h, z_{<k}), \qquad \mathcal{B}_\ell = \operatorname*{arg\,top\text{-}B}_{\substack{z_{1:\ell-1}\in\mathcal{B}_{\ell-1} \\ z_\ell \in \mathcal{V}}} \; s(z_{1:\ell})$$

where:
- $h$ — encoded user interaction history (the conditioning context)
- $\mathbf{z} = (z_1, \ldots, z_L)$ — the item identifier (Semantic ID) being decoded
- $L$ — identifier length (number of autoregressive steps per item)
- $B$ — beam width (number of partial hypotheses kept per step; also the size of the final list)
- $\mathcal{V}$ — codebook vocabulary of allowed next codes (size $K$); in constrained decoding this is restricted by the trie
- $s(z_{1:\ell})$ — cumulative log-probability score of a partial sequence
- arg-top-$B$ — selection of the $B$ highest-scoring continuations; the rest are **pruned**

Final output: the $B$ completed sequences in $\mathcal{B}_L$, sorted by $s(\cdot)$, give the ranked recommendation list (e.g., $(12,48,5),\ (12,48,7),\ (7,18,3) \to B$ ranked items).

## Key Properties / Variants

- **Per-step cost.** Each item costs $L$ sequential decoder steps; at each step all $B$ beams are expanded over $K$ codes and re-pruned. Inference cost is roughly $O(B \cdot L \cdot K)$ plus a trie lookup per step — at catalogue scale this **dominates latency** (real systems target <50 ms).
- **Length normalization.** Raw cumulative log-prob favors shorter sequences; when comparing variable-length hypotheses, scores are typically divided by length. In GenRec all SIDs are fixed length $L$, so this is usually unnecessary.
- **Constrained beam search.** Most of the $K^L$ code combinations are not real items (**the validity problem**). Beam search is paired with [[Trie-Constrained Decoding]]: a trie stores all valid catalogue SIDs, and at each step a **logit mask** zeros out codes that do not extend a valid prefix, renormalizing the distribution over allowed codes only. Every emitted sequence is then guaranteed to be a real item. See also [[Constrained Decoding]].
- **Greedy = $B=1$.** Greedy decoding is beam search with width one. Atomic-ID generation is the degenerate case $L=1$, so the whole item is decoded in a single step and "beam search" reduces to taking the top-$B$ codes directly.
- **Decoding pathologies (hit GenRec hard).**
  - *Amplification / popularity bias* — popular SID prefixes (e.g. $(12,48,\cdot)$) dominate the beam, so long-tail items get pruned early (see [[Popularity Bias]], [[Long Tail]]).
  - *Homogeneity* — surviving beams share long prefixes, so the top-$B$ items are near-duplicates (e.g. five similar action films); hurts [[Diversity]] and [[Novelty]].
  - *Local optima* — a greedy-ish first-code choice locks in a region of SID space; a better item under a different prefix is unreachable.
- **Diversity-aware variants.** *Diverse beam search* splits hypotheses into groups and penalizes a group for repeating earlier choices; *sampling / temperature* injects randomness so a less-likely opening code can survive; post-hoc re-ranking with [[Maximal Marginal Relevance (MMR)|MMR]] greedily prefers items unlike those already chosen. RL fine-tuning (GRPO) or tokenizer design (LETTER) can also attack homogeneity upstream.

```pseudo
Algorithm: Constrained Beam Search over Semantic IDs
─────────────────────────────────────────────────────
Input: history encoding h, beam width B, id length L,
       validity trie T over catalogue SIDs
beam ← { (prefix=∅, score=0) }            # one empty hypothesis
for ℓ = 1 .. L:
    cand ← ∅
    for (prefix, score) in beam:
        allowed ← T.next_tokens(prefix)    # trie mask: valid codes only
        logits  ← model(h, prefix)
        for z in allowed:
            cand ← cand ∪ { (prefix·z, score + log softmax(logits)[z]) }
    beam ← top-B of cand by score          # prune to width B
return sort(beam by score)                 # B complete SIDs = ranked list
                                           # then filter seen / dedup / business rules
```

## Connections

- Operates on: [[Autoregressive Generation]] / [[Autoregressive Decoding]] of a [[Generative Recommendation|generative recommender]]
- Decodes: [[Semantic IDs]] (or [[Atomic Item IDs]] in the $L=1$ case)
- Paired with: [[Trie-Constrained Decoding]], [[Constrained Decoding]] to guarantee valid items
- Special case: greedy decoding ($B=1$); contrasted with full search over $K^L$ codes
- Suffers from: [[Popularity Bias]], [[Long Tail]] pruning, low [[Diversity]] / [[Novelty]] (homogeneous beams)
- Diversity remedies: [[Maximal Marginal Relevance (MMR)]], diverse beam search, sampling; upstream via GRPO RL or LETTER tokenizer
- Produces: a [[Top-K Recommendation|top-K]] ranked list for the [[Sequential Recommendation|next-item]] task

## Appears In

- [[RS-L04 - Generative Recommendation]]
