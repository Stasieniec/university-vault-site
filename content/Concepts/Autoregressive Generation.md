---
type: concept
aliases: [Autoregressive Decoding]
course: [RecSys]
tags: [generative-rec, sequential-rec, llm, exam-topic]
status: complete
---

# Autoregressive Generation

## Definition

> [!definition] Autoregressive Generation
> **Autoregressive generation** factorizes the probability of an output sequence $\mathbf{z}$ given an input $\mathbf{x}$ into a **product of next-token conditionals**, decoding one token at a time, each conditioned on the input and all tokens already produced. In [[Generative Recommendation]], the output is a fixed-length **item identifier** $\mathbf{z}_i = (z_{i,1}, \ldots, z_{i,L})$ (e.g. a [[Semantic IDs|Semantic ID]]) and the input $\mathbf{x}$ is the user's interaction history. Instead of scoring a fixed candidate set, the model **generates the next item's id token by token** and maps it back to a real catalogue item.

## Intuition

> [!intuition] Recommendation as Next-Token Prediction
> User behaviour is *already* a sequence $i_1, i_2, \ldots, i_t \to i_{t+1}$, so [[Next-Item Prediction|next-item prediction]] is structurally identical to a language model's **next-token prediction**. The chain rule lets us write any joint distribution over a sequence as a left-to-right product of conditionals — so we never need a model over the whole catalogue at once, only a model of "what comes next given everything so far." Each emitted token narrows the item down: with [[Hierarchical Semantic IDs|hierarchical ids]] the first token picks a coarse region (e.g. genre), and later tokens refine toward one specific item. This is exactly why the whole LLM decoding toolkit ([[Beam Search]], sampling, [[Constrained Decoding]]) transfers to RecSys.

## Mathematical Formulation

> [!formula] Autoregressive Factorization of the Item Identifier
> $$p_\theta(\mathbf{z}_i \mid \mathbf{x}) = \prod_{\ell=1}^{L} p_\theta\big(z_{i,\ell} \mid \mathbf{x},\, z_{i,<\ell}\big)$$
>
> where:
> - $\mathbf{x} = (x_1, \ldots, x_t)$ — user interaction history (the conditioning context)
> - $\mathbf{z}_i = (z_{i,1}, \ldots, z_{i,L})$ — fixed-length identifier of item $i$ (atomic id is the special case $L=1$)
> - $z_{i,<\ell} = (z_{i,1}, \ldots, z_{i,\ell-1})$ — tokens already generated (the autoregressive dependence)
> - $L$ — identifier length; each position drawn from a codebook of size $K$, giving $K^L$ possible codes
> - $\theta$ — Transformer parameters ([[Transformers|encoder–decoder]] or decoder-only)

The identifier likelihood doubles as a **score** for the item, recovering the classical score-and-rank view:

> [!formula] Likelihood-as-Score
> $$s_\theta(\mathbf{x}, i) = \log p_\theta(\mathbf{z}_i \mid \mathbf{x}) = \sum_{\ell=1}^{L} \log p_\theta\big(z_{i,\ell} \mid \mathbf{x},\, z_{i,<\ell}\big)$$
> Higher likelihood = more compatible with the history. Recommendations are produced by **decoding valid ids**, not by scoring a fixed candidate set.

**Training** uses standard **next-token cross-entropy** with teacher forcing (the target id is known, so each position conditions on the *true* prefix):

> [!formula] Next-Token Cross-Entropy Loss
> $$\mathcal{L} = -\sum_{\ell=1}^{L} \log p_\theta\big(z_\ell \mid \text{history},\, z_{<\ell}\big)$$
> Identical to language-model training in every respect except that tokens are item codes from a small learned codebook ($K \sim 256$–$4096$) rather than a large subword vocabulary.

## Key Properties / Variants

- **Two-direction inference vs. teacher-forced training.** At training, the full target id is available, so all $L$ positions are scored in parallel against the true prefix. At inference, token $\ell$ depends on the model's *own* previous predictions $z_{<\ell}$, making decoding inherently **sequential** ($L$ steps per item).
- **Decoding strategies:** greedy (top-1 each step) yields a single item; **[[Beam Search]]** keeps $B$ partial candidates per step to return a ranked top-$B$ list after $L$ steps; sampling / temperature injects diversity.
- **Validity problem:** the code space has $K^L$ sequences but the catalogue uses only a tiny fraction, so most decoded sequences are not real items. Fixed by **[[Trie-Constrained Decoding]]** (logit mask restricting each step to tokens on a valid trie path, then renormalize) and/or **[[Group Relative Policy Optimization|GRPO]]** reward shaping that rewards valid ids.
- **Architecture choices** (both produce the code one token at a time): encoder–decoder (T5-style; TIGER, LETTER) reads the history fully then writes the id; decoder-only (GPT-style; HSTU, OneRec) treats `[history || target]` as one continuous stream and predicts the next token throughout.
- **Beyond CE:** preference / RL fine-tuning ([[Group Relative Policy Optimization|GRPO]], [[Direct Preference Optimization (DPO)|DPO]]) optimizes list-level reward (validity, [[Diversity|diversity]], freshness) that token-level CE cannot express.
- **Decoding pathologies:** amplification bias (popular prefixes dominate the beam, long-tail pruned early), homogeneity (top-$B$ share a prefix → near-duplicate list), local optima (greedy first token locks a region), and latency ($L$ sequential steps + trie lookup per recommendation).

```pseudo
Algorithm: Constrained Autoregressive Decoding (one ranked list)
─────────────────────────────────────────────────────────────────
Input: history x, beam size B, id length L, valid-id trie T
Initialize beams ← { (prefix=<BOS>, score=0) }
for ℓ = 1 .. L:
  candidates ← {}
  for each beam (prefix, score) in beams:
    allowed ← T.children(prefix)            # validity mask
    for token z in allowed:
      lp ← log p_θ(z | x, prefix)           # renormalized over allowed
      candidates ← candidates ∪ { (prefix·z, score + lp) }
  beams ← top-B candidates by score          # prune the rest
return decode-ids(beams)                      # B complete, valid item ids
```

## Connections

- Core mechanism of: [[Generative Recommendation]], [[Generative Retrieval]]
- Output tokens are: [[Semantic IDs]] / [[Atomic Item IDs]] (special case $L=1$)
- Implemented by: [[Transformers]] (encoder–decoder or decoder-only), [[Self-Attention]]
- Inference uses: [[Beam Search]], [[Trie-Constrained Decoding]], [[Constrained Decoding]]
- Trained by: next-token cross-entropy + Teacher Forcing, optionally [[Group Relative Policy Optimization]] / [[Direct Preference Optimization (DPO)]]
- Same idea applied to documents: [[Differentiable Search Index|DSI]], [[Generative Retrieval]]
- Reframes: [[Next-Item Prediction]] as next-token prediction
- Tension with: [[Diversity]] (homogeneous beams), [[Cold Start]] (cold ids have near-zero generation probability)

## Appears In

- [[RS-L04 - Generative Recommendation]]
