---
type: concept
aliases: [LLM4Rec, LLM-as-Recommender, LLM-as-RS, LLM-based Recommenders, LLM-based Generative Recommendation, LLM-as-Enhancer]
course: [RecSys]
tags: [llm, generative-rec, collaborative-filtering, exam-topic]
status: complete
---

# LLM-based Recommendation

## Definition

> [!definition] LLM-based Recommendation
> **LLM-based recommendation** (LLM4Rec) treats the recommendation problem as a **language task**: users, items, and interaction histories are translated into something a pretrained [[Large Language Models (LLM)|LLM]] can read, and the model **generates** the recommendation (an item title, an item ID, or a textual answer) rather than scoring a fixed candidate pool. It is one of the two routes to [[Generative Recommendation]] — the other being native [[Large Recommendation Models (LRM)|LRMs]] — and it borrows scaling, world knowledge, and reasoning from language pre-training.
>
> The defining contrast with classical [[Sequential Recommendation]] is the output space: a discriminative model learns a scoring function $f(\text{user}, \text{item})$ over a fixed catalogue; an LLM-based recommender **decodes the target directly**, token by token.

## Intuition

> [!intuition] Why "generate" instead of "score"?
> A classical recommender (e.g. [[SASRec]]) only knows what it saw in click logs — it has no idea what *Inception* is about. A pretrained LLM already encodes enormous **world knowledge** and **natural-language understanding**, so it can reason about a brand-new item from its description alone. This is what makes LLM4Rec strong on **cold-start** and **cross-domain** transfer, where a discriminative model starves for signal.
>
> The catch: LLM pre-training never saw **click/interaction signal**, the **Top-K ranking objective**, **long-tail coverage** incentives, or **exposure-bias** awareness. So a vanilla prompted LLM often loses to a small specialized rec model. Closing that gap — injecting collaborative signal and grounding the output in real items — is the whole technical content of the field.

## Mathematical Formulation

The dominant formulation casts next-item prediction as **autoregressive decoding** of an item identifier $\mathbf{z}_i = (z_{i,1}, \ldots, z_{i,L})$ conditioned on the user history $\mathbf{x} = (x_1, \ldots, x_t)$:

$$p_\theta(\mathbf{z}_i \mid \mathbf{x}) = \prod_{\ell=1}^{L} p_\theta\!\left(z_{i,\ell} \mid \mathbf{x},\, z_{i,<\ell}\right)$$

where:
- $\mathbf{x}$ — the user's interaction history, verbalized as a prompt or as a sequence of item tokens
- $\mathbf{z}_i$ — the target item's identifier (a textual title, an atomic token, or an $L$-token [[Semantic ID]])
- $z_{i,<\ell}$ — identifier tokens already generated (teacher-forced at training, beam-searched at inference)
- $L$ — identifier length ($L=1$ for an atomic item ID; $L\sim 3$–$4$ for codebook semantic IDs)

The likelihood of a valid identifier doubles as a **score** for the item, $s_\theta(\mathbf{x}, i) = \log p_\theta(\mathbf{z}_i \mid \mathbf{x})$, so ranking is recovered by decoding the highest-probability valid identifiers.

**Training objective (next-token cross-entropy / SFT).** The model is fit exactly like a language model:

$$\mathcal{L} = -\sum_{\ell=1}^{L} \log p_\theta\!\left(z_\ell \mid \mathbf{x},\, z_{<\ell}\right)$$

where the tokens are item codes drawn from a small learned codebook ($K \sim 256$–$4096$) instead of a 50K BPE vocabulary. This is [[Supervised Fine-Tuning (SFT)|SFT]] on positive sequences; because it only rewards copying the exact next click, it has no explicit negatives and can be augmented with SSL, RL, or [[Direct Preference Optimization (DPO)|DPO]].

## Key Properties / Variants

**Three research lines (Hou et al. 2025 §4.1):**
- **(1) LLM directly as RS** — no fine-tuning; drive recommendation through prompt design and [[In-Context Learning|ICL]]. Two flavors:
  - **[[LLM-as-Enhancer|LLM-as-Enhancer]]** — rewrite user/item profiles and history into rich natural-language features, then feed them into a CF / sequential / re-ranking model.
  - **[[LLM-as-Recommender|LLM-as-Recommender]]** — a templated prompt; the LLM outputs item titles or IDs directly (e.g. Chat-REC). Zero training cost, but prompt-sensitive, carries no collaborative signal, and may **hallucinate non-existent items**.
- **(2) Align an LLM to the recommendation task** — fine-tune so the model carries collaborative signal and item structure (see paradigms below).
- **(3) Training objectives and inference** — choose [[Supervised Fine-Tuning (SFT)|SFT]] / SSL ([[Contrastive Learning]]) / [[Reinforcement Learning|RL]] / [[Direct Preference Optimization (DPO)|DPO]]; ensure generated items actually exist.

**Three alignment paradigms** (how the user/item profile is structured into the input):
- **① Text Prompting** — pure natural language; the entire problem (task instruction + temporal history) is expressed in text, fine-tuned parameter-efficiently with [[Low-Rank Adaptation|LoRA]] (e.g. TALLRec, LlamaRec). No collaborative signal → weak when inter-item dependencies dominate.
- **② Inject Collaborative Signal** — language + CF info together (e.g. CoLLM, CoRAL). Sub-strategies: project a learned CF embedding into the LLM's token space; distill CF into a short text **summary**; or **verbalize** CF as sentences ("users who liked A also liked B"). The deeper issue: dense CF embeddings are **not directly readable** by an LLM — a fundamental gap between collaborative and language semantics.
- **③ [[Item Tokenization]]** — items as learned discrete tokens ([[Semantic IDs]]), so the LLM can *generate* them. The identifier ladder L1→L5: atomic ID → item text → [[RQ-VAE]] codebook semantic ID → semantic ID + CF (LETTER) → adaptive. This is the bridge to [[Generative Recommendation]] proper (TIGER, P5, LC-Rec).

**Grounding the output (validity).** Most $K^L$ codes do not map to a real item. Two complementary fixes:
- **[[Trie-Constrained Decoding|Trie-constrained decoding]]** — store all valid catalogue IDs in a [[Trie|trie]]; at each step a logit mask allows only tokens on a valid path. Guarantees validity but the trie must track catalogue changes.
- **Reward validity** — make "is this a real item?" part of a [[Group Relative Policy Optimization|GRPO]] reward; valid IDs are pushed up. Only *likely* valid, but needs no live trie. Often combined.

**Two formulations** (Liu et al. ICLR 2026 submission):

| | Quantization tokenizer? | Input to RS | Backbone | Output |
|---|---|---|---|---|
| **SID-based GR** | Yes | item [[Semantic IDs]] | Transformer | item semantic IDs |
| **LLM-as-RS** | No | text descriptions | LLM (+[[Low-Rank Adaptation|LoRA]]) | item text titles |

**Pseudo-code — LLM-as-Recommender at inference:**

```pseudo
Algorithm: LLM-based Recommendation (generate + ground)
────────────────────────────────────────────────────────
Input: user history x = (x_1, ..., x_t), beam size B
1. Build prompt:  verbalize history (titles or SID tokens)
                  + task instruction
2. Constrained beam search over the LLM:
     keep B partial identifiers; at each of L steps,
     mask logits to tokens allowed by the catalogue trie
3. Emit B complete, valid identifiers z^(1..B)
4. Map each z to its catalogue item (id-to-item lookup)
5. Post-process: drop items already in history,
                 dedup, apply business rules
6. Return ranked list
```

**Pros:** zero/low training cost; strong semantics and world knowledge; good cold-start and cross-domain generalization; reasoning and instruction-following; rides the scaling law.
**Cons:** prompt-sensitive; no native collaborative signal (must be injected); hallucination unless grounded; expensive autoregressive decoding ($L$ steps + trie lookup) vs. a real-time latency budget; offline metrics ([[NDCG]], [[Recall]]) under-credit the novelty it produces.

## Connections

- Route within: [[Generative Recommendation]] (the other route is native [[Large Recommendation Models (LRM)|LRMs]])
- Backbone: [[Large Language Models (LLM)]], [[Transformer Model]], [[Self-Attention]]
- Contrasts with: discriminative [[Sequential Recommendation]] ([[SASRec]], [[BERT4Rec]], [[GRU4Rec]]) and [[Matrix Factorization]] / [[Collaborative Filtering]]
- Output grounding: [[Item Tokenization]], [[Semantic IDs]], [[RQ-VAE]], [[Trie-Constrained Decoding]]
- Adaptation: [[Parameter-Efficient Fine-Tuning]] / [[Low-Rank Adaptation|LoRA]], [[In-Context Learning]]
- Objectives: [[Supervised Fine-Tuning (SFT)]], [[Contrastive Learning]], [[Direct Preference Optimization (DPO)]], [[Group Relative Policy Optimization|GRPO]], [[Reinforcement Learning from Human Feedback]]
- Motivated by: [[Cold Start Problem]], [[Cross-Domain Recommendation]], [[Scaling Laws]]
- Parallel idea in IR: [[Generative Retrieval]] / [[Differentiable Search Index]]

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03a - Sequential Recommendation Models]]
- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
