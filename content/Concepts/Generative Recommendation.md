---
type: concept
aliases: [GR, Generative Recommender]
course: [RecSys]
tags: [generative-rec, sequential-rec, llm, exam-topic]
status: complete
---

# Generative Recommendation

## Definition

> [!definition] Generative Recommendation
> **Generative Recommendation (GR)** reframes recommendation from **scoring a fixed candidate set** to **generating the target item directly**, token by token. Instead of computing a score $s(\text{user}, \text{item})$ for every catalogue item and ranking, a sequence model **decodes an item identifier** $\mathbf{z}_i = (z_{i,1}, \ldots, z_{i,L})$ autoregressively from the user history, then maps that identifier back to a real catalogue item. The classical pipeline is "encode history → score catalogue → rank"; GR is "encode history → **decode identifier** → look up item." This lecture focuses on **semantic-ID-based GR** (items are tokenized into a few shared codebook tokens, e.g. TIGER), as distinct from LLM-as-RS (which generates item text titles), diffusion embedding-denoising (DDRM), and content generation (DiFashion).

## Intuition

> [!intuition] Recommendation as Next-Token Prediction
> User behaviour is *already* a sequence: $i_1, i_2, \ldots, i_t \to i_{t+1}$. Predicting the next item is exactly analogous to a language model predicting the next token. So rather than maintaining a softmax over millions of items (which grows linearly with the catalogue and treats `item_3487` as a meaningless atomic symbol), the model **generates** the next item's identifier one piece at a time.
>
> The payoff comes from how items are tokenized. With [[Semantic IDs]], a 4-position code with a 256-entry codebook spans $256^4 \approx 4.3 \times 10^9$ possible items using only $4 \times 256$ token embeddings — capacity is decoupled from vocabulary size. Related items share code **prefixes**, so generating a coarse prefix and refining it token-by-token gives a built-in category hierarchy, cheaper cold-start (a new item gets a valid code from its content, no new embedding row), and the ability to retrieve and rank in **one** model instead of a multi-stage cascade.

## Mathematical Formulation

> [!formula] Autoregressive Item-Identifier Generation
> Each catalogue item $i \in \mathcal{I}$ has a fixed-length identifier $\mathbf{z}_i = (z_{i,1}, \ldots, z_{i,L})$. Given user history $\mathbf{x} = (x_1, \ldots, x_t)$, the model decodes the next item's identifier one token at a time, each token conditioned on the history and the tokens already produced:
> $$p_\theta(\mathbf{z}_i \mid \mathbf{x}) = \prod_{\ell=1}^{L} p_\theta\big(z_{i,\ell} \mid \mathbf{x},\, z_{i,<\ell}\big)$$
> The identifier's log-likelihood is the item's **score**, so ranking falls out of generation:
> $$s_\theta(\mathbf{x}, i) = \log p_\theta(\mathbf{z}_i \mid \mathbf{x})$$
>
> where:
> - $L$ — identifier length (number of code positions per item); $L = 1$ with codebook $=$ catalogue recovers [[Atomic IDs]]
> - $z_{i,\ell}$ — the $\ell$-th code token of item $i$, drawn from a small learned codebook of size $K \sim 256$–$4096$
> - $z_{i,<\ell}$ — the code tokens decoded so far (teacher-forced at training time)
> - $\theta$ — encoder–decoder or decoder-only Transformer parameters

> [!formula] Training Loss — Next-Token Cross-Entropy
> Identical machinery to a language model; only the vocabulary differs (item codes, not BPE subwords):
> $$\mathcal{L} = -\sum_{\ell=1}^{L} \log p_\theta\big(z_\ell \mid \text{history},\, z_{<\ell}\big)$$
>
> where the target identifier is known during training (teacher forcing), and the loss is averaged over all $L$ positions and all items in the batch. Optionally followed by RL/preference fine-tuning ([[GRPO]], [[DPO]]) to reward **validity** (real items), **listwise quality**, and goals like diversity/freshness that next-token CE never sees.

## Key Properties / Variants

- **Two formulations:** *SID-based GR* (this lecture's focus) — items → semantic IDs via a frozen quantization tokenizer, then a Transformer generates those IDs; vs *LLM-as-RS* — a frozen LLM + trained [[LoRA]] generates item text titles from text descriptions.
- **Item tokenization is a modelling choice, not preprocessing.** Three designs: atomic IDs (one token/item — simple but vocab explodes, no structure, strict cold-start), textual IDs (full description — meaningful but very long, hard to constrain), and **semantic IDs** (short, reusable, structured — the middle ground). A good item ID is *compact, grounded, learnable, structured*.
- **Semantic ID construction (TIGER, RQ-VAE):** offline, before generator training. Item text/metadata → Sentence-T5 embedding → residual quantization over $L$ codebooks (pick nearest codeword, subtract, pass residual on) → code tuple $(z_1, \ldots, z_L)$; coarse-to-fine. A collision-handling token is appended so each tuple maps to exactly one item. The RQ-VAE is trained by reconstruction: $\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{rqvae}}$, $\mathcal{L}_{\text{recon}} = \lVert \mathbf{x}_i - \hat{\mathbf{x}}_i \rVert_2^2$. Construction families: residual quantization (RQ-VAE, RK-Means), product quantization (VQ-Rec), hierarchical clustering (P5-CID), LM/textual IDs. Behaviour-aware tokenizers (CoST adds a contrastive loss; LETTER adds semantic + collaborative + diversity regularizers; ActionPiece is context-aware) fold collaborative signal into the codes.
- **Architecture:** encoder–decoder ("read fully, then write" — T5-style; TIGER, OneRec) or decoder-only ("one continuous stream" `[history || target SID]` — GPT-style; HSTU, scales to long histories).
- **Decoding is part of the model.** Greedy gives top-1; beam search keeps $B$ partial candidates and emits $B$ ranked SIDs after $L$ steps.
- **The validity problem:** most of the $K^L$ codes are not real items. **Trie-constrained decoding** stores all valid catalogue SIDs in a trie and masks logits to only on-path tokens, guaranteeing every output is a real item (but the trie must stay synced with the catalogue). A complementary fix rewards validity inside [[GRPO]] instead of hard-masking; often combined.
- **Decoding pathologies:** popularity-prefix amplification bias, homogeneity (top-$B$ share long prefixes → look-alike lists), local optima from greedy first-token choice, and inference cost ($L$ sequential steps + trie lookup, hard under <50 ms budgets). Mitigations: temperature/sampling, diverse beam search, [[MMR]] re-ranking, RL diversity rewards, tokenizer-level fixes (LETTER), speculative/parallel decoding.

```pseudo
Algorithm: Generative Recommendation (SID-based, inference)
────────────────────────────────────────────────────────────
Offline (once):
  for each item i in catalogue I:
    e_i  ← ContentEncoder(text/metadata of i)      # e.g. Sentence-T5
    z_i  ← RQ-VAE(e_i)  = (z_1,...,z_L)             # frozen tokenizer
    append collision token if z_i not unique
  build TRIE over all valid SIDs {z_i}

Serving (per user history x = i_1,...,i_t):
  SID_history ← lookup z_{i_1},...,z_{i_t}          # flatten to L*t tokens
  beams ← { empty }                                 # beam size B
  for step ℓ = 1..L:
    for each partial SID b in beams:
      allowed ← TRIE.next_tokens(b)                 # validity mask
      logits  ← decoder(x, b);  mask out  ∉ allowed
      expand b by top tokens; rescore by Σ log p
    beams ← top-B partial SIDs by cumulative log-prob
  cands ← B complete SIDs  →  map each to its item
  filter (drop items already in history; dedup; business rules)
  return ranked list
```

## Connections

- Shift from: [[Sequential Recommendation]] models ([[SASRec]], [[BERT4Rec]], [[GRU4Rec]], [[FPMC]]) that still **score** [[Atomic IDs]] directly
- Item representation: [[Semantic IDs]] via [[RQ-VAE]] / [[Residual Quantization]] / [[Product Quantization]]; alternative [[Atomic IDs]]
- Decoding: [[Autoregressive Generation]], [[Beam Search]], [[Trie-Constrained Decoding]], [[Constrained Decoding]]
- Training: next-token cross-entropy (same as [[BERT4Rec]]'s MLM idea), then [[GRPO]] / [[DPO]] / [[Reinforcement Learning from Human Feedback]] fine-tuning
- Parallel idea in IR: [[Generative Retrieval]] / [[Differentiable Search Index]] (generate a document ID instead of scoring)
- Sibling route: [[LLM-based Recommendation]] (LLM-as-RS, prompting + [[LoRA]] alignment) and [[Large Recommendation Models (LRM)]] ([[HSTU]], scaling laws)
- Backbone: [[Transformer Model]], [[Self-Attention]]; cold-start handled via hybrid dense re-ranking
- Quality goals it can target: [[Diversity]], [[Novelty]], [[Cold Start]], [[Top-K Recommendation]]

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03a - Sequential Recommendation Models]]
- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
