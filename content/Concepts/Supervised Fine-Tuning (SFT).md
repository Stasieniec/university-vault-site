---
type: concept
aliases: [SFT]
course: [RecSys]
tags: [generative-rec, llm, exam-topic]
status: complete
---

# Supervised Fine-Tuning (SFT)

## Definition

> [!definition] Supervised Fine-Tuning (SFT)
> **Supervised Fine-Tuning (SFT)** adapts a pretrained [[LLM]] to the recommendation task by training it on **(input, target)** pairs with the standard next-token (cross-entropy) objective. For [[Generative Recommendation|generative recommendation]], the input is a verbalized user profile / interaction history (or [[Semantic ID|semantic-ID]] token stream) and the target is the **ground-truth next item** — represented as a title, an atomic ID, or a sequence of codebook tokens. SFT learns **only from positive examples**: it maximizes the likelihood of observed (clicked / liked) items and never explicitly sees negatives.

## Intuition

> [!intuition] Teacher-forcing on what the user actually did
> SFT treats recommendation as supervised sequence modeling. You show the model "history → the item the user really chose next" and push up the probability of that item, token by token (teacher forcing). It is the simplest of the four training objectives on Slide 15 (SFT / SSL / [[RL]] / [[DPO]]): just maximum likelihood, no reward model, no contrastive negatives.
>
> The price of that simplicity is that the loss only ever says "make the positive more likely." It never says "and make this *other* item less likely." Ranking is fundamentally about *ordering* items, so a pure likelihood objective gives no **explicit margin** between a good item and a bad one — the model can assign high probability to the target yet still rank plausible distractors just as high.

## Mathematical Formulation

SFT minimizes the negative log-likelihood of the target item sequence $y = (y_1, \dots, y_T)$ given the input prompt $x$, autoregressively:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{(x, y)\in\mathcal{D}} \sum_{t=1}^{T} \log p_\theta\big(y_t \mid y_{<t},\, x\big)$$

where:
- $\theta$ — parameters being tuned (full model, or only adapter weights under [[LoRA]] / [[Parameter-Efficient Fine-Tuning|PEFT]])
- $\mathcal{D}$ — training set of **positive** pairs only: $x$ = verbalized history / instruction, $y$ = the observed next item
- $y_t$ — the $t$-th target token; an item is one token (atomic ID), several text tokens (title), or a fixed-length code (e.g. a 3-level semantic ID `<a><b><c>`)
- $y_{<t}$ — previously generated target tokens (teacher-forced to ground truth during training)
- $p_\theta(\cdot)$ — the LLM's softmax over its vocabulary; the per-token term is exactly cross-entropy against the one-hot ground truth

Contrast with the pairwise/preference objectives that *do* encode a margin — [[BPR]] uses $-\log\sigma(\hat{s}_{u,i^+}-\hat{s}_{u,i^-})$ over a sampled negative $i^-$, and [[DPO]] optimizes a (preferred, rejected) pair directly. SFT has no such second term, hence "no explicit ranking margin."

## Key Properties / Variants

- **Positives only.** The objective sums over observed interactions; there is no negative sampling and no explicit negative term in the loss. This is the lecture's headline limitation (Slide 15): "Learn only positive samples. Simple, but no explicit negatives → the ranking margin is hard to learn."
- **Standard next-token loss.** Identical machinery to LLM pretraining/instruction-tuning — just a different dataset. Makes it cheap to implement and stable to train relative to [[RL]].
- **Usually parameter-efficient.** In practice SFT for rec is done with [[LoRA]] / [[PEFT]] adapters over a frozen backbone, not full fine-tuning, to keep cost low (Text-Prompting paradigm, Slide 9).
- **Works across all item-tokenization levels.** The target $y$ can be an atomic ID (P5), an item title (text-based), or a learned [[Semantic ID]] from [[RQ-VAE]] (TIGER-style); SFT is agnostic to how the item is tokenized.
- **Representative methods (course):** P5 (multi-task seq2seq rec via prompts), LGIR. Listed as the SFT row in Hou et al. (2025) §4.1.3, Table 5.
- **Position in the objective taxonomy:** one of four ways to shape the next-item-prediction loss — SFT (this note), SSL (contrastive, e.g. FELLAS/EasyRec), [[RL]] (reward-driven, can encode explicit negatives & non-differentiable metrics, but unstable), and [[DPO]] (preference pairs, stable, no reward model). SFT is the baseline the others improve upon.
- **Common pipeline:** SFT first to teach the format / grounding, then optionally [[DPO]] or [[RL]] on top to inject the ranking margin SFT lacks — mirroring the LLM post-training recipe (SFT → preference optimization).

```pseudo
Algorithm: SFT for Generative Recommendation
─────────────────────────────────────────────
Input: pretrained LLM p_θ, positive interaction data D
       item tokenizer τ (atomic ID | title | semantic ID)

Build dataset:
  for each user interaction (history h, next-item i*) in logs:
    x ← verbalize(h)            # prompt / instruction + history
    y ← τ(i*)                   # target item as token sequence
    add (x, y) to D

Train (teacher forcing):
  repeat until converged:
    sample minibatch {(x, y)} ⊂ D
    L ← 0
    for each (x, y):
      for t = 1 .. |y|:
        L ← L − log p_θ(y_t | y_<t, x)   # cross-entropy, ground-truth y_<t
    θ ← θ − η ∇_θ L              # (∇ only over LoRA/PEFT weights if used)

Inference:
  decode ŷ ~ p_θ(· | x) autoregressively (often beam search +
  constrained/trie decoding so ŷ maps to a REAL item)
```

> [!warning] No explicit negatives, so the ranking margin is hard to learn
> Because the loss only pushes up positive likelihood and never pushes down competitors, SFT does not directly optimize the [[Top-K Ranking|Top-K]] ordering objective that recommendation actually cares about. A model can perfectly fit the targets yet still mis-rank because no margin separates positives from plausible negatives. This is precisely why the lecture pairs SFT with negative-aware objectives ([[BPR]]-style negatives, [[RL]], [[DPO]]).

> [!warning] Generated items may not exist
> SFT teaches *what* to generate but not the constraint that the output must be a valid catalog item. At inference the decoded token sequence can correspond to a non-existent item (hallucination). This motivates **generation grounding** — constrained / trie-based decoding over the valid item vocabulary so every generated sequence maps to a real item.

## Connections

- One of four training objectives for [[LLM-based Recommendation|LLM-based GR]], alongside SSL, [[RL]], and [[DPO]] (Slide 15)
- Fixed by / complemented by: [[DPO]] (preference pairs add the margin SFT lacks), [[RL]] (reward-driven, explicit negatives), [[BPR]] (pairwise ranking baseline with explicit negatives)
- Typically implemented with: [[LoRA]] / [[Parameter-Efficient Fine-Tuning]]
- Mechanism shared with: [[Reinforcement Learning from Human Feedback|RLHF]] post-training recipe (SFT stage), [[Autoregressive Generation]]
- Operates over targets produced by: [[Item Tokenization]] / [[Semantic ID]] (atomic ID, text, [[RQ-VAE]] codes)
- Task it instantiates: [[Next-Item Prediction]] for [[Generative Recommendation]]
- Used by alignment paradigm: [[LLM-as-Recommender]] (Text Prompting)

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
