---
type: concept
aliases: [ICL]
course: [RecSys]
tags: [llm, generative-rec, exam-topic]
status: complete
---

# In-Context Learning

## Definition

> [!definition] In-Context Learning (ICL)
> **In-context learning** is the ability of a pretrained [[Large Language Models (LLM)|LLM]] to adapt to a new task **purely from examples and instructions placed in its prompt**, with **no gradient updates** to its parameters. The "learning" happens at inference time inside the forward pass: the demonstrations $(x_i, y_i)$ in the context condition the model's next-token distribution so that, given a new query $x_{\text{query}}$, it generates the appropriate $y_{\text{query}}$.
> In RecSys, ICL is what makes **LLM-as-Recommender** possible: a templated prompt ("user watched A, B, C; recommend the next movie") elicits a recommendation from a **frozen** model, exploiting its world knowledge for cold-start and cross-domain transfer.

## Intuition

> [!intuition] Conditioning, not training
> A standard ML pipeline learns a task by *changing weights* via [[Gradient Descent]]. ICL does something different: the weights stay fixed, and the **prompt itself** carries the task. You "program" the model by *showing* it a few input→output pairs (few-shot), or just a description (zero-shot), and it pattern-matches the continuation.
> Think of the transformer's [[Self-Attention]] as a lookup over the context: the demonstrations act like a tiny, ephemeral training set the model attends to while predicting. Nothing is saved — change the prompt and the "learned" behavior changes immediately.
> For recommendation, the appeal is **zero training cost**: a vanilla LLM, never fine-tuned on click logs, can still recommend by reading a user's history in natural language. The catch is that this borrowed knowledge is *semantic*, not *collaborative*.

## Mathematical Formulation

ICL conditions the autoregressive language model on a prompt built from $k$ demonstrations plus the query, then decodes the answer. With a frozen parameter set $\theta$, the predicted output for a new input $x_{\text{query}}$ is sampled from:

$$
y_{\text{query}} \sim p_\theta\big(\, y \mid \underbrace{(x_1, y_1), (x_2, y_2), \dots, (x_k, y_k)}_{k \text{ demonstrations}},\; x_{\text{query}} \,\big)
$$

Because the model is autoregressive, this is realized token-by-token over the answer tokens $y = (y^{(1)}, \dots, y^{(T)})$:

$$
p_\theta(y \mid C) = \prod_{t=1}^{T} p_\theta\big(y^{(t)} \mid y^{(<t)},\, C\big), \qquad C = \big[(x_1,y_1),\dots,(x_k,y_k),\, x_{\text{query}}\big]
$$

where:
- $\theta$ — the LLM's pretrained parameters, **held fixed** (no $\nabla_\theta$ update; this is the defining property)
- $C$ — the full **context** (prompt): task instruction + $k$ in-context demonstrations + the query
- $k$ — number of demonstrations: $k=0$ is **zero-shot**, $k\geq 1$ is **few-shot**
- $(x_i, y_i)$ — demonstration pairs; in RecSys, e.g. $x_i$ = a user history, $y_i$ = the item that user picked
- $x_{\text{query}}$ — the new instance to solve (the target user's history)
- $y^{(<t)}$ — already-generated answer tokens, fed back in (autoregressive decoding)

Contrast with fine-tuning, which would instead solve $\theta^* = \arg\min_\theta \sum_i \mathcal{L}\big(y_i, f_\theta(x_i)\big)$ and **modify the weights**. ICL leaves $\theta$ untouched and pushes all task adaptation into $C$.

## Key Properties / Variants

- **No parameter updates.** Distinguishes ICL from [[Supervised Fine-Tuning (SFT)]], [[Low-Rank Adaptation|LoRA]] / [[Parameter-Efficient Fine-Tuning|PEFT]], and [[Reinforcement Learning from Human Feedback|RLHF]] — all of which change $\theta$. ICL changes only the prompt.
- **Zero-shot vs few-shot.** $k=0$: instruction only. $k$ small: a handful of demonstrations, which usually improves accuracy and output formatting but consumes context-window budget.
- **Emergent with scale.** ICL is a capability that strengthens sharply as model size and pretraining compute grow — a manifestation of the [[Scaling Laws|Scaling Law]]; small models show little ICL.
- **Prompt-sensitive.** Performance depends heavily on phrasing, demonstration choice, and ordering — the main fragility flagged for LLM-as-Recommender.
- **In RecSys taxonomy (Hou et al. 2025, §4.1.1):** ICL is the engine of *line (1) — use a pretrained LLM directly, no fine-tuning*, split into:
  - **LLM-as-Enhancer** — prompt the LLM to rewrite user/item profiles into rich text features, then feed those into a CF / sequential / reranking model.
  - **LLM-as-Recommender** — a templated prompt makes the LLM output item titles or IDs directly (e.g., Chat-REC's prompt-constructor → ChatGPT pipeline).

> [!warning] Why ICL alone is weak for recommendation
> A frozen, in-context-prompted LLM **never saw collaborative signal, the Top-K ranking objective, long-tail coverage, or exposure-bias** during pretraining. So it relies on *semantic / world knowledge* only and can be beaten by a specialized rec model. It is also prone to **hallucinating items that do not exist** in the catalog — motivating *generation grounding* via constrained decoding and item tokenization. These gaps are exactly what the alignment paradigms (text prompting fine-tune, CF injection, semantic IDs) try to close.

Conceptual ICL recommendation prompt (LLM-as-Recommender):

```pseudo
Procedure: ICL_Recommend(history, candidates?, theta)
──────────────────────────────────────────────────────
  # theta is FROZEN — no training step anywhere below
  instruction ← "Given the user's watch history, recommend the next item."
  demos       ← few-shot examples [(history_i, picked_item_i)]   # optional, k≥0
  prompt      ← instruction ⊕ demos ⊕ "History: " ⊕ history ⊕ "Next:"
  output      ← LLM_decode(prompt; theta)        # autoregressive generation
  # output is a title/ID string; must be grounded back to the real catalog
  return map_to_catalog(output)                  # guard against hallucinated items
```

## Connections

- Enables: [[LLM-as-Recommender]], [[LLM-as-Enhancer]] — the no-fine-tuning route of [[LLM-based Generative Recommendation]]
- Property of: [[Large Language Models (LLM)|LLM]]; realized via [[Self-Attention]] / [[Transformer Model|Transformers]] and [[Autoregressive Decoding]]
- Contrasts with (weight-updating adaptation): [[Supervised Fine-Tuning (SFT)]], [[Low-Rank Adaptation|LoRA]], [[Parameter-Efficient Fine-Tuning|PEFT]], [[Reinforcement Learning from Human Feedback|RLHF]], [[Direct Preference Optimization (DPO)]]
- Strengthens with: [[Scaling Laws|Scaling Law]]
- Strong on: [[Cold Start Problem]], [[Cross-Domain Recommendation]]
- Limitation motivates: alignment via collaborative-signal injection and [[Item Tokenization]] / [[Semantic IDs]]
- Broader paradigm: [[Generative Recommendation]] vs discriminative [[Collaborative Filtering]] scoring

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
