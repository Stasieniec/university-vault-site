---
type: concept
aliases: [Low-Rank Adaptation, Parameter-Efficient Fine-Tuning, PEFT]
course: [RecSys]
tags: [llm, generative-rec, exam-topic]
status: complete
---

# LoRA

## Definition

> [!definition] LoRA (Low-Rank Adaptation)
> **LoRA** is a **parameter-efficient fine-tuning** (PEFT) method that adapts a large pretrained model to a downstream task **without updating its original weights**. Instead of fine-tuning a weight matrix $W_0 \in \mathbb{R}^{d \times k}$ directly, LoRA **freezes** $W_0$ and learns a **low-rank additive update** $\Delta W = BA$, where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$. Only $A$ and $B$ are trained, so the number of trainable parameters drops by orders of magnitude.
>
> In this course LoRA is the mechanism that makes the **LLM-as-RS** formulation tractable: a frozen LLM backbone plus a small trainable LoRA adapter is fine-tuned on recommendation data (e.g. TALLRec, LLaRA, CoLLM).

## Intuition

> [!intuition] Fine-tune the *change*, not the weights — and keep it skinny
> Full fine-tuning of a billion-parameter LLM means learning a dense update $\Delta W$ the same size as $W_0$ — expensive in compute, memory, and storage (one full copy of the model per task). The empirical observation behind LoRA is that the **update needed to adapt a model has a low "intrinsic rank"**: the meaningful change lives in a tiny subspace. So instead of a full $d \times k$ matrix, we factor the update through a narrow $r$-dimensional bottleneck, $\Delta W = BA$. With $r$ as small as 4–16, you train a fraction of a percent of the parameters yet recover most of the quality of full fine-tuning.
>
> For recommendation this is the enabling trick: a vanilla LLM never saw click signal, Top-K ranking, or item structure during pretraining (the alignment problem). LoRA lets you **inject that recommendation-specific signal cheaply** while leaving the frozen LLM's world knowledge intact — the snowflake (frozen LLM) + flame (trainable LoRA) picture from the LLM-as-RS slides.

## Mathematical Formulation

> [!formula] Low-Rank Weight Update
> For a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA replaces the forward pass $h = W_0 x$ with
> $$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}\, B A x$$
>
> where:
> - $W_0$ — frozen pretrained weights ($\nabla W_0 = 0$, not updated)
> - $B \in \mathbb{R}^{d \times r}$ — trainable down-projection, initialized to $0$
> - $A \in \mathbb{R}^{r \times k}$ — trainable up-projection, initialized random Gaussian
> - $r$ — rank of the adapter, $r \ll \min(d,k)$ (the bottleneck width)
> - $\alpha$ — scaling constant; $\tfrac{\alpha}{r}$ rescales the update so it is roughly $r$-independent
> - $x$ — layer input, $h$ — layer output
>
> Because $B$ is initialized to $0$, $\Delta W = BA = 0$ at the start, so training **begins exactly at the pretrained model** and only learns to deviate as needed.

> [!formula] Trainable Parameter Count
> A full update has $d \times k$ parameters; the LoRA update has only
> $$|\Theta_{\text{LoRA}}| = r(d + k) \ll d \times k$$
> e.g. for $d = k = 4096$, $r = 8$: $65{,}536$ trainable params vs $16.8$M — a $\sim 256\times$ reduction **per adapted matrix**.

In the **LLM-as-RS** training objective the LoRA adapter is optimized with the same next-token / instruction loss as the LLM, e.g. supervised fine-tuning on instances like "Given the user's liked/disliked items and a target item, answer Yes/No":

$$\min_{A, B}\ \mathcal{L} = -\sum_{t} \log p_{\,W_0 + BA}\big(y_t \mid x, y_{<t}\big)$$

where the gradient flows **only** into $A$ and $B$; $W_0$ stays frozen. The output is item text titles (LLM-as-RS) or, more generally, the adapted LLM carries the injected collaborative signal.

## Key Properties / Variants

- **No inference latency.** After training, $\Delta W = BA$ can be **merged** into the frozen weights: $W = W_0 + \tfrac{\alpha}{r} BA$. The merged model is identical in shape to the original, so unlike adapter-layer methods LoRA adds **zero** extra latency at serving time.
- **Cheap task switching.** Each task is just a small $(A, B)$ pair (megabytes, not gigabytes). You keep one frozen backbone and swap adapters — ideal for multi-domain / multi-task recommendation.
- **Where it is applied.** Typically injected into the attention projection matrices ($W_q, W_k, W_v, W_o$) of each Transformer block; sometimes the feed-forward layers too.
- **Hyperparameters.** Rank $r$ trades capacity vs cost; scaling $\alpha$ controls the update magnitude. Both are small (e.g. $r \in \{4,8,16\}$).
- **In the LLM-based GR taxonomy** LoRA appears in two of the three alignment paradigms:
  - **Text prompting** (paradigm ①): TALLRec uses lightweight LoRA fine-tuning on natural-language preference instructions.
  - **Inject collaborative signal** (paradigm ②): iLoRA, LLaRA, CoLLM project a learned CF embedding into the LLM's token-embedding space and fine-tune a LoRA adapter on top of the frozen LLM. iLoRA additionally instance-customizes the adapter.
- **Relation to other PEFT.** Belongs to the broader [[PEFT|parameter-efficient fine-tuning]] family (alongside prefix-tuning, prompt-tuning, soft prompts, and adapter layers); LoRA is the most widely used because of the zero-merge-latency property.
- **Contrast with full fine-tuning / [[SFT|SFT]].** LoRA can implement SFT cheaply, and is compatible with later [[DPO|preference optimization]] or RL stages on the same frozen backbone.

## Connections

- Subtype of: [[PEFT|Parameter-Efficient Fine-Tuning]]
- Enables: [[LLM-as-RS]] (frozen LLM + trainable LoRA adapter)
- Applied to: [[Transformers|Transformer]] attention projections in [[LLMs|Large Language Models]]
- Training objective: [[SFT|Supervised Fine-Tuning]]; followed optionally by [[DPO]] / [[RL]]
- Alternative alignment routes that avoid text-only adapters: [[Item Tokenization]] → [[Semantic IDs]] (the SID-based GR formulation)
- Contrasts with: full fine-tuning, [[In-Context Learning|in-context learning]] (zero-training prompting)

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
