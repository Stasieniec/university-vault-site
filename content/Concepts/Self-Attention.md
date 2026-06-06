---
type: concept
aliases: [Self Attention]
course: [RecSys]
tags: [sequential-rec, generative-rec, exam-topic]
status: complete
---

# Self-Attention

## Definition

> [!definition] Self-Attention
> **Self-attention** is a sequence-modeling mechanism in which every position in a sequence computes a weighted combination of the representations of **all positions** (including itself), where the weights are learned from the content of the positions themselves. Each item is mapped to a **query**, a **key**, and a **value** vector; the attention weight from one position to another is the (scaled, normalized) compatibility between their query and key. It is the core building block of the [[Transformer Model]] and of attention-based sequential recommenders such as [[SASRec]] and BERT4Rec.

## Intuition

> [!intuition] Content-Based Soft Lookup
> Think of each position as issuing a **query**: "given what I am, which other items in the history matter to me?" Every other item advertises a **key** ("here is what I am") and carries a **value** ("here is the information I contribute"). The query is matched against all keys to produce a soft, normalized attention distribution, and the output is the weighted average of the values under that distribution.
>
> Unlike an [[Recurrent Neural Network (RNN)|RNN]]/[[Gated Recurrent Unit (GRU)|GRU]] (which compresses the past into a single hidden state passed step-by-step), self-attention gives every position **direct access** to every other position in one step. There is no fixed-distance bottleneck, so long-range dependencies (e.g. "phone case after phone", a series of sequels) are captured equally well regardless of how far apart they are. In [[Sequential Recommendation]] this lets the model decide, per position, which past interactions are relevant to predicting the next item.

## Mathematical Formulation

Given an input sequence packed into a matrix $X \in \mathbb{R}^{n \times d}$ (one row per position), self-attention first projects $X$ into queries, keys, and values, then applies **scaled dot-product attention**:

> [!formula] Scaled Dot-Product Self-Attention
> $$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$
> $$\text{Attention}(Q, K, V) = \text{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V$$
>
> where:
> - $X \in \mathbb{R}^{n \times d}$ — input embeddings ($n$ positions, dimension $d$). In recommenders this is item embedding + positional embedding.
> - $W^Q, W^K, W^V$ — learned projection matrices ($W^Q, W^K \in \mathbb{R}^{d \times d_k}$, $W^V \in \mathbb{R}^{d \times d_v}$).
> - $Q K^\top \in \mathbb{R}^{n \times n}$ — the matrix of all pairwise query–key compatibilities (attention logits).
> - $\sqrt{d_k}$ — scaling factor; prevents large dot products from pushing the softmax into regions with vanishingly small gradients.
> - $\text{softmax}(\cdot)$ — applied **row-wise**, turning each position's logits into an attention distribution over all positions.
> - The output is $n$ vectors, each a convex combination of the value rows $V$.

**Multi-head attention** runs $h$ such projections in parallel and concatenates them, letting different heads attend to different relationship types:

> [!formula] Multi-Head Attention
> $$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O, \qquad \text{head}_i = \text{Attention}(X W_i^Q,\, X W_i^K,\, X W_i^V)$$
> where $W^O$ projects the concatenated heads back to dimension $d$.

In a Transformer block, this is followed by a residual connection, layer normalization, and a position-wise feed-forward network (the "Add & Norm + FFN" structure used in [[SASRec]] and BERT4Rec).

## Key Properties / Variants

- **Causal (masked) self-attention** — used by [[SASRec]]. A mask sets logits for all future positions ($j > i$) to $-\infty$ before the softmax, so position $i$ may only attend to itself and earlier items. This makes the model **unidirectional / autoregressive** and suitable for left-to-right [[Next-Item Prediction|next-item prediction]].
- **Bidirectional self-attention** — used by BERT4Rec. No causal mask; every position attends to every other position. Trained with a **Cloze / masked-item (MLM)** objective (predict randomly masked items from both left and right context), since unmasked bidirectional attention would otherwise let the target leak.
- **Permutation-equivariance ⇒ needs positional encoding.** The raw mechanism is order-agnostic (it is a set operation). Order is reinjected via **positional embeddings** added to the item embeddings at the input.
- **Complexity.** Time and memory are $O(n^2 d)$ because of the full $n \times n$ attention matrix — quadratic in sequence length $n$. This is the main bottleneck for very long user histories, but parallelizes well (unlike RNN recurrence), which is why [[SASRec]] trains an order of magnitude faster than CNN/RNN baselines.
- **Direct long-range modeling.** Constant path length between any two positions (versus $O(n)$ for an [[Recurrent Neural Network (RNN)|RNN]]), so it does not suffer the vanishing-gradient bottleneck on long dependencies.

Pseudo-code for one causal self-attention layer (as in SASRec):

```pseudo
Algorithm: Causal Self-Attention (single head)
──────────────────────────────────────────────
Input: X ∈ R^{n×d}  (item emb + positional emb)
  Q ← X W^Q;  K ← X W^K;  V ← X W^V
  S ← (Q Kᵀ) / sqrt(d_k)           # n×n logits
  for i in 1..n, j in 1..n:
    if j > i: S[i,j] ← -∞           # causal mask: no peeking ahead
  A ← softmax(S, axis=rows)         # attention weights
  O ← A V                           # weighted sum of values
  return O                          # contextual representation per position
```

## Connections

- Core component of: [[Transformer Model]] / [[Transformers]]
- Used by: [[SASRec]] (causal self-attention), BERT4Rec (bidirectional), [[Self-Attentive Sequential Recommendation]]
- Alternative sequence encoder to: [[Recurrent Neural Network (RNN)|RNN]] / [[Gated Recurrent Unit (GRU)|GRU]] in [[GRU4Rec]]
- Applied to the task of: [[Sequential Recommendation]] / [[Next-Item Prediction]]
- Foundation for: [[LLM-based Recommendation]] and [[Generative Recommendation]] (decoder self-attention over token / [[Semantic IDs|semantic ID]] sequences)
- Trained with losses such as: [[BPR]], BCE, full cross-entropy / MLM (loss choice strongly affects results)

## Appears In

- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03a - Sequential Recommendation Models]]
