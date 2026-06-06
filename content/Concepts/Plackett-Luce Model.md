---
type: concept
aliases: [Plackett-Luce model]
course: [IR]
tags: [ltr, listwise-loss, ranking, exam-topic]
status: complete
---

# Plackett-Luce Model

## Definition

> [!definition] Plackett-Luce (PL) Model
> The **Plackett-Luce model** is a probability distribution over **permutations** (rankings) of a set of items, parameterized by a real-valued **score** $s_i$ per item. It defines $P(\pi \mid s)$ — the probability of producing ranking $\pi$ — as a sequence of **softmax choices made without replacement**: at each rank position you pick the next item with probability proportional to $e^{s_i}$ among the items not yet placed.
>
> In [[Listwise LTR]] it is the probabilistic foundation of [[ListNet]] and [[ListMLE]]: instead of scoring documents independently (pointwise) or in pairs (pairwise), PL gives a single coherent distribution over the **whole** ranking, which we can then fit by maximum likelihood / cross-entropy.

## Intuition

Think of repeatedly drawing items from a bag, but biased: an item with a higher score is more likely to be drawn next. Once drawn, it is removed, and the remaining items renormalize.

- It is the **softmax extended to ranking**. A single softmax answers "which item is best?"; PL answers "which item is best, then which of the rest is best, then..." — applying softmax to a shrinking candidate set at every step.
- The model is a generative story for a ranking, so a model that outputs scores $s_i$ implicitly defines a distribution over all $n!$ orderings. Training = make the *correct* ranking probable under this distribution.
- It satisfies **Luce's choice axiom** (independence of irrelevant alternatives): the relative odds of choosing $d_i$ over $d_j$ at any step depend only on $s_i$ and $s_j$, not on what other items are present.

## Mathematical Formulation

**Full ranking probability.** For a permutation $\pi = (\pi(1), \pi(2), \dots, \pi(n))$ of $n$ items with scores $s = (s_1, \dots, s_n)$:

$$P(\pi \mid s) = \prod_{k=1}^{n} \frac{e^{s_{\pi(k)}}}{\sum_{j=k}^{n} e^{s_{\pi(j)}}}$$

where:
- $\pi(k)$ — the item placed at rank position $k$
- $s_{\pi(k)}$ — score of that item (in IR, the model's relevance score for the document)
- $\sum_{j=k}^{n} e^{s_{\pi(j)}}$ — normalizer over the items **still remaining** at step $k$ (positions $k$ through $n$); the already-placed items $\pi(1),\dots,\pi(k-1)$ are dropped from the denominator
- the product runs over all $n$ choice steps (the last factor is always $1$)

**Top-1 / first-place probability.** The marginal probability that item $i$ is ranked first is a plain softmax over all items:

$$P(\text{top-1} = i \mid s) = \frac{e^{s_i}}{\sum_{j=1}^{n} e^{s_j}}$$

where:
- $e^{s_i}$ — the (positive) "strength" of item $i$; PL is often written with strengths $w_i = e^{s_i} > 0$, so $P(\text{top-1}=i) = w_i / \sum_j w_j$
- $\sum_j e^{s_j}$ — partition over the full candidate set

**Worked example (3 documents, ranking $\pi = (d_2, d_1, d_3)$):**

$$P(\pi \mid s) = \underbrace{\frac{e^{s_2}}{e^{s_1}+e^{s_2}+e^{s_3}}}_{\text{pick }d_2\text{ first}}\cdot \underbrace{\frac{e^{s_1}}{e^{s_1}+e^{s_3}}}_{\text{pick }d_1\text{ next}}\cdot \underbrace{\frac{e^{s_3}}{e^{s_3}}}_{=\,1}$$

**Use in losses.** The two main listwise losses are likelihoods under PL:

- **ListMLE** maximizes the likelihood of the *ground-truth* ranking $\pi^*$ (labels sorted descending), i.e. minimizes $-\log P(\pi^* \mid s)$:
$$L_{\text{ListMLE}} = -\sum_{k=1}^{n} \log \frac{e^{s_{\pi^*(k)}}}{\sum_{j=k}^{n} e^{s_{\pi^*(j)}}}$$
- **ListNet** keeps only the **top-1** PL marginal and minimizes cross-entropy between the label-softmax $P^*(d_i) = e^{y_i}/\sum_j e^{y_j}$ and the score-softmax $P(d_i)=e^{s_i}/\sum_j e^{s_j}$:
$$L_{\text{ListNet}} = -\sum_{i=1}^{n} P^*(d_i) \log P(d_i)$$

where $y_i$ is the relevance label of document $i$ and $s_i$ the model's predicted score.

## Key Properties / Variants

- **Generative sampling procedure** (how to draw a ranking from PL):

```pseudo
Algorithm: Sample a ranking from Plackett-Luce(s)
──────────────────────────────────────────────────
Input: scores s_1..s_n
Remaining R ← {1, ..., n}
For k = 1 to n:
    # softmax over the items still in R
    For each i in R:  p_i ← exp(s_i) / Σ_{j in R} exp(s_j)
    Sample item m ~ Categorical(p)      # pick next-ranked item
    π(k) ← m
    R ← R \ {m}                          # sample without replacement
Return ranking π = (π(1), ..., π(n))
```

- **Softmax = top-1 PL.** A single softmax is exactly the first step of PL; full PL is "softmax with replacement removed", applied $n$ times.
- **Scale by exponentiation, shift-invariant.** Adding a constant $c$ to every score leaves $P(\pi \mid s)$ unchanged (the $e^{c}$ cancels in each ratio), so scores are only identifiable up to an additive constant.
- **Luce's choice axiom / IIA.** Pairwise odds $w_i / w_j = e^{s_i - s_j}$ are independent of the other alternatives — a known limitation when context matters.
- **Factorial blow-up.** There are $n!$ permutations, so the full distribution is intractable for large $n$ (e.g. $10! \approx 3.6\times 10^6$). Practical fixes: **ListNet's top-1 truncation**, or top-$k$ PL (only model the first $k$ choice steps).
- **Ties / multiple valid orders.** When several items share a label, any ordering among them is a valid ground truth; ListMLE handles this since the loss decomposes per step.
- **Relation to softmax policies in RL.** The same Gibbs/softmax form underlies the [[Softmax Policy]] used in [[Policy Gradient]] methods; ranking under PL can be seen as a sequential (without-replacement) softmax policy over documents.
- **Complexity.** Evaluating $-\log P(\pi^*\mid s)$ is $O(n)$ given a sorted list (precompute suffix sums of $e^{s_j}$), plus $O(n\log n)$ to sort by label.

## Connections

- Foundation of: [[Listwise LTR]] — specifically [[ListNet]] (top-1 PL) and [[ListMLE]] (full-ranking PL likelihood)
- Generalizes: the softmax / [[Softmax Policy]] (top-1 PL is a single softmax)
- Contrast with: [[Pointwise LTR]] (independent scores) and [[Pairwise LTR]] / [[RankNet]] (pairwise comparisons), neither of which models the full permutation
- Alternative listwise objective: metric-based losses like [[ApproxNDCG]] and [[LambdaRank]] / [[LambdaMART]] that approximate [[NDCG]] directly instead of fitting a permutation likelihood
- Applied within: [[Learning to Rank]], evaluated with [[NDCG]] and [[MAP]]

## Appears In

- [[Listwise LTR]]
- [[IR-L10 - Learning to Rank]]
- [[Learning to Rank]]
