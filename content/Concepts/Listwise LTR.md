---
type: concept
aliases: [Listwise Learning to Rank, ListNet, ListMLE, LambdaRank, ApproxNDCG, Metric-based LTR, Probabilistic LTR]
course: [IR]
tags: [ltr, ranking, machine-learning, listwise-loss, neural-ranking, ranking-metrics]
status: complete
---

# Listwise LTR

## Definition

**Listwise Learning to Rank** is an approach to [[Learning to Rank]] that considers the **entire ranking** (list of documents) when computing the loss function. Unlike pointwise methods (single documents) or pairwise methods (pairs), listwise methods directly optimize ranking quality.

Two main families:
1. **Probabilistic**: Model ranking distributions using the Plackett-Luce model
2. **Metric-based**: Directly approximate or optimize ranking metrics like [[NDCG]]

## Intuition

The core insight: A ranking is a **permutation of documents**, not just individual scores or pairwise comparisons. By considering all documents together, we can:

1. Directly optimize ranking metrics (NDCG, MAP)
2. Account for position-sensitive importance (top-10 matters more than bottom-10)
3. Avoid pathological cases where pointwise/pairwise methods produce poor results

```
Query: "machine learning"
Documents: A (rel=2), B (rel=0), C (rel=1)

Listwise loss considers the entire ranking:
- Ranking [A, C, B]: gains at positions 1,2,3 weighted by position
  Loss = compute_metric([2, 1, 0]) 
  
- Ranking [B, A, C]: would be terrible, high loss
- Ranking [A, C, B]: good ranking, low loss

Instead of independent scores or just pairwise comparisons!
```

## Mathematical Formulation

### Probabilistic Listwise: Plackett-Luce Model

The **Plackett-Luce (PL) model** generates random rankings by sequential sampling without replacement:

#### Top-1 Probability (for a single document)

$$P(d_i | \text{remaining docs}) = \frac{e^{s_i}}{\sum_{j} e^{s_j}}$$

This is the **softmax function**: probability of selecting document $i$ is proportional to its score relative to all others.

#### Full Ranking Probability

For a ranking $\pi = (d_2, d_1, d_3, ...)$ of $n$ documents:

$$P(\pi | s) = \prod_{k=1}^{n} \frac{e^{s_{\pi(k)}}}{\sum_{j=k}^{n} e^{s_{\pi(j)}}}$$

At each step $k$, select document $\pi(k)$ from the remaining $n-k+1$ documents with probability proportional to its score.

**Example with 3 documents:**

$$P(\pi = (d_2, d_1, d_3) | s) = \underbrace{\frac{e^{s_2}}{e^{s_1} + e^{s_2} + e^{s_3}}}_{\text{step 1: select } d_2} \cdot \underbrace{\frac{e^{s_1}}{e^{s_1} + e^{s_3}}}_{\text{step 2: select } d_1} \cdot \underbrace{\frac{e^{s_3}}{e^{s_3}}}_{\text{step 3: select } d_3}$$

#### Computational Challenge

The number of possible rankings is $n!$, which grows **factorially**. For $n=10$, there are $3.6 \times 10^6$ possible rankings. Computing the full distribution is infeasible.

### ListNet (Cao et al., 2007)

**ListNet** simplifies by considering only **top-1 probabilities** instead of full ranking distributions.

#### Target Distribution

Based on relevance labels, compute the softmax:

$$P^*(d_i) = \frac{e^{y_i}}{\sum_{j=1}^{n} e^{y_j}}$$

This is the probability that document $i$ would be selected first if we sampled from the label distribution.

#### Predicted Distribution

From the model scores:

$$P(d_i) = \frac{e^{s_i}}{\sum_{j=1}^{n} e^{s_j}}$$

#### ListNet Loss

Cross-entropy between target and predicted distributions:

$$L_{\text{ListNet}} = -\sum_{i=1}^{n} P^*(d_i) \log P(d_i)$$

Expanding:

$$L = -\sum_{i=1}^{n} \frac{e^{y_i}}{\sum_j e^{y_j}} \log \frac{e^{s_i}}{\sum_j e^{s_j}}$$

**Advantage**: Computationally efficient - just softmax.

**Limitation**: Only considers top-1, ignores structure of full ranking.

### ListMLE (Xia et al., 2008)

**ListMLE** directly maximizes the probability of the ground-truth ranking under the Plackett-Luce model.

#### The Ground-Truth Ranking

Order documents by their relevance labels in descending order:

$$\pi^* = \text{argsort}(y, \text{descending})$$

Example: If labels are $y = [2, 0, 1]$, then $\pi^* = (d_1, d_3, d_2)$.

#### ListMLE Loss

For each position $i$ in the ground-truth ranking, compute:

$$L_{\text{ListMLE}} = -\sum_{i=1}^{n} \log \frac{e^{s_{\pi^*(i)}}}{\sum_{j=i}^{n} e^{s_{\pi^*(j)}}}$$

**Interpretation**: At each position, maximize probability that the ground-truth document appears at that position, given the remaining documents.

**Step 1** ($i=1$): Among all $n$ documents, the best should rank first
$$-\log \frac{e^{s_{\text{best}}}}{\sum_{j=1}^{n} e^{s_j}}$$

**Step 2** ($i=2$): Among remaining $n-1$ documents, the second-best should rank second
$$-\log \frac{e^{s_{\text{2nd-best}}}}{e^{s_{\text{2nd-best}}} + e^{s_{\text{3rd-best}}} + ...}$$

And so on.

**Advantage**: Directly models full ranking structure.

**Complexity**: $O(n \log n)$ for sorting + $O(n)$ for loss.

**Note**: Handles ties (multiple valid rankings) naturally.

### Metric-Based Listwise: ApproxNDCG

**ApproxNDCG** directly approximates the [[NDCG]] metric using smooth approximation.

#### The Ranking Function (Non-differentiable)

The rank of document $i$ in a sorted list:

$$\text{rank}(d_i) = 1 + \sum_{j: j \neq i} \mathbb{1}[s_j > s_i]$$

where $\mathbb{1}[\cdot]$ is the indicator function.

**Problem**: The indicator function has zero gradient everywhere.

#### Smooth Approximation

Replace the indicator with sigmoid:

$$\mathbb{1}[s_j > s_i] \approx \frac{1}{1 + e^{(s_i - s_j)}} = \sigma(s_i - s_j)$$

Approximate rank:

$$\tilde{\text{rank}}(d_i) = 1 + \sum_{j: j \neq i} \frac{1}{1 + e^{(s_i - s_j)}}$$

#### ApproxNDCG Loss

Plug the approximate rank into DCG formula:

$$\text{DCG} = \sum_{i=1}^{n} \frac{2^{y_i} - 1}{\log_2(1 + \tilde{\text{rank}}(d_i))}$$

Loss is negative DCG (to minimize):

$$L_{\text{ApproxNDCG}} = -\text{DCG}$$

**Advantage**: Directly optimizes NDCG metric.

**Limitation**: Assumes metric weights decrease smoothly with rank (not valid for fairness or absolute cutoff metrics).

### Metric-Based Listwise: LambdaRank

**LambdaRank** (Burges et al., 2006) bridges pairwise and listwise thinking by scaling pairwise gradients by metric impact.

#### The Insight

To optimize a metric, we only need the **gradient** of the loss, not the loss itself. We can:

1. Use a differentiable loss (e.g., pairwise logistic)
2. Scale its gradient by how much the pair affects the ranking metric

#### LambdaRank Loss

$$L_{\text{LambdaRank}} = \sum_{y_i > y_j} \log(1 + e^{-(s_i - s_j)}) \cdot |\Delta M(i, j)|$$

where $\Delta M(i, j)$ is the change in ranking metric $M$ (e.g., NDCG) if we swap documents $i$ and $j$.

**Formula for gradient scaling:**

$$\frac{\partial L}{\partial s_i} = \sum_{j: y_j > y_i} (1 - \sigma(s_j - s_i)) \cdot |\Delta M(j, i)| - \sum_{j: y_i > y_j} \sigma(s_i - s_j) \cdot |\Delta M(i, j)|$$

**Key Effect**: Pairs at top positions (which affect NDCG more) get larger gradients.

#### LambdaMART

**LambdaMART** applies LambdaRank with MART ([[Multiple Additive Regression Trees]]). This remains **one of the strongest baseline LTR methods** in practice.

## Key Properties

- **Ranking-Aware**: Considers entire document lists
- **Metric-Aligned**: Can directly optimize ranking metrics (in metric-based variants)
- **Position-Sensitive**: Can weight top positions more heavily
- **Theoretically Grounded**: Probabilistic variants based on Plackett-Luce model
- **Computationally Intensive**: Higher per-batch cost than pairwise methods

## Advantages

1. **Direct Metric Optimization**: Metric-based methods optimize what you actually care about
2. **Position-Aware**: Can naturally emphasize top-ranked documents
3. **Theoretically Sound**: Probabilistic methods grounded in ranking theory
4. **Better Quality**: Generally outperforms pointwise/pairwise when computational budget allows
5. **Handles Ties**: Works well when multiple valid rankings exist

## Limitations

1. **Computational Cost**: More expensive per training batch than pairwise methods
2. **Scalability**: Full PL model intractable for large $n$ (typically $n \leq 100$)
3. **Metric Assumptions**: ApproxNDCG assumes smooth metric decay (not all metrics qualify)
4. **Target Probabilities**: ListNet uses somewhat arbitrary softmax over labels
5. **Memory**: Typically need all documents per query in memory

## Comparison of Listwise Methods

| Method | Input | Output | Metric | Computational |
|--------|-------|--------|--------|---|
| **ListNet** | Labels | Probability distribution | Top-1 alignment | $O(n)$ |
| **ListMLE** | Ranking order | PL probability | Full ranking | $O(n \log n)$ |
| **ApproxNDCG** | Scores | Approximate NDCG | Direct NDCG approx. | $O(n^2)$ |
| **LambdaRank** | Scores + metric | Metric-scaled gradient | Any metric | $O(n^2)$ |

## Modern Extensions

- **Deep Listwise**: Neural networks for end-to-end learning
- **Differentiable Sorting**: Research on fully differentiable sorting
- **Multi-Task Learning**: Combine multiple ranking objectives (NDCG, MAP, MRR)
- **Domain Adaptation**: Transfer learning across different ranking tasks

## When to Use

✓ **Use listwise LTR when:**
- NDCG or other position-sensitive metrics are critical
- You have all documents in memory (small lists)
- You can afford computational cost
- Top-position ranking quality matters most
- You want theoretically justified optimization

✗ **Avoid listwise LTR when:**
- Memory is extremely constrained
- You have massive lists ($n > 1000$)
- Metric optimization not critical
- Pairwise methods already working well

## Connections

- **Related Methods**: [[Pointwise LTR]], [[Pairwise LTR]]
- **Foundations**: Based on [[Plackett-Luce model]], ranking theory
- **Modern Usage**: Integrated into neural ranking with [[Transformers]], [[BERT for IR]]
- **Ranking Metrics**: [[NDCG]], [[MAP]], [[Precision]], [[Recall]]
- **Practice**: [[LambdaMART]] remains industry standard in many systems

## Appears In

- [[IR-L10 - Learning to Rank]] (lecture)
- [[Learning to Rank]] (concept)
- [[Pointwise LTR]] (concept)
- [[Pairwise LTR]] (concept)

## References

- Cao, Z., Qin, T., Liu, T. Y., Tsai, M. F., & Li, H. (2007). Learning to rank: From pairwise approach to listwise approach. In ICML.
- Xia, F., Liu, T. Y., Wang, J., Zhang, W., & Li, H. (2008). Listwise approach to learning to rank. In ICML.
- Burges, C., Shaked, T., Renshaw, E., et al. (2006). Learning to rank using gradient descent. In ICML.
- Qin, T., Liu, T. Y., & Li, H. (2010). A general approximation framework for direct optimization of information retrieval measures. Information Retrieval, 13(4), 375-397.
- Luce, R. D. (2012). *Individual choice behavior: A theoretical analysis*. Courier Corporation.
- Plackett, R. L. (1975). The analysis of permutations. Journal of the Royal Statistical Society, 24(2), 193-202.
- Liu, T. Y. (2009). *Learning to rank for information retrieval*. Foundations and Trends in IR, 3(3), 225-331.
