---
type: concept
aliases: [Pointwise Learning to Rank, Pointwise Ranking]
course: [IR]
tags: [ltr, ranking, machine-learning, regression, classification]
status: complete
---

# Pointwise LTR

## Definition

**Pointwise Learning to Rank** is an approach to [[Learning to Rank]] that treats ranking as a **regression or classification problem on individual query-document pairs**, independent of other documents in the ranking.

The core idea: predict a relevance score for each query-document pair, then sort by predicted scores to produce a ranking.

## Intuition

Pointwise methods operate on the principle that if we can accurately predict the true relevance label for each query-document pair, sorting by predicted scores should naturally produce a correct ranking. Each document is evaluated in isolation.

```
Query: "machine learning"
Document A: features [5.2, 3.1, 1.4, ...] → predicted score 0.8 (relevant)
Document B: features [2.1, 1.5, 0.9, ...] → predicted score 0.3 (irrelevant)
Document C: features [4.1, 2.8, 1.2, ...] → predicted score 0.6 (somewhat relevant)

Ranking: A (0.8) > C (0.6) > B (0.3)
```

## Mathematical Formulation

### Regression Formulation

Minimize squared error between predicted scores and relevance labels:

$$L_{\text{pointwise}} = \sum_{q,d} (y_{q,d} - f(\vec{x}_{q,d}))^2$$

where:
- $y_{q,d} \in \mathbb{R}$ is the continuous relevance label (or relevance converted to continuous)
- $f(\vec{x}_{q,d})$ is the predicted relevance score

### Classification Formulation

Treat relevance categories as unordered classes and use cross-entropy loss:

$$L_{\text{classification}} = -\sum_{q,d} \sum_{c=0}^{C} y_{q,d}^{(c)} \log \hat{y}_{q,d}^{(c)}$$

### Ordinal Regression Formulation

Treat relevance as ordered categories (e.g., 0 < 1 < 2 < 3 < 4) and use ordinal-specific losses that respect the ordering.

## Key Properties

- **Independence Assumption**: Each document-query pair is scored independently
- **No Ranking Structure**: The loss doesn't account for how scores interact in a ranking
- **Differentiable**: Uses standard ML losses (MSE, cross-entropy) that are fully differentiable
- **Interpretable**: Direct connection between label and predicted score

## Limitations

### Fundamental Problem: Lower Loss ≠ Better Ranking

The core issue with pointwise methods is **illustrated by counterexample**:

**Configuration 1:**
- Labels: [1, 0, 0, 0]
- Predictions: [0.6, 0.5, 0.5, 0.5]
- Loss: $\sum (y - \hat{y})^2 = 0.16 + 0.25 + 0.25 + 0.25 = 0.91$
- Ranking: doc₀ (rel) > doc₁,₂,₃ (irrel)
- MRR: 1.0 ✓

**Configuration 2:**
- Labels: [1, 0, 0, 0]
- Predictions: [0.2, 0.2, 0.2, 0.1]
- Loss: $\sum (y - \hat{y})^2 = 0.64 + 0.04 + 0.04 + 0.01 = 0.73$
- Ranking: doc₀,₁,₂ (score 0.2) > doc₃ (score 0.1)
- MRR: 0.33 ✗

**Conclusion**: Configuration 2 has **lower loss but worse ranking quality** (MRR = 0.33 vs 1.0).

### Other Limitations

1. **Class Imbalance**: Irrelevant documents vastly outnumber relevant ones, skewing the loss
2. **Ignores Ranking Context**: An error in top positions equally weighted as error in bottom positions
3. **Ignores Document Dependencies**: Doesn't account for how changing one document's score affects the ranking

## Variants & Approaches

| Variant | Loss | Target | Common Use |
|---------|------|--------|-----------|
| **Regression** | MSE, MAE | Continuous scores | Simple baselines |
| **Classification** | Cross-entropy | Probability of each class | Multi-level relevance |
| **Ordinal Regression** | Order-preserving loss | Ordered categories | Respecting label ordering |
| **Logistic Regression** | Logistic loss | Binary relevance (relevant/irrelevant) | Click prediction |

## Why Still Used?

Despite limitations, pointwise methods persist because:

1. **Simplicity**: Straightforward to implement and understand
2. **Standard ML Framework**: Uses well-known regression/classification techniques
3. **Baseline**: Good starting point before exploring pairwise/listwise methods
4. **Scalability**: Can handle very large datasets efficiently
5. **Feature Engineering**: Focuses attention on good feature design

## Connections

- **Related Methods**: [[Pairwise LTR]], [[Listwise LTR]]
- **In Practice**: Often used as first-pass ranker before neural reranking
- **Extensions**: Enhanced with feature engineering or ensemble methods
- **Modern Usage**: Largely superseded by [[Pairwise LTR]] and [[Listwise LTR]] for final rankers, but still relevant for understanding LTR fundamentals

## Appears In

- [[IR-L10 - Learning to Rank]] (lecture)
- [[Learning to Rank]] (concept)

## References

- Liu, T. Y. (2009). *Learning to rank for information retrieval*. Foundations and Trends in IR, 3(3), 225-331.
- Cossock, D., & Zhang, T. (2006). Subset ranking using regression. In COLT.
- Fuhr, N. (1989). Optimum polynomial retrieval functions. ACM TOIS, 7(3), 183-204.
