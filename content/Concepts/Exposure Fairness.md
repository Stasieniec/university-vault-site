---
type: concept
aliases: [exposure fairness, fairness of exposure, attention fairness]
course: [IR]
tags: [ir-society]
status: complete
---

# Exposure Fairness

> [!definition] Exposure Fairness
> **Exposure fairness** is the principle that items (or their providers) should receive attention/visibility proportional to their relevance or merit, not distorted by position bias or systematic discrimination.

> [!intuition] The Core Problem
> Position bias means top-ranked items receive disproportionate attention regardless of actual merit. Combined with any systematic bias in the ranking algorithm, this creates compounding unfairness—items that deserve attention may never receive it simply because they start lower in rankings.

## Key Properties / Variants

### Position Bias Effect

Users interact predominantly with top-ranked results, creating a rich-get-richer dynamic:
- Items ranked higher receive more clicks → more training signal → ranked higher
- Items ranked lower receive fewer clicks → less data → remain low
- This compounds any initial bias in the ranking system

### Exposure Allocation Approaches

| Approach | Principle | Trade-off |
|----------|-----------|-----------|
| **Merit-based exposure** | Exposure ∝ relevance | Optimal for users, may perpetuate historical biases |
| **Equality-based exposure** | Equal exposure across groups | Promotes diversity, may reduce user utility |
| **Equity-based exposure** | Compensates for historical disadvantage | Addresses systemic bias, complex to implement |

### Amortized Fairness (Biega et al., 2018)

> [!important] Key Insight
> Fairness should be measured not just at a single point in time but across repeated exposures. An item might be fairly ranked in expectation but experience high variance that harms its provider.

- **Single-query fairness**: Is this ranking fair for this query?
- **Amortized fairness**: Over many queries, does each item receive fair total exposure?

### Formal Framework

For item $i$ with relevance $r_i$, exposure fairness requires:

$$\frac{\text{Exposure}(i)}{\text{Exposure}(j)} \approx \frac{r_i}{r_j}$$

This ensures proportional visibility based on merit, not position artifacts.

## Technical Approaches

- **Stochastic ranking**: Sample rankings from a distribution that achieves fairness in expectation
- **Fair re-ranking**: Post-process deterministic rankings to improve fairness
- **Fairness-constrained learning**: Train models with fairness as an explicit constraint

## Connections

- Instance of: [[Algorithmic Fairness]] (specific fairness criterion for ranking)
- Addresses: Position bias in [[Learning to Rank]]
- Evaluation: Extends [[NDCG]] with exposure-based metrics
- Contrasted with: [[Emancipatory IR]] (questions whether exposure metrics capture real fairness)

## Appears In

- [[IR-L12 - IR and Society]]
