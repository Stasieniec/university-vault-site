---
type: concept
aliases: [rank bias, presentation bias]
course: [IR]
tags: [bias, ranking, user-behavior]
status: complete
---

# Position Bias

## Definition

**Position bias** is the systematic effect of an item's display position on the interaction it receives, independent of the item's true relevance. Users are more likely to interact with items at higher positions simply because they are more visible, prominent, or trusted—not necessarily because they are more relevant.

In ranking and search contexts, position bias manifests as:
- Items at top positions receive disproportionately more clicks
- Click-through rates decrease rapidly with rank
- This pattern persists across different items and queries

## Intuition

Imagine you're looking at search results:
1. You see the first result and it's there, so you read it
2. The second result is right below, so you examine it too
3. The third result requires scrolling; maybe you skip it
4. The tenth result is below the fold; you probably never see it

This isn't about relevance—it's about **effort and trust**. Users exhibit positional assumptions: "Google would put the best results first, so I'll trust the top items more." This creates a self-reinforcing bias in click logs.

## Mathematical Formulation

### Position-Based Click Model (PBM)

The standard decomposition of clicks into position bias and relevance:

$$P(\text{Click}_{d,k}) = P(\text{Exam}_k) \cdot P(\text{Rel}_d | q)$$

Where:
- $P(\text{Click}_{d,k})$ = probability of a click on document $d$ at position $k$
- $P(\text{Exam}_k)$ = **examination propensity** at rank $k$ (position bias term)
- $P(\text{Rel}_d | q)$ = true relevance of document $d$ for query $q$ (relevance term, independent of position)

### Typical Examination Propensities

Empirically observed examination probabilities follow a sharp decay:

| Rank | Examination Probability |
|------|------------------------|
| 1    | 100%                   |
| 2    | 100%                   |
| 3    | 90%                    |
| 4    | 80%                    |
| 5    | 70%                    |
| 6    | 60%                    |
| 7    | 50%                    |
| 8    | 40%                    |

This exponential decay reflects how quickly users stop examining further results.

## Key Properties

### Invertibility

The examination propensities are **invertible**, meaning if we know the click probabilities and the true relevance, we can recover position bias:

$$P(\text{Exam}_k) = \frac{P(\text{Click}_{d,k})}{P(\text{Rel}_d | q)}$$

This is the basis for [[Inverse Propensity Weighting]].

### Selectivity

Position bias creates a **selection mechanism**:
- High-position items are oversampled in terms of user interactions
- Low-position items are undersampled
- This is NOT random sampling; the bias is systematic and predictable

### Confounding

Position bias acts as a confounder in the causal diagram:

```
Position k → Examination → Click ← Relevance
```

If we naively treat clicks as relevance signals without accounting for position, we learn to optimize for position, not relevance.

## Variants

### Extreme Case: Item Selection Bias

In top-k ranking, items below rank $k$ have **zero examination probability**:

$$P(\text{Exam}_k) = \begin{cases} > 0 & \text{if } k \leq k_{\max} \\ 0 & \text{if } k > k_{\max} \end{cases}$$

This is position bias taken to its logical extreme.

### Cascade Model

In real user behavior, examination at position $k$ depends on whether previous positions were relevant:

$$P(\text{Exam}_k) = \prod_{j=1}^{k-1} (1 - P(\text{Rel}_j | q))$$

Users examine the next position only if they didn't find a satisfactory result above.

## Estimation Methods

### Online Randomization

**RandTop-k**: Shuffle the top-k results randomly. The mean CTR per position reveals the examination propensity.

**RandPair**: Swap a random item with a fixed pivot position. CTR ratios reveal position bias ratios.

**Trade-off**: Accurate but harms user experience.

### Intervention Harvesting

Leverage historical A/B tests where different rankers showed the same items at different positions. Use this natural variation to estimate bias without new randomization.

**Trade-off**: Non-intrusive but requires multiple rankers and assumes stable item relevance.

## Consequences for Learning

### Biased Gradient

If you train a ranking model using clicks as labels without correcting for position bias, the model learns to **affirm the logging policy**—it prefers items already ranked highly.

### The Alignment Problem

High-performing items in the training data tend to come from high positions (due to bias), so the model learns "these positions are good," not "these items are good."

## Connections

- **Causal mechanism**: Position bias is a selection bias problem, treatable through [[Inverse Propensity Weighting]]
- **Model family**: [[Click Models]] explicitly decompose position bias and relevance
- **Solutions**: [[Counterfactual Learning to Rank]], [[Doubly Robust Estimation]]
- **Related**: [[Trust Bias]], [[Cascading Position Bias]], [[Item Selection Bias]], [[Outlier Bias]]

## Appears In

- [[Learning to Rank]]
- [[Unbiased Learning to Rank]]
- [[Click Models]]
- [[Counterfactual Learning to Rank]]
