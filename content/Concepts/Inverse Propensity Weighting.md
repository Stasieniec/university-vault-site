---
type: concept
aliases: [IPW, IPS, inverse propensity scoring]
course: [IR]
tags: [unbiased-estimation, causal-inference, ranking]
status: complete
---

# Inverse Propensity Weighting

## Definition

**Inverse Propensity Weighting (IPW)** is a general technique for obtaining unbiased estimates from selection-biased data. The core idea: weight observations inversely to their probability of being selected/observed.

In the context of [[Learning to Rank]], IPW corrects for [[Position Bias]] by reweighting clicks by the inverse of their examination probability:

$$\text{Estimate} = \sum_{\text{observed}} \frac{\text{Observation}}{P(\text{observed})}$$

## Intuition

**Simple analogy**: Imagine surveying people on the street:
- Rich people are less likely to stop and answer your questions (low propensity)
- Poor people are more likely to participate (high propensity)
- Without correction, your survey is biased toward poor respondents
- **Solution**: Up-weight the opinions of rich people and down-weight poor people's opinions
- Result: An unbiased view of the whole population

In ranking:
- Clicks at position 1 have high propensity (100% examined)
- Clicks at position 8 have low propensity (40% examined)
- Without correction, you overlearn from position 1 and ignore position 8
- **Solution**: Down-weight position 1 clicks, up-weight position 8 clicks
- Result: An unbiased estimate of relevance independent of position

## Mathematical Formulation

### General IPW

For a selection process where observations are biased by propensity $p(x)$:

$$\widehat{\mathbb{E}}[Y] = \frac{1}{n} \sum_{i=1}^{n} \frac{Y_i}{p(X_i)}$$

This is an **unbiased** estimator:

$$\mathbb{E}[\widehat{\mathbb{E}}[Y]] = \mathbb{E}\left[\frac{Y}{p(X)}\right] = \mathbb{E}[Y]$$

### IPW for Position Bias

Under the [[Position-Based Click Model]]:

$$\text{Click}_{d,k} = \text{Exam}_k \cdot \text{Rel}_d$$

The unbiased estimator of relevance:

$$\widehat{\text{Rel}}_d = \sum_{\text{rankings}} \frac{\text{Click}_{d,k}}{P(\text{Exam}_k)}$$

Or for a ranking:

$$\text{DCG}_{\text{IPS}} = \sum_{k=1}^{n} \frac{\text{Click}_{y[k]}}{P(\text{Exam}_k) \cdot \log_2(k+1)}$$

### Inverse Propensity Weights

Concrete weights for standard examination propensities:

| Rank | $P(\text{Exam}_k)$ | Weight $1/P(\text{Exam}_k)$ |
|------|-----------------|---------------------------|
| 1    | 1.00            | 1.00x                     |
| 2    | 1.00            | 1.00x                     |
| 3    | 0.90            | 1.11x                     |
| 4    | 0.80            | 1.25x                     |
| 5    | 0.70            | 1.43x                     |
| 6    | 0.60            | 1.67x                     |
| 7    | 0.50            | 2.00x                     |
| 8    | 0.40            | 2.50x                     |

A click at rank 8 is worth 2.5× a click at rank 1 in terms of relevance evidence.

## Key Properties

### Unbiasedness

IPW produces an **unbiased** estimate under the assumed model:

$$\mathbb{E}[\text{IPS estimate}] = \text{True value}$$

This means:
- On average, across many replicates, the estimate is correct
- It doesn't mean any single estimate is correct
- It doesn't mean low variance

### Variance Problem

IPW suffers from **high variance** when propensities are low:

When $P(\text{Exam}_k)$ is small, the weight $1/P(\text{Exam}_k)$ is large, amplifying noise.

**Example**: 
- If $P(\text{Exam}_8) = 0.1$, weight = 10x
- A single noisy click at rank 8 becomes a massive signal
- Variance explodes

### Variance-Bias Trade-off

Unbiasedness is a property of the expected value, not the actual estimate quality.

```
        Biased, Low Var    Unbiased, High Var    Unbiased, Low Var
        ________________   ________________      ________________
            ╱╲                    │                     ╱╲
           ╱  ╲                   │                    ╱  ╲
          ╱    ╲                  │                   ╱    ╲
       __|______|__          _____|_____         ___|______|___
       True Value            True Value          True Value
       (Consistent           (Correct            (Correct & 
        miss)                 on average)        stable)
```

**Practical lesson**: Unbiasedness alone is not sufficient. We need both unbiasedness AND low variance.

## Practical Considerations

### Propensity Clipping

Prevent extreme weights by clipping propensities:

$$w_k = \min\left(\frac{1}{P(\text{Exam}_k)}, w_{\max}\right)$$

Common choices: $w_{\max} = 5, 10, 20$

**Trade-off**: Introduces slight bias but dramatically reduces variance.

### Non-Zero Propensities

IPW requires all items to have non-zero propensity:

$$P(\text{Exam}_k) > 0 \quad \forall k$$

**Problem**: In top-k ranking (k=10), items at rank 11+ have zero propensity. Standard IPW fails.

**Solutions**:
- Enforce stochastic policies (show every item with some probability)
- Use [[Doubly Robust Estimation]] instead

### Propensity Estimation Error

IPW depends on accurate propensity estimates $P(\text{Exam}_k)$.

If estimates are wrong:
- Weights are wrong
- Estimates become biased

Estimation error compounds, especially for low-propensity items.

## Assumptions

### 1. Correct User Model
Users must behave according to the assumed model (e.g., [[Position-Based Click Model]]).

**Violation**: If users follow [[Cascading Position Bias]] instead, PBM-based IPW fails.

### 2. Correct Propensity Estimation
Must have accurate estimates of $P(\text{Exam}_k)$.

**How to ensure**:
- Online randomization (gold standard)
- Intervention harvesting (good if assumptions hold)

### 3. Overlap / Positivity
All items must have non-zero propensity.

$$P(\text{Exam}_k) > 0 \quad \forall k$$

## Origin & Connection to Importance Sampling

IPW is essentially [[Importance Sampling]] applied to observational data:

In importance sampling, to estimate $\mathbb{E}_p[f(X)]$ when samples are from $q(X)$:

$$\mathbb{E}_p[f(X)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx \approx \frac{1}{n} \sum_{i} f(x_i) \frac{p(x_i)}{q(x_i)}$$

In ranking:
- $p(x) = $ true relevance distribution
- $q(x) = $ observed distribution (biased by position)
- Weight = $p(x) / q(x) = $ inverse propensity

## Strengths & Weaknesses

### Strengths
✓ Theoretically guaranteed unbiasedness (under assumptions)  
✓ Works across any user model (if you model it correctly)  
✓ Simple to implement  
✓ No learned parameters needed (only propensities)

### Weaknesses
✗ High variance from low-propensity items  
✗ Sensitive to propensity estimation errors  
✗ Fails with zero propensities  
✗ Variance-reduction via clipping introduces bias  
✗ Can be unstable in practice

## Connections

- **Generalization**: [[Doubly Robust Estimation]] combines IPW with learned models for lower variance
- **Alternative**: [[Click Models]] provide lower variance but weaker guarantees
- **Causal inference**: IPW is a core technique in causal inference for observational studies
- **Importance sampling**: IPW is the observational version of importance sampling

## Appears In

- [[Unbiased Learning to Rank]]
- [[Counterfactual Learning to Rank]]
- [[Doubly Robust Estimation]]
- [[Position Bias]] estimation and correction
