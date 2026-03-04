---
type: concept
aliases: [DR, doubly robust, DR estimator]
course: [IR]
tags: [causal-inference, unbiased-estimation, ranking]
status: complete
---

# Doubly Robust Estimation

## Definition

**Doubly Robust (DR) Estimation** is a technique that combines two estimation methods—a direct method (learned model) and inverse propensity weighting (IPW)—in a way that is **unbiased if either component is correct**.

In the context of ranking:

$$\text{DCG}_{\text{DR}} = \underbrace{\sum_k \hat{r}_{y[k]}}_{\text{Direct Method}} - \underbrace{\sum_k \frac{(\hat{r}_{y[k]} - \text{Click}_{y[k],k})}{P(\text{Exam}_k)}}_{\text{IPS Correction}}$$

## Intuition

### The Problem We're Solving

We have two approaches to unbiased estimation:

1. **Direct Method (DM)**: Learn a relevance model $\hat{r}$, predict for all items
   - ✓ Low variance (no clicks needed)
   - ✗ Biased if model is wrong

2. **Inverse Propensity Weighting (IPS)**: Reweight observed clicks
   - ✓ Unbiased (theoretically)
   - ✗ High variance (when propensities are low)

**Goal**: Combine them to get the best of both.

### The Key Insight

The decomposition:

$$\text{True value} = \text{DM prediction} + (\text{True value} - \text{DM prediction})$$

The second term (error of DM) is the **treatment residual**. We can estimate this residual using IPS:

$$\text{True value} \approx \text{DM prediction} + \text{IPS estimate of residual}$$

If DM is perfect, residual is zero → no need for IPS correction.  
If DM is wrong, IPS fixes it with unbiased residual correction.

## Mathematical Formulation

### The DR Estimator

$$\hat{\mu}_{\text{DR}} = \underbrace{\frac{1}{n}\sum_i \hat{\mu}(X_i)}_{\text{DM}} - \underbrace{\frac{1}{n}\sum_i \frac{\hat{\mu}(X_i) - Y_i}{p(X_i)} \cdot \mathbb{1}[\text{observed}_i]}_{\text{IPS residual correction}}$$

For ranking evaluation:

$$\text{DCG}_{\text{DR}} = \underbrace{\sum_{k=1}^{n} \frac{\hat{r}_{y[k]}}{\log_2(k+1)}}_{\text{DM: predicted value}} - \underbrace{\sum_{k=1}^{n} \frac{(\hat{r}_{y[k]} - \text{Click}_{y[k],k})}{P(\text{Exam}_k) \cdot \log_2(k+1)}}_{\text{IPS correction for prediction error}}$$

Where:
- $\hat{r}_{y[k]}$ = predicted relevance of item at position $k$ in ranking $y$
- $\text{Click}_{y[k],k}$ = observed click (0 or 1)
- $P(\text{Exam}_k)$ = examination propensity

### Variance Analysis

DM variance:
$$\text{Var}_{\text{DM}} = \text{Var}(\hat{r})$$

IPS variance:
$$\text{Var}_{\text{IPS}} = \mathbb{E}\left[\frac{\text{Var}(\text{Click})}{p^2}\right]$$

When propensities are low, IPS variance explodes.

DR variance:
$$\text{Var}_{\text{DR}} \approx \text{Var}_{\text{IPS}} \cdot \frac{(\text{DM error})^2}{(\text{true value})^2}$$

**Key**: If DM is decent (error is small), DR variance is much lower than IPS.

## Why "Doubly Robust"?

### Unbiasedness Guarantee

DR is unbiased if **either**:
1. The direct method is correct: $\mathbb{E}[\hat{\mu}(X)] = \mu(X)$ for all $X$, OR
2. The propensity model is correct: $p(X)$ is accurate

**You get unbiasedness even if one is wrong** (as long as the other is right).

This is the "doubly robust" property—dual sources of insurance.

### Proof Sketch

Assume true value is $\mu$ and we have an item at position $k$:

**Case 1**: DM is correct ($\hat{\mu} = \mu$)
$$\text{DR} = \mu - \frac{(\mu - \text{Click})}{p} \cdot \mathbb{1}[\text{observed}]$$

For observed items:
$$\mathbb{E}[\text{DR}] = \mu - \frac{\mathbb{E}[\mu - \text{Click}]}{p} = \mu - \frac{\mu - \mu}{p} = \mu$$ ✓

**Case 2**: Propensities are correct (regardless of $\hat{\mu}$)
$$\mathbb{E}[\text{DR}] = \mathbb{E}[\hat{\mu}] - \mathbb{E}\left[\frac{\hat{\mu} - Y}{p}\right] = \mathbb{E}[\hat{\mu}] - (\mathbb{E}[\hat{\mu}] - \mu) = \mu$$ ✓

## Practical Advantages

### 1. Robustness

If your propensity estimates are noisy, DR still works if the learned model is decent.  
If your learned model is noisy, DR still works if propensities are accurate.

**Mutual insurance policy**.

### 2. Variance Control

By leveraging a learned model, DR dramatically reduces the variance compared to pure IPS:

```
IPS Variance:    ▓▓▓▓▓▓▓▓▓ (high)
DR Variance:     ▓▓▓ (low)
Accuracy:        Both unbiased
```

### 3. Scalability

DM can be applied to all items (even those not in logs), so:
- Evaluate new rankings with unseen items
- No zero-propensity problem

### 4. Easy Implementation

Just fit a relevance model, get propensities, apply the formula.

## Comparison: IPS vs DM vs DR

| Aspect | IPS | Direct Method | Doubly Robust |
|--------|-----|---------------|---------------|
| **Requires correct propensities** | ✓ Required | ✗ Not needed | ◐ Helps but not required |
| **Requires correct model** | ✗ Not needed | ✓ Required | ◐ Helps but not required |
| **Variance** | High | Low | Low |
| **Bias** | Low | High | Low |
| **Can handle new items** | ✗ (if zero prop) | ✓ | ✓ |
| **Practical deployment** | Risky | Good | Best |

## Design Choices

### 1. Which Model to Use?

Common choices for the direct method:
- **Click models** (RegressionEM, EM-based PBM)
- **Two-tower neural networks**
- **LambdaMART or other learned-to-rank models**

**Principle**: Use a model that captures relevance well but doesn't overfit to position.

### 2. Treatment of Observed vs. Unobserved

**Option A**: Only apply IPS correction to observed items
$$\text{DCG}_{\text{DR}} = \sum_k \hat{r}_{y[k]} - \sum_{k: \text{observed}} \frac{(\hat{r}_{y[k]} - \text{Click}_{y[k],k})}{p_k}$$

**Option B**: Include counterfactual for unobserved items
$$\text{DCG}_{\text{DR}} = \sum_k \left[\hat{r}_{y[k]} - \frac{(\hat{r}_{y[k]} - \text{Click}_{y[k],k})}{p_k} \cdot \mathbb{1}[k \in \text{observed}]\right]$$

Most common: Option A (only correct for observed).

### 3. Clipping & Regularization

Even in DR, extreme propensities can cause instability:

$$w_k = \min\left(\frac{1}{p_k}, w_{\max}\right)$$

Common: $w_{\max} = 5, 10, 20$

## When DR Fails

### 1. Both Components Wrong

If the learned model **and** propensities are both misspecified, DR is biased.

```
Reality:
  Items are relevant if they match user intent
  Position bias is [100%, 80%, 60%, ...]

DR Model:
  Learned relevance from clicks (conflates popularity with relevance)
  Propensity estimates are wrong (estimated [90%, 70%, 50%, ...])

Result: DR is doubly wrong
```

### 2. High Correlation in Errors

If DM errors are correlated with low propensities (confounding), DR can amplify bias.

### 3. Severe Identifiability Issues

If the click model is unidentifiable (multiple solutions), which one was used in DM?

Different training initializations might converge to different models, each with valid likelihood but different biases.

## Implementation Considerations

### Algorithm: Training with DR

```
Input: Historical logs D with (query, ranking, clicks)
       Examination propensities P(Exam_k)

1. Fit direct method (e.g., click model or neural network)
   on click data to get relevance estimates r̂_d

2. For evaluation:
   For new ranking y with items [d_1, ..., d_n]:
     DCG = 0
     for position k = 1 to n:
       if (d_k, k) in historical logs:
         click_kdk = observed click value
       else:
         click_dk = 0  (counterfactual: no observation)
       
       dcg_contrib = r̂_dk / log2(k+1)
       if observed:
         correction = (r̂_dk - click_dk) / P(Exam_k) / log2(k+1)
         dcg_contrib -= correction
       
       DCG += dcg_contrib
   
   return DCG

3. Optimize a new ranking model on DR signals (e.g., via gradient descent)
```

### Propensity Smoothing

In practice, propensity estimates are noisy. Smooth them:

```
P(Exam_k) = (counts_k + α) / (total + α · #ranks)
```

Adds pseudocounts to prevent extreme estimates.

## Variants & Extensions

### Normalized DR

Some formulations normalize by propensities:

$$\text{DR}_{\text{norm}} = \frac{\sum_k \hat{r}_{y[k]} / p_k}{\sum_k 1/p_k}$$

### Trimmed DR

Remove observations with extreme propensities:

$$\text{DR}_{\text{trim}} = \mathbb{E}_k[\cdot | p_k > p_{\min}]$$

### Augmented IPW

A related technique that augments IPS with an outcome model:

$$\text{AIPW} = \hat{\mu}(X) - p(X) \cdot (\text{residual})$$

Similar in spirit to DR.

## Connections

- **Foundation**: Combines [[Inverse Propensity Weighting]] + learned model
- **Click Models**: Provides the direct method component
- **Causal Inference**: Core technique in observational causal inference
- **Counterfactual Evaluation**: Used in [[Counterfactual Learning to Rank]]
- **Off-Policy Learning**: Applied to ranking from logged interactions

## Appears In

- [[Unbiased Learning to Rank]]
- [[Counterfactual Learning to Rank]]
- [[Click Models]] (when combined with IPS)
- Production ranking systems (A/B testing, offline evaluation)
