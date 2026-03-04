---
type: lecture
course: IR
week: 5
lecture: 11
book_sections: []
topics:
  - "[[Learning to Rank]]"
  - "[[Click Models]]"
  - "[[Position Bias]]"
  - "[[Inverse Propensity Weighting]]"
  - "[[Counterfactual Learning to Rank]]"
  - "[[Examination Hypothesis]]"
  - "[[Trust Bias]]"
  - "[[Item Selection Bias]]"
  - "[[Cascading Position Bias]]"
  - "[[Outlier Bias]]"
  - "[[Doubly Robust Estimation]]"
status: complete
---

# Unbiased Learning to Rank

## Overview & Motivation

Traditional [[Learning to Rank]] methods rely on expensive and stationary annotated datasets with expert relevance judgments. This approach is infeasible in many practical scenarios: it's costly, raises privacy concerns, cannot handle personalization, and often disagrees with actual user preferences. 

The key insight is that ranking systems already generate implicit feedback through user interactions (clicks, dwells, conversions). However, this data is severely biased—clicks don't reflect true relevance but are heavily influenced by how items are displayed. **Position bias** is the dominant confound: users examine top-ranked items more frequently, so clicks correlate with position as much as relevance. This lecture introduces the mathematical and algorithmic foundations for **unbiased learning to rank (ULTR)**, which corrects for selection biases in user interaction data to recover true relevance estimates.

---

## Part 1: The Problem — Position Bias

### Why Clicks Are Biased

> [!definition]
> **Position Bias**: The display position of an item has a large effect on the interactions it will receive, independent of the item's true relevance.

Users don't examine all ranks for several reasons:
- **Limited examination**: Users do not scan the entire ranking; many abandon after finding a satisfactory result
- **Sequential scanning**: Users examine items top-to-bottom and often stop early
- **Trust in ranking**: Items at top positions receive more trust, inflating click rates regardless of relevance
- **Limited display**: In top-k settings, items below the fold are never displayed, thus never clicked

**Key observation**: Click-through rates often say more about *how an item was displayed* than about its *actual relevance*.

### Position Bias as Selection Bias

> [!intuition]
> Position bias is a **selection bias problem**: the data we observe (clicks) are filtered through a systematic process (examination propensity) that depends on position, not on true item relevance alone.

Formally, we can decompose the click probability under a **Position-Based Model (PBM)**:

$$P(\text{Click}_{d,k}) = P(\text{Exam}_k) \cdot P(\text{Rel}_d | q)$$

where:
- $P(\text{Exam}_k)$ = probability of examining rank $k$ (depends only on position)
- $P(\text{Rel}_d | q)$ = true relevance probability of document $d$ for query $q$ (independent of position)

Typical examination propensities:

| Rank | Examination Prob. |
|------|------------------|
| 1    | 100%             |
| 2    | 100%             |
| 3    | 90%              |
| 4    | 80%              |
| 5    | 70%              |
| 6    | 60%              |
| 7    | 50%              |
| 8    | 40%              |

The examination propensity decreases rapidly, creating severe positional bias.

### Why Naive Learning Fails

Suppose we naively optimize a ranking model by maximizing click frequency as a proxy for relevance:

$$\text{DCG}_{\text{naive}} = \sum_{k=1}^{n} \frac{\text{Click}_{y[k]}}{\log_2(k+1)}$$

This estimator is **biased**:

$$\mathbb{E}[\text{Click}_{d,k}] = P(\text{Exam}_k) \cdot P(\text{Rel}_d | q) \neq P(\text{Rel}_d | q)$$

Result: A model trained on this naive objective will learn to rank items by *position bias*, not by relevance. It will prefer items that are already ranked highly (affirming the logging policy) rather than discovering truly relevant items.

---

## Part 2: Inverse Propensity Scoring (IPS)

### The General Solution: IPS

> [!definition]
> **Inverse Propensity Scoring (IPS)**: Weight observations inversely to their observation probability. In expectation, this removes the effect of any selection bias.

The core idea comes from [[Importance Sampling]]: if we observe something with low probability, we up-weight it; if something is observed with high probability, we down-weight it. This balances the selection process.

For position bias:

$$\text{DCG}_{\text{IPS}} = \sum_{k=1}^{n} \frac{\text{Click}_{y[k]}}{P(\text{Exam}_k) \cdot \log_2(k+1)}$$

By weighting each click by the **inverse** of the examination propensity, we get an unbiased estimate:

$$\mathbb{E}[\text{IPS}] = \mathbb{E}\left[\frac{\text{Click}_{d,k}}{P(\text{Exam}_k)}\right] = \mathbb{E}\left[\frac{P(\text{Exam}_k) \cdot P(\text{Rel}_d | q)}{P(\text{Exam}_k)}\right] = P(\text{Rel}_d | q)$$

**Inverse Propensity Weights** for standard examination propensities:

| Rank | Exam. Prob. | IPS Weight |
|------|-------------|-----------|
| 1    | 100%        | 1.00x     |
| 2    | 100%        | 1.00x     |
| 3    | 90%         | 1.11x     |
| 4    | 80%         | 1.25x     |
| 5    | 70%         | 1.43x     |
| 6    | 60%         | 1.67x     |
| 7    | 50%         | 2.00x     |
| 8    | 40%         | 2.50x     |

Note how IPS dramatically upweights clicks at lower ranks, compensating for the reduced examination probability.

### Unbiasedness vs. Variance Trade-off

> [!warning]
> **Unbiasedness does not mean zero error!**
> 
> - **Unbiased** = the expected value of the estimator equals the true value
> - **Low variance** = the estimator is stable and doesn't fluctuate wildly
> 
> These are two separate properties.

An unbiased but high-variance estimator can still be useless in practice. IPS suffers from a critical problem: **when propensities are low (rare observations), inverse weights become very large, amplifying noise**.

Example: A click at rank 8 (40% examination) gets 2.50x weight. If that click is due to noise rather than true relevance, it severely distorts the estimate.

> [!tip]
> **Clipping Propensities**: A common practical solution is to clip propensities to a minimum threshold (e.g., 0.1), preventing weights from becoming too extreme:
> 
> $$w_k = \min\left(\frac{1}{P(\text{Exam}_k)}, w_{\max}\right)$$

### Conditions for Unbiasedness

IPS unbiasedness relies on critical assumptions:

1. **Correct propensity estimation**: We must have accurate estimates of $P(\text{Exam}_k)$
2. **Correct user model**: Users must actually behave according to the assumed model (e.g., PBM)
3. **Non-zero propensities**: All items must have non-zero examination probability

**Violation of assumption 3**: In extreme cases like top-k ranking where items below the fold have 0% examination probability, standard IPS fails entirely.

---

## Part 3: Estimating Position Bias

Before applying IPS, we must estimate $P(\text{Exam}_k)$. Three main approaches:

### Method 1: Online Randomization

The most direct approach: deliberately randomize rankings online to estimate position bias.

#### RandTop-k: Shuffle Top-k
- Randomly shuffle the top-k results
- Observe where each item receives clicks
- Mean CTR per rank is proportional to examination propensity

**Advantage**: Direct, unbiased measurement  
**Disadvantage**: Harms user experience by showing random rankings

#### RandPair: Pairwise Swaps
- Swap a random document with a fixed pivot rank $k$ (e.g., $k=3$)
- Only perturb one item at a time, reducing user impact
- CTR ratio reveals position bias ratio:

$$\frac{P(\text{Exam}_i)}{P(\text{Exam}_j)} \propto \frac{\text{CTR}_i}{\text{CTR}_j}$$

**Advantage**: Lower user impact than RandTop-k  
**Disadvantage**: Still introduces randomness into the ranking

### Method 2: Intervention Harvesting

Leverage existing variance in historical data instead of online randomization.

> [!definition]
> **Intervention Harvesting**: Use historical A/B/C tests and ranker diversity to identify items shown at different positions, enabling bias estimation without new randomization.

**Key idea**: If different rankers placed the same item at different ranks in the past, we can use those natural interventions.

From historical logs:
- Ranker A ranks document D at position 1
- Ranker B ranks document D at position 3
- Compare clicks on D across these two contexts

Requires weighting by exposure: documents appear at certain ranks more frequently, so we must control for that.

**Assumptions**:
- Different rankers don't handle fundamentally different query distributions
- Item relevance doesn't shift significantly over time

### Method 3: Randomized Online Experiments

> [!tip]
> The gold standard: run online A/B tests with intentional randomization to collect a gold-standard bias estimate. Use this estimate offline for model training.

---

## Part 4: Click Models

An alternative to IPS is to **jointly estimate both bias and relevance** using probabilistic models of user behavior.

### The Position-Based Click Model (PBM)

> [!definition]
> **Position-Based Click Model (PBM)**: A probabilistic model that decomposes clicks into two independent factors:
> 
> $$P(\text{Click}_{d,k} = 1) = P(\text{Exam}_k) \cdot P(\text{Rel}_d | q)$$

**Assumptions**:
1. **Examination hypothesis**: A user clicks only on items they examine AND find relevant
2. **Position independence**: Examination probability depends only on rank, not on other items
3. **Item independence**: Whether an item is relevant depends only on the query and item, not on surrounding items

**Estimation**: Use Expectation-Maximization (EM) to jointly infer examination and relevance probabilities from click data.

### RegressionEM

Problem: Classic PBM requires multiple observations of the same query-document pair to estimate per-pair relevance. This is sparse in practice.

> [!definition]
> **RegressionEM**: Extend click models to leverage query-document features by regressing estimated relevance on features during the M-step.

In the **E-step**, estimate relevance from the current model:
$$r^{(t)}_{d,q} = \frac{P(\text{Click}_{d,k}) / P(\text{Exam}_k)}{1 + \text{smoothing}}$$

In the **M-step**, fit a regression model to predict relevance from features:
$$\hat{r}_{d,q} = f(x_d, x_q; \theta)$$

Then update examination probabilities using the fitted model.

**Advantage**: Generalizes across query-document pairs via features  
**Caveat**: If the relevance model is misspecified, it can introduce bias

### Neural Click Models

> [!intuition]
> Replace EM-based click models with neural networks that parameterize the same probabilistic structure.

Examples:
- **RNNs/GNNs**: Capture dependencies between items in the ranking
- **Attention mechanisms**: Learn which contextual factors influence examination
- **Two-tower architectures**: Explicitly separate relevance and bias towers

Benefits:
- More flexible function approximation
- Can model complex user behavior patterns
- Scalable to large datasets

### Two-Tower Models

A specific neural architecture that enforces the PBM decomposition:

```
                    Relevance Tower              Bias Tower
                           |                            |
                    (doc features only)         (position, context)
                           |                            |
                    [Neural Layers]             [Neural Layers]
                           |                            |
                    Relevance Score             Examination Prob.
                           |___________________________|
                                      |
                                 Click Prob.
```

**Training**: Maximize likelihood of observed clicks:

$$\max_\theta \sum_{\text{logs}} \log P(\text{Click} | \text{relevance}(\theta), \text{exam}(\theta))$$

**Regularization** (prevent overfitting to bias):
- Dropout in bias tower
- Data randomization
- Careful feature selection

### Identifiability in Click Models

> [!warning]
> **Critical problem**: For certain data distributions, there exist *multiple* different combinations of examination and relevance parameters that equally well explain the click data.
> 
> This means:
> - Different model initializations can converge to different solutions
> - All solutions have equally high likelihood on training data
> - Yet they make *different predictions* about relevance

**Why?** When items always appear in the same positions or when features are confounded with bias, we cannot uniquely separate relevance from bias.

**No easy fix exists**. Even neural models suffer from identifiability issues.

### Click Models: Likelihood Guarantees

Whether maximizing likelihood of click data leads to correct estimates depends on:

1. Randomization in logged rankings (or overlapping feature distributions)
2. Availability of informative features for both relevance and bias
3. Sufficient observations per query/document/user
4. Whether the problem is identifiable
5. Whether optimization finds a good local minimum

**Conclusion**: Click models are relatively easy to apply but provide **no strong guarantees** of unbiasedness. They rely on implicit structural assumptions that may or may not hold.

---

## Part 5: Doubly Robust Estimation

### The Direct Method (DM)

If we had a learned model $\hat{r}_{d,q}$ that estimates relevance from features (e.g., from a click model), we could directly evaluate any ranking:

$$\text{DCG}_{\text{DM}} = \sum_{k=1}^{n} \frac{\log_2(k+1)}{\max(0, \hat{r}_{y[k],q})}$$

**Advantage**: No variance from clicks; uses all documents  
**Disadvantage**: Requires all relevance estimates to be 100% correct to be unbiased

In practice, $\hat{r}$ will have errors, so **DM is almost always biased**.

### Combining IPS and DM: Doubly Robust Estimation

> [!definition]
> **Doubly Robust Estimation**: Combine the unbiasedness guarantee of IPS with the low variance of DM.
> 
> The key insight: subtract the direct method's bias from itself using IPS.

The estimator:

$$\text{DCG}_{\text{DR}} = \underbrace{\sum_{k=1}^{n} \frac{\hat{r}_{y[k],q}}{\log_2(k+1)}}_{\text{Direct Method}} - \underbrace{\sum_{k=1}^{n} \frac{(\hat{r}_{y[k],q} - \text{Click}_{y[k],k}) \cdot \mathbb{1}[\text{Click}_{y[k],k} = 1]}{P(\text{Exam}_k) \cdot \log_2(k+1)}}_{\text{IPS Correction}}$$

**Why it works**:
- If $\hat{r}$ is accurate (close to true relevance), the IPS correction term is small
- If $\hat{r}$ is inaccurate, IPS corrects it using observed clicks (with inverse weighting)
- The estimator is unbiased if **either**:
  - IPS is unbiased (propensities and user model correct), OR
  - DM is unbiased (relevance model correct per document)

**Advantage**: Much lower variance than pure IPS, more robust than pure DM  
**Trade-off**: Requires both a reasonable relevance model AND reasonable propensity estimates

### Comparison: IPS vs. DM vs. DR

| Method | IPS | Direct Method | Doubly Robust |
|--------|-----|---------------|---------------|
| **Requires correct propensities** | ✓ | ✗ | Helps but not required |
| **Requires correct relevance model** | ✗ | ✓ | Helps but not required |
| **Variance** | High | Low | Low-Medium |
| **Bias** | Low | High | Low |
| **Unbiased if either condition fails** | No | No | **Yes** |

---

## Part 6: Advanced User Models Beyond PBM

### Cascading Position Bias

The PBM assumes a very simplified user behavior: examination depends *only* on position. In reality:

> [!definition]
> **Cascading Position Bias**: Users examine items sequentially (top-to-bottom), but stop earlier if they find a relevant item. Thus, examination of rank $k$ depends on whether ranks $1, \ldots, k-1$ were relevant.

**Cascade Model**:

$$P(\text{Exam}_k) = \prod_{j=1}^{k-1} (1 - P(\text{Rel}_j | q))$$

Examination at position $k$ = product of irrelevance of all previous items.

**Problem**: Examination propensities are item-specific and ranking-specific, not just position-specific.

**Solutions**:
1. **Joint estimation** (e.g., RegressionEM with cascade assumptions)
2. **Session-dependent probabilities**: Use clicks in the current session to estimate propensities for that specific ranking

**Gotcha**: IPS using PBM assumptions on cascading behavior causes severe bias. The wrong model kills the method.

### Trust Bias

> [!definition]
> **Trust Bias**: Users click on top-ranked items more than justified by relevance because they trust search engines and assume top items are more relevant.

This creates false positives: irrelevant items at high ranks get undeserved clicks.

**Model**:

$$P(\text{Click}_{d,k}) = P(\text{Exam}_k) \cdot [P(\text{Rel}_d) + (1 - P(\text{Rel}_d)) \cdot \tau_k]$$

where $\tau_k$ = trust bias at rank $k$ (false positive click rate).

Trust bias applies an **affine transformation** to relevance:
- Relevant items get even more clicks (boosted)
- Irrelevant items still get some clicks (false positives)
- Effect is strongest at top ranks

**Estimation**: Can be fit using RegressionEM with the extended affine model. Reduces to 2 parameters instead of 3.

### Item Selection Bias

> [!definition]
> **Item Selection Bias**: In top-k ranking, only k items are displayed. Items below the fold have 0% examination probability.

**Extreme form of position bias** with a hard cutoff.

**Problem**: Standard IPS requires non-zero propensities:

$$\text{IPS} = \sum \frac{\text{Click}}{P(\text{Exam})}$$

If $P(\text{Exam}) = 0$, this is undefined.

**Solutions**:
1. **Stochastic policies**: Ensure non-zero probability of showing every item (but harms user experience)
2. **Doubly robust estimation**: DM handles unseen items; DR adds IPS correction for observed ones

### Surrounding Item Bias (Context and Outlier Bias)

#### Surrounding Item Bias
Users' examination of item $k$ might depend on surrounding items:
- Position alone doesn't capture it
- The relevance/quality of previous items matters
- Layout and presentation of neighbors matters

#### Outlier Bias
Items with unusual/distinctive features attract disproportionate attention:
- A high-priced item among low-priced ones
- Colored tags or visual distinctiveness
- Outlier items get more clicks independent of relevance

**Models**: Extend PBM to condition examination on surrounding item context:

$$P(\text{Exam}_k) = f(k, \text{context}(\text{nearby items}))$$

$$P(\text{Exam}_k) = g(k, \text{outlier position})$$

---

## Part 7: Real-World Results

While simulation experiments often show large ULTR improvements, what do online A/B tests reveal?

> [!intuition]
> **Median uplift from ULTR methods in online A/B tests: 2.08%**

Meta-analysis of papers with online experiments:
- Best-case: ~5-6% uplift
- Median: ~2%
- Worst-case: <1% or occasionally negative

**Why not bigger gains?**
1. Production systems are already fairly good
2. Many already incorporate some form of bias correction
3. Offline gains from ULTR don't always translate to online
4. Models must be robust, not just optimal on offline metrics
5. Implementation complexity and infrastructure overhead

**Conclusion**: ULTR makes a real-world difference, but improvements are usually incremental rather than transformative.

---

## Part 8: Beyond PBM — Other Biases

The field is actively exploring biases beyond position:

- **Presentation bias**: Visual design influences clicks
- **Conformity bias**: Users click what others have clicked (social proof)
- **Popularity bias**: Well-known items get more clicks
- **Positivity bias**: Users are more likely to give positive feedback
- **Cross-platform effects**: User behavior differs by device

---

## Key Takeaways & Summary

> [!summary]
> 
> ### Core Concepts
> 1. **Position bias** is the dominant confound in learning from clicks. Raw clicks conflate relevance with position.
> 
> 2. **Selection bias** framework: clicks are observations filtered through position-dependent examination propensities.
> 
> 3. **Inverse Propensity Scoring (IPS)** recovers unbiased relevance estimates by weighting clicks inversely to examination probability. Unbiased but high-variance.
> 
> 4. **Click models** jointly estimate relevance and bias via probabilistic models (EM) or neural networks. Lower variance but no strong unbiasedness guarantees.
> 
> 5. **Doubly robust estimation** combines the benefits of both: unbiased + lower variance than IPS.
> 
> ### Position Bias Estimation
> - **Online randomization** (gold standard): RandTop-k, RandPair
> - **Intervention harvesting**: Leverage historical A/B test diversity
> - Both have trade-offs between accuracy and user impact
> 
> ### Advanced Models
> - **Cascading**: Exam at rank $k$ depends on relevance of previous ranks
> - **Trust bias**: False positive clicks at high ranks
> - **Item selection bias**: Zero probability below cutoff
> - **Context/outlier bias**: Exam depends on surrounding items
> 
> ### Real World
> - ULTR methods produce ~2% median online uplift
> - Improvements are consistent but incremental
> - Implementation complexity is non-trivial
> 
> ### Unbiasedness ≠ Low Variance
> - Unbiased means correct expected value, not correct estimate
> - IPS has low bias but high variance (when propensities are low)
> - Propensity clipping is a practical variance-reduction technique
> - DR elegantly trades off bias and variance

---

## References

The lecture is based on the tutorial:
> Recent Advances in the Foundations and Applications of Unbiased Learning to Rank  
> Presented at SIGIR 2023, FIRE 2023, and WSDM 2024  
> Authors: Shashank Gupta, Philipp Hager, Jin Huang, Ali Vardasbi, Harrie Oosterhuis

### Key Papers

**Position Bias & IPS:**
- Joachims, T., Swaminathan, A., & Schnabel, T. (2017). Unbiased learning-to-rank with biased feedback. WSDM.
- Wang, X., Bendersky, M., Metzler, D., & Najork, M. (2016). Learning to rank with selection bias in personal search. SIGIR.

**Click Models:**
- Chuklin, A., Markov, I., & Rijke, M. D. (2015). Click models for web search. Foundations and Trends in IR.
- Wang, X., et al. (2018). Position bias estimation for unbiased learning to rank in personal search. WSDM.

**Advanced Methods:**
- Oosterhuis, H. (2023). Doubly robust estimation for correcting position bias in click feedback. TOIS.
- Vardasbi, A., de Rijke, M., & Markov, I. (2020). Cascade model-based propensity estimation. SIGIR.
- Yan, L., et al. (2022). Revisiting two-tower models for unbiased learning to rank. SIGIR.

**Intervention Harvesting:**
- Agarwal, A., et al. (2019). Estimating position bias without intrusive interventions. WSDM.
- Fang, Z., Agarwal, A., & Joachims, T. (2019). Intervention harvesting for context-dependent examination-bias estimation. SIGIR.

**Beyond Position Bias:**
- Agarwal, A., et al. (2019). Addressing trust bias for unbiased learning-to-rank. WWW.
- Sarvi, F., et al. (2023). On the impact of outlier bias on user clicks. SIGIR.
- Wu, X., et al. (2021). Unbiased learning to rank in feeds recommendation. WSDM.
