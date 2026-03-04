---
type: concept
aliases: [counterfactual LTR, offline evaluation, counterfactual evaluation]
course: [IR]
tags: [learning-to-rank, causal-inference, offline-evaluation]
status: complete
---

# Counterfactual Learning to Rank

## Definition

**Counterfactual Learning to Rank (CLTR)** is the problem of training a ranking model using interaction data (clicks, conversions, etc.) from a *different* ranking system (the logging policy), while accounting for the biases introduced by that different system.

**Motivation**: We want to evaluate and improve a new ranking policy using historical data collected under an old (possibly inferior) policy.

```
Old Policy (Logging Policy)
    ↓
  (logs interactions)
    ↓
New Policy (Counterfactual)
    ↓
  (evaluate/train using old data)
```

The "counterfactual" aspect: We're asking "what would happen if we deployed the *new* policy?" using data from when the *old* policy was active.

## Intuition

### The Problem

Imagine a search engine:
- **Current ranking system**: Ranks by PageRank
- **New ranking system**: Ranks by neural learned model
- **Historical logs**: Clicks collected under PageRank (biased by it)
- **Goal**: Can we use PageRank clicks to evaluate/train the neural model?

**Naive approach**: Train directly on the clicks  
**Result**: The neural model learns that "PageRank-favored items are good," not that "these items are truly relevant"

**Better approach**: Use the fact that clicks are biased by position, correct for that bias using [[Inverse Propensity Weighting]], and train an unbiased model.

### Core Insight

The key is to separate:
- **Click data** = function of (old policy's ranking, true relevance, [[Position Bias]])
- **True relevance** = what we want to learn
- **Old policy's ranking** = what we want to correct for

By modeling and removing the effect of the old policy + position bias, we recover true relevance.

## Mathematical Framework

### The Counterfactual Setup

**Available data**: Interactions from logging policy $\pi_0$

$$D = \{\text{query}_q, \text{ranking}_{y_0}, \text{clicks}_c\}_{\text{episodes}}$$

**Goal**: Train a new policy $\pi_1$

$$\max_\theta \mathbb{E}_{q}[\text{reward}_q(\pi_1(\theta))]$$

**Challenge**: We only have data from $\pi_0$, not $\pi_1$.

### Unbiased Evaluation via IPS

To evaluate $\pi_1$ using data from $\pi_0$:

$$\text{DCG}_{\pi_1} = \mathbb{E}_{(q,y_0,c) \sim D}\left[\sum_{k} \frac{\text{Click}_{y_1[k]}}{P(\text{Exam}_k | y_0)} \cdot \mathbb{1}[y_1[k] \text{ appeared in } y_0]\right]$$

Where:
- $y_0$ = ranking shown under $\pi_0$
- $y_1$ = ranking from new policy $\pi_1$
- $\text{Click}_{y_0[k]}$ = click observed in logs
- $P(\text{Exam}_k | y_0)$ = examination propensity under $\pi_0$'s ranking

The indicator $\mathbb{1}[y_1[k] \text{ appeared in } y_0]$ ensures we only evaluate on items we have data for.

### Doubly Robust Formulation

Combine a learned relevance model $\hat{r}_d$ with [[Inverse Propensity Weighting]]:

$$\text{DCG}_{\text{DR}} = \mathbb{E}\left[\sum_k \hat{r}_{y_1[k]} - \sum_k \frac{\hat{r}_{y_0[k]} - \text{Click}_{y_0[k]}}{P(\text{Exam}_k)}\right]$$

First term: Direct prediction for all items  
Second term: IPS correction on prediction errors  

Result: Unbiased if either the relevance model OR the propensities are correct.

## Key Problem: Off-Policy Bias

### The Mismatch

```
Data Generation (Logging Policy π₀)
    Query q
      ↓
    Ranking y₀ ← π₀(q)
      ↓
    Position Bias: Exam_k
      ↓
    True Relevance: Rel_d
      ↓
    Click: Exam_k · Rel_d
      ↓
    Historical Log

---

Evaluation (New Policy π₁)
    Query q
      ↓
    Ranking y₁ ← π₁(q)
      ↓
    We want to know: Reward(y₁)
    But we have: Data from y₀!
```

If $y_0$ and $y_1$ are different, we need to account for:
1. **Position bias in $y_0$**: Items at high positions in $y_0$ got more clicks
2. **Different positions in $y_1$**: Items appear at different positions in $y_1$
3. **Items not in $y_0$**: If $y_1$ contains items not in $y_0$, we have no direct data

### Selection Bias

Even if we correctly estimate relevance, we need to account for the fact that **the new policy makes different choices**:

- Old policy ranked [A, B, C, D, E]
- New policy ranks [B, A, E, D, C]
- Clicks on A in position 1 (old) don't directly tell us about A in position 2 (new)

This is [[Inverse Propensity Weighting]]: we need position bias estimates for the *new* ranking too.

## Solutions

### Solution 1: Online Randomization

The most direct approach: deploy the new policy online with some randomization, collect new data, and train.

**Advantages**:
- Gets true data under the new policy
- No need for complex counterfactual machinery

**Disadvantages**:
- Requires online deployment (risky, resource-intensive)
- Harms user experience during collection
- Slow feedback loop

### Solution 2: Offline Evaluation with IPS

Use historical logs with careful IPS weighting:

1. **Estimate position bias** from historical logs (or randomization data)
2. **Correct for bias** using IPS: $\widehat{\text{Rel}}_d = \sum_k \text{Click}_{d,k} / P(\text{Exam}_k)$
3. **Train new model** on unbiased relevance estimates
4. **Evaluate** on held-out test set using IPS

**Advantages**:
- Offline, no user impact
- Theoretically sound

**Disadvantages**:
- High variance (due to IPS)
- Depends on accurate propensity estimation
- Fails for zero-propensity items

### Solution 3: Doubly Robust Estimation

Combine IPS with a learned relevance model:

1. **Estimate position bias** from logs
2. **Train relevance model** via click models or [[Doubly Robust Estimation]]
3. **Evaluate new policy** using DR formula
4. **Train new model** using evaluation signal

**Advantages**:
- Lower variance than pure IPS
- More robust to propensity errors
- Practical and scalable

**Disadvantages**:
- Requires learning two models
- Still vulnerable to model misspecification

## Practical Considerations

### Propensity Estimation

**Critical step**: Accurately estimate $P(\text{Exam}_k)$ for the logging policy.

Methods:
- **Online randomization** (Ideal): Run RandTop-k or RandPair experiments
- **Intervention harvesting** (Good): Use historical A/B tests
- **Click model** (Reasonable): Fit EM-based [[Click Models]]

### Handling Unseen Items

Items in $y_1$ that never appeared in $y_0$ have **zero propensity**.

**Problem**: IPS weight is $1/0 = \infty$  
**Solutions**:
- Use DR estimation (DM provides estimates for unseen items)
- Restrict to items seen in logs
- Use item features to extrapolate relevance

### Multi-Step Deployment

In practice, counterfactual learning is often an iterative process:

1. Collect baseline logs under policy $\pi_0$
2. Develop new policy $\pi_1$ using CLTR
3. Deploy $\pi_1$ with some randomization
4. Collect new logs under $\pi_1$
5. Use these new logs for CLTR to develop $\pi_2$
6. Repeat

Each step provides fresh, less-biased data for the next iteration.

## Assumptions & Failures

### Critical Assumptions

1. **Correct user model**: Users behave as modeled (e.g., PBM)
2. **Stable relevance**: Item relevance doesn't change between logging and new policy
3. **Overlap**: Items in new policy were seen in logs (or extrapolatable)
4. **No hidden confounders**: Position is the only confounder

### Common Failures

- **Wrong user model**: Using PBM when cascading behavior dominates
- **Temporal drift**: Item relevance changes between logs and deployment
- **Distribution shift**: New policy queries/items are different from logged distribution
- **Cascading bias**: IPS assuming PBM breaks under cascading

## Real-World Example

**Scenario**: E-commerce search engine

```
Old System (Logging Policy)
    ↓
  Ranks by: BM25 + manual rules
    ↓
  Logs: 1M clicks on various queries
    ↓
  Observed patterns:
    - Items at rank 1 get ~20% CTR
    - Items at rank 5 get ~8% CTR
    - Position bias is strong
    ↓
  Estimate: Exam probabilities [1.0, 1.0, 0.9, 0.8, 0.7, ...]
    ↓

New System (Counterfactual)
    ↓
  Train neural ranker using:
    - Raw clicks (naive): Learns position bias ❌
    - IPS-corrected clicks: Learns true relevance ✓
    ↓
  Evaluate using DR:
    - Relevance model from neural ranker
    - IPS correction on errors
    - Estimate: New policy would improve DCG by X%
    ↓
  Deploy & measure: +X% in online A/B test
```

## Connections

- **Foundation**: [[Inverse Propensity Weighting]] for bias correction
- **Enhancement**: [[Doubly Robust Estimation]] for lower variance
- **User model**: [[Position Bias]], [[Click Models]]
- **Application**: [[Unbiased Learning to Rank]]
- **Causal frame**: Off-policy evaluation in reinforcement learning

## Appears In

- [[Learning to Rank]] (offline evaluation)
- [[Unbiased Learning to Rank]] (core concept)
- [[Information Retrieval]] (production systems)
