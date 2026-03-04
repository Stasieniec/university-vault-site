---
type: concept
aliases: [trust position bias, high-position bias]
course: [IR]
tags: [bias, click-models, user-behavior]
status: complete
---

# Trust Bias

## Definition

**Trust Bias** is the tendency of users to click on items at top positions **regardless of their actual relevance**, because they trust search engines to rank relevant items first.

Formally, trust bias creates **false positive clicks**:

$$P(\text{Click}_{d,k}) = P(\text{Exam}_k) \cdot [P(\text{Rel}_d | q) + (1 - P(\text{Rel}_d | q)) \cdot \tau_k]$$

where $\tau_k$ = false positive click rate at rank $k$ (trust bias component).

The second term represents: even irrelevant items get clicked at high positions because users trust them.

## Intuition

### The Phenomenon

User behavior: "The search engine put it first, so it must be relevant. I'll click."

Even if the item is actually irrelevant, the user:
1. Trusts the ranking system
2. Gives the item the benefit of the doubt
3. Clicks to investigate

Result: Clicks conflate (position trust) with (true relevance).

### Visual Example

```
Query: "climate change" in 2010 vs 2024

2010 Search Results:
  1. [Low-quality science denial site] ← User trusts Google → Click
     (But actually irrelevant)
  2. [Scientific consensus article]
  3. [IPCC report]

User clicked position 1 because of trust, not relevance.
```

## Mathematical Formulation

### Affine Click Model

Trust bias is modeled as an affine transformation of relevance:

$$P(\text{Click}_{d,k}) = P(\text{Exam}_k) \cdot [\alpha_k + \beta_k \cdot P(\text{Rel}_d)]$$

where:
- $\alpha_k$ = bias term (false positives)
- $\beta_k$ = relevance weight (relevance scaling)

**Simplification to 2 parameters**:

Often combined:

$$P(\text{Click}_{d,k}) = P(\text{Exam}_k) \cdot [P(\text{Rel}_d) + \gamma_k \cdot (1 - P(\text{Rel}_d))]$$

where $\gamma_k$ = trust bias strength at rank $k$ (typically decreases with rank).

### Typical Trust Bias Pattern

| Rank | $\gamma_k$ (Trust Bias) |
|------|------------------------|
| 1    | 0.3-0.5 (strong)       |
| 2    | 0.2-0.3 (moderate)     |
| 3    | 0.1-0.2 (weak)         |
| 4    | ~0.05 (minimal)        |
| 5+   | ~0 (negligible)        |

High-rank items get "trust bonus" clicks even when irrelevant.

## Contrast with Other Biases

| Bias | Mechanism | Effect |
|------|-----------|--------|
| **Position Bias (PBM)** | Users don't examine low ranks | Low ranks: fewer clicks |
| **Trust Bias** | Users click high ranks more (same relevance) | High ranks: bonus clicks |
| **Cascading** | User stops after relevant item | Lower ranks: fewer exams |
| **Item Selection** | Below-fold items hidden | Below cutoff: zero clicks |

## Empirical Evidence

### Eye Tracking Studies

Research (Joachims et al., 2005):
- Users fixate more on top results
- Click on position 1 before reading it fully
- Shorter dwell times on position 1 (quick trust click)

### Controlled Experiments

Swap identical items between positions:

```
Search 1: [Item A at rank 1, Item B at rank 2]
Search 2: [Item B at rank 1, Item A at rank 2]

Item B clicked more often in Search 2 (same content, different position)
→ Position affects click probability beyond position bias alone
```

### Magnitude

Studies estimate:
- False positive rate at rank 1: 5-15% of relevant-item baseline
- Decreases rapidly with rank
- Varies by query type (navigational vs. informational)

## Consequences for Learning

### Biased Ranking Models

Training a model on clicks without accounting for trust bias:

1. Model learns: "High-rank items are good"
2. Model learns: "Low-quality items at rank 1 are actually good"
3. Model gets stuck, re-ranking quality items down

### Positive Feedback Loop

```
Initial ranking: [A (meh), B (excellent)]

Users see A first:
  → Trust A → Click A
  → System learns: "A is good"
  
Next ranking: [A (meh), B (excellent)]
  → Same problem repeats
```

The ranking system learns from biased clicks and perpetuates bias.

## Estimation

### RegressionEM with Trust Bias

Extend [[Click Models]] to include trust parameters:

**E-step**: Given current trust estimates, infer true relevance from clicks.

**M-step**: Fit:
- Relevance regression model: $P(\text{Rel}_d) \sim f(x_d)$
- Trust parameters: $\gamma_k$ per rank

### Identification Challenges

Problem: Hard to distinguish:
- Truly relevant item at rank 2 (low clicks due to low exam)
- Irrelevant item at rank 1 (gets clicks due to trust)

**Can we identify both**?

Requires:
- Items at different ranks for same query (variation)
- Relevance judgments or diversity (ground truth or distribution shift)
- Multiple rankers showing different items at different positions

### Trust Bias Estimator

From Vardasbi et al. (2020):

Using items that appear in multiple positions:

$$\hat{\gamma}_k = \frac{P(\text{Click}_k) - P(\text{Click}_{k'})}{1 - P(\text{Rel})}$$

(Approximate)

## Corrections

### Solution 1: Model Trust Bias Explicitly

Include trust parameters in click model:

$$\hat{r}_d = \text{infer from clicks, accounting for } \tau_k$$

Then train on unbiased relevance estimates.

### Solution 2: Online Randomization

Show relevant items at various positions:

$$\text{Clicks on relevant items at rank 1 vs. rank 3} \xrightarrow{\text{different}} \text{infer trust bias}$$

### Solution 3: Doubly Robust Estimation

Combine a learned model with [[Inverse Propensity Weighting]]:

$$\text{DR} = \text{model prediction} - \text{IPS correction}$$

If model is well-trained, corrections are small.

### Solution 4: Query-Level Adjustments

Manually review and adjust top results:
- **Curate high-rank items carefully** (quality > position rank)
- **Demote misleading items** explicitly
- **Update trust calibration** over time

## Real-World Examples

### Example 1: Search Engine Pollution

```
Query: "diet pills"
Rank 1: [Fake pill advertiser site]
         (No pharmaceutical basis, gets clicks due to trust)
Rank 2: [Medical journal article]
         (Actually relevant, lower clicks)

Trust bias causes harm by promoting low-quality results.
```

### Example 2: Recommendation Systems

```
Netflix recommendations (top-to-bottom):
  1. [Trending movie, not in user's interest]
     (User clicks due to position trust)
  2. [Perfect match for user preferences]
     (Lower click due to lower exposure)

System learns: "Trending is good for this user" (wrong)
```

### Example 3: E-commerce

```
Product search "running shoes"
  1. [Popular brand, expensive, not best for runner type]
  2. [Best performance review, specialized features]

Trust bias: User clicks 1 due to prominence (Amazon ranks it first).
Relevance: User would prefer 2 if they read both.
```

## Variants & Extensions

### Context-Dependent Trust

Trust varies by:
- **Domain**: Search engine brand trust differs
- **Query type**: Navigational queries → lower trust bias (confident intention)
- **User expertise**: Experts trust less, novices trust more

### Interaction Effects

Trust bias can **interact** with other factors:
- High-quality item at rank 1: Trust amplifies relevance
- Low-quality item at rank 1: Trust creates false positive

## Measurement & Audit

### How to Detect Trust Bias in Your System

1. **Position swap experiments**: Swap identical items between ranks → measure click difference
2. **Eye tracking**: Where do users look before clicking?
3. **Dwell time analysis**: Do high-rank clicks have shorter dwell times?
4. **Click-to-purchase ratio**: Do trust clicks convert?

### Quantifying Impact

```
Observed CTR(rank 1): 30%
Expected CTR(rank 1) with PBM only: 20%
Trust bias effect: 10% (or ~0.3 false positive rate)
```

## Ethical Considerations

### Harm from Trust Bias

- **Misinformation amplification**: False or misleading items promoted
- **Reduced visibility**: High-quality results get lower attention
- **User manipulation**: Users click untrustworthy items
- **Feedback loops**: System learns from biased clicks, perpetuates bias

### Mitigation

- **Explicit quality signals**: Show confidence scores, sources
- **Diverse presentation**: Don't always rely on top-k ranking
- **User education**: Teach users to evaluate sources
- **Manual curation**: Review and adjust top results

## Connections

- **Related bias**: [[Position Bias]] (different mechanism)
- **Alternative**: [[Cascading Position Bias]] (different user model)
- **Estimation**: [[Click Models]], RegressionEM
- **Correction**: [[Inverse Propensity Weighting]], [[Doubly Robust Estimation]]
- **Framework**: [[Examination Hypothesis]]

## Appears In

- [[Unbiased Learning to Rank]]
- [[Click Models]]
- [[Counterfactual Learning to Rank]]
- Search engine and recommendation system research
