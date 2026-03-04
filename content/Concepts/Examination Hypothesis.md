---
type: concept
aliases: [click hypothesis, examination assumption]
course: [IR]
tags: [click-models, user-behavior, ranking]
status: complete
---

# Examination Hypothesis

## Definition

The **examination hypothesis** states that:

> A user clicks on an item **if and only if** they examine it **AND** find it relevant.

Formally:

$$P(\text{Click}_{d,k}) = P(\text{Exam}_d | \text{position } k) \cdot P(\text{Rel}_d | q)$$

This is the foundational assumption of [[Click Models]], particularly the [[Position-Based Click Model]].

## Intuition

The hypothesis decomposes the click event into two independent components:

1. **Examination**: Did the user even see the item?
   - Depends on position (users scan top-to-bottom)
   - Depends on context (surrounding items, layout)
   - Can depend on user behavior (e.g., did they find something already?)

2. **Relevance**: Given that they saw it, would they click?
   - Depends on item quality
   - Depends on query intent
   - Independent of presentation (in the hypothesis)

**Visual**: A user must pass both filters to produce a click:

```
User
  ↓
[Examines position k?] ← Position bias
  ↓ Yes
[Relevant to query?] ← True relevance
  ↓ Yes
CLICK
```

## Mathematical Formulation

### Decomposition

Under the examination hypothesis, the joint probability of a click factors:

$$P(\text{Click}_{d,k}) = P(\text{Exam}_k, \text{Rel}_d)$$

By conditional independence (the hypothesis):

$$P(\text{Exam}_k, \text{Rel}_d) = P(\text{Exam}_k) \cdot P(\text{Rel}_d)$$

Note: $\text{Rel}_d$ is independent of $k$ (position).

### Logistic Formulation

In probabilistic models, often expressed:

$$P(\text{Click}_{d,k}) = \sigma(\text{exam}(k) + \text{rel}(d))$$

where $\sigma$ is the sigmoid function, and:
- $\text{exam}(k)$ captures position effects
- $\text{rel}(d)$ captures relevance effects

### In Click Models

The hypothesis enables clean graphical models. For the [[Position-Based Click Model]]:

```
        Exam_k
         /   \
        /     \
       /       \
      ↓         ↓
    Click ← Rel_d
      ↑
    observed
```

The examination node and relevance node conditionally determine the click.

## Implications

### 1. Invertibility of Position Bias

Under the hypothesis, if we know clicks and true relevance, we can back out position bias:

$$P(\text{Exam}_k) = \frac{P(\text{Click}_{d,k})}{P(\text{Rel}_d | q)}$$

This is crucial for [[Inverse Propensity Weighting]].

### 2. No Interaction Effects

The hypothesis assumes **no interaction** between examination and relevance:
- A highly relevant item is equally likely to be clicked at any position (conditional on examination)
- A position-1 item is equally likely to be clicked at any relevance level (conditional on relevance)

In reality, this is violated (e.g., [[Trust Bias]], outlier effects).

### 3. Separability of Factors

Because of independence, we can:
- Estimate position bias from clicks alone (with sufficient diverse data)
- Estimate relevance from data that varies position (e.g., randomized experiments)
- Fit separate models for examination and relevance

## Key Assumptions

### 1. Examination is Binary

An item is either examined or not; there's no "partial examination."

**Violation**: Users might scan an item title (partial exam) without clicking body text.

### 2. Examination Only Depends on Position

$$P(\text{Exam}_{d,k}) = P(\text{Exam}_k)$$

The probability of examining item $d$ at position $k$ is the same for all $d$.

**Violations**:
- [[Cascading Position Bias]]: Examination depends on previous items' relevance
- [[Outlier Bias]]: Distinctive items attract attention regardless of position
- [[Surrounding Item Bias]]: Examination of item $k$ depends on neighbors

### 3. Relevance is Independent of Position

$$P(\text{Rel}_d | q, k) = P(\text{Rel}_d | q)$$

Whether item $d$ is relevant doesn't change based on where it's displayed.

**Violations**:
- [[Trust Bias]]: Items appear more relevant when displayed at high positions
- [[Context Bias]]: Relevance judgment depends on surrounding items

### 4. No Examination = No Click

Users cannot accidentally click unseen items, and there are no random clicks.

**Violations**:
- Misdirected clicks (clicking the wrong item)
- Spam clicks
- Bot activity

## Estimation Under the Hypothesis

### Expectation-Maximization

The hypothesis enables the EM algorithm for [[Click Models]]:

**E-step**: Given current estimates of $\text{exam}_k$ and $\text{rel}_d$, infer the latent examination events from clicks.

**M-step**: Update estimates of $\text{exam}_k$ and $\text{rel}_d$ to maximize data likelihood.

### Probabilistic Inference

With the examination event as a latent variable:

$$P(\text{Exam}_k = 1 | \text{Click} = 1) = \frac{P(\text{Click} = 1 | \text{Exam} = 1) \cdot P(\text{Exam} = 1)}{P(\text{Click} = 1)}$$

This allows inference:
- From clicks → estimates of examination
- From examination → estimates of relevance

## Violations in Practice

### 1. Cascading Behavior

Users examine items sequentially and stop when satisfied:

$$P(\text{Exam}_k) = \prod_{j < k} (1 - P(\text{Rel}_j))$$

**Impact**: IPS using PBM assumptions becomes severely biased.

**Solution**: Use [[Cascading Position Bias]] models instead.

### 2. Trust Bias

Items at top positions get more clicks than justified:

$$P(\text{Click}) = P(\text{Exam}) \cdot [P(\text{Rel}) + (1 - P(\text{Rel})) \cdot \tau_k]$$

Introduces false positives at high ranks.

**Impact**: Learned models overestimate relevance of top-ranked items.

**Solution**: Model trust bias explicitly.

### 3. Context Effects

Examination and relevance are not independent:
- Outlier items attract more attention
- Relevant neighbors make an item harder to notice
- Visual distinctiveness increases examination

**Impact**: Separability assumption fails; click models become less identifiable.

**Solution**: Include context features in the model.

## Identifiability

A critical issue: under the examination hypothesis, **identifiability is not guaranteed**.

For certain data distributions, multiple combinations of $\text{exam}$ and $\text{rel}$ equally well explain the clicks:

```
Data: Clicks on
Search (all items at various positions)

Explanation 1:
  exam = [1.0, 0.9, 0.7, ...]
  rel = [0.8, 0.9, 0.6, ...]
  Likelihood: L₁

Explanation 2:
  exam = [1.0, 1.0, 1.0, ...]
  rel = [0.8, 0.81, 0.42, ...]
  Likelihood: L₁ (same!)

Both explain the data equally well!
```

**Why?** If items always appear at similar positions, we can't distinguish:
- "Item A gets clicks because position 1 is examined" (high exam, low rel)
- "Item A gets clicks because it's relevant" (low exam, high rel)

**Consequence**: Model retraining might converge to different solutions.

## Connections

- **Foundation**: Basis for [[Click Models]], [[Position-Based Click Model]]
- **Violation**: [[Cascading Position Bias]], [[Trust Bias]], [[Outlier Bias]]
- **Estimation**: [[Inverse Propensity Weighting]] assumes hypothesis
- **Alternative**: [[Doubly Robust Estimation]] if hypothesis is violated

## Appears In

- [[Click Models]]
- [[Position-Based Click Model]]
- [[Unbiased Learning to Rank]]
- [[Position Bias]] estimation
