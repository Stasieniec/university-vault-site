---
type: concept
aliases: [cascade model, cascading bias, dependent click model]
course: [IR]
tags: [click-models, position-bias, user-behavior]
status: complete
---

# Cascading Position Bias

## Definition

**Cascading Position Bias** is a refinement of [[Position Bias]] that models more realistic user behavior: users examine items sequentially (top-to-bottom) and the probability of examining an item at position $k$ depends on whether they found satisfactory items at positions $1, ..., k-1$.

Formally:

$$P(\text{Exam}_k) = \prod_{j=1}^{k-1} (1 - P(\text{Rel}_j | q))$$

Examination at position $k$ = **product of irrelevance of all previous items**.

## Intuition

### Sequential Scanning

Real users don't independently evaluate every item. They scan top-to-bottom and **stop** when satisfied:

```
User sees ranking: [A, B, C, D, E]

Step 1: Examine A
        Is A relevant? 
        YES → Click & Leave
        NO → Continue

Step 2: Examine B (only if A was irrelevant)
        Is B relevant?
        YES → Click & Leave
        NO → Continue

Step 3: Examine C (only if A & B irrelevant)
        ...
```

This creates **dependence**: whether you examine item $k$ depends on items $1, ..., k-1$.

### Contrast with Position-Based Model

**PBM** (Position-Based Model):
- Examination probability is independent of item relevance
- $P(\text{Exam}_k) = 0.8$ regardless of what's above

```
Even if A, B, C are all irrelevant,
C still has 0.8 chance of examination
```

**Cascade Model**:
- Examination probability depends on previous items
- If A & B are relevant, user stops; C never examined
- If A & B are irrelevant, C has high exam probability

## Mathematical Formulation

### The Cascade Model (CM)

Click probability under cascading:

$$P(\text{Click}_{d,k}) = P(\text{Exam}_k) \cdot P(\text{Rel}_d | q)$$

where:

$$P(\text{Exam}_k) = \prod_{j=1}^{k-1} (1 - P(\text{Rel}_j | q))$$

**Key**: Examination at position $k$ is **item-specific and ranking-specific** (not just position-specific).

### Example

Ranking: [A, B, C, D]  
True relevances: $P(\text{Rel}_A | q) = 0.8$, $P(\text{Rel}_B | q) = 0.6$, $P(\text{Rel}_C | q) = 0.7$, $P(\text{Rel}_D | q) = 0.4$

**Examination probabilities**:
- $P(\text{Exam}_A) = 1.0$ (always examined first)
- $P(\text{Exam}_B) = 1 - 0.8 = 0.2$ (20% reach B)
- $P(\text{Exam}_C) = (1 - 0.8)(1 - 0.6) = 0.2 \times 0.4 = 0.08$ (8% reach C)
- $P(\text{Exam}_D) = 0.2 \times 0.4 \times 0.3 = 0.024$ (2.4% reach D)

**Click probabilities**:
- $P(\text{Click}_A) = 1.0 \times 0.8 = 0.80$
- $P(\text{Click}_B) = 0.2 \times 0.6 = 0.12$
- $P(\text{Click}_C) = 0.08 \times 0.7 = 0.056$
- $P(\text{Click}_D) = 0.024 \times 0.4 = 0.0096$

**Note**: Items lower in ranking have much lower click probabilities because of cascading, not just position bias.

### Dependent Click Model (DCM)

A specific cascade model parameterization:

$$P(\text{Click}_{d,k}) = P(\text{Exam}_k) \cdot P(\text{Rel}_d | q) \cdot P(\text{Satisfy}_d | q)$$

Additional factor:
- $P(\text{Satisfy}_d | q)$ = probability the user is satisfied after clicking (decides to leave)

Parameter $\lambda_k$ = rank-dependent satisfaction probability.

### Dynamic Bayesian Network (DBN) Model

Another cascade variant:

$$P(\text{Exam}_k | \text{history}) = P(\text{Exam}_{k-1}) \cdot (1 - \gamma)$$

where:
- $\gamma$ = probability of session abandonment (user leaves without finding anything)

## Key Differences from PBM

| Aspect | PBM | Cascade Model |
|--------|-----|---------------|
| **Exam depends on position** | Yes | Yes |
| **Exam depends on relevance of previous items** | No | **Yes** |
| **Ranking-specific exams** | No | **Yes** |
| **Identifiability** | Easier | Harder |
| **Empirical fit** | Good for some queries | Better overall |
| **Computational complexity** | Low | Higher |

## Empirical Evidence

### When Does Cascading Dominate?

Cascading behavior is observed more in:
- **Navigational queries**: "find this specific item" → user stops once found
- **Informational queries in some contexts**: "learn about X" → user might scan exhaustively

Less pronounced in:
- **Exploratory queries**: "show me options" → user might browse many items

### Empirical Studies

Research shows:
1. **Navigational queries**: Cascade model fit better than PBM
2. **Commercial queries**: Mixed—sometimes cascade, sometimes PBM dominates
3. **Informational queries**: Varies by context and result quality

## Challenges: Session-Dependent Propensities

The **critical problem**: Under cascading, examination propensities are **ranking-specific**.

$$P(\text{Exam}_k | \text{ranking } y)$$

depends on $y$, not just on $k$.

### Example Problem

```
Scenario 1: Ranking [Relevant, Irrelevant, ...]
  P(Exam_2) = 1 - P(Rel_1) = small (user likely satisfied)

Scenario 2: Ranking [Irrelevant, Relevant, ...]
  P(Exam_2) = 1 - P(Rel_1) = large (user continues scanning)

Same position (position 2), but very different examination probability!
```

This breaks the standard IPS formula which assumes position-specific propensities.

### Solution: Session-Dependent Probabilities

Use clicks in the current session to estimate propensities:

$$\hat{P}(\text{Exam}_k | \text{session}) = \frac{\#\text{seen item } k}{\#\text{sessions where user reached } k}$$

This requires:
- Logging which positions users examined
- Per-session adaptation
- More data per session

## Cascade Model Estimation

### EM for Cascade Models

Classic approach: Expectation-Maximization with latent examination variables.

**E-step**: Given current parameters, infer which items were examined.

**M-step**: Update parameters to maximize likelihood.

**Complication**: Must jointly estimate relevance **and** satisfaction parameters.

### RegressionEM with Cascade

Extend cascade model to use features:

$$P(\text{Rel}_d | q) = \text{sigmoid}(f(x_d, x_q; \theta))$$

Fits regression model to cascade-inferred relevance.

## IPS Breaks Down with Cascading

### The Problem

If you use PBM-based IPS on cascade-generated data:

$$\text{IPS}_{\text{PBM}} = \sum_k \frac{\text{Click}_k}{P(\text{Exam}_k^{\text{PBM}})}$$

where $P(\text{Exam}_k^{\text{PBM})}}$ is position-specific (ignoring cascading):

**Result**: Severe bias.

### Why?

- PBM assumes high examination at rank 3
- Cascade model implies low examination at rank 3 (if ranks 1-2 were relevant)
- IPS weights rank-3 clicks too lightly
- Underestimates relevance of rank-3 items

### Empirical Impact

Studies show:
- IPS with PBM on cascade data: **biased, noisy estimates**
- Specially designed cascade-based IPS: **better, but still challenging**

## Solutions for Cascading

### Solution 1: Cascade-Based Propensity Estimation

Estimate propensities accounting for cascading:

$$P(\text{Exam}_k | \text{ranking}) = \prod_{j < k} (1 - \hat{P}(\text{Rel}_j))$$

Then apply IPS with cascade propensities.

**Advantage**: Theoretically sound  
**Disadvantage**: Requires accurate relevance estimation first (circular dependency)

### Solution 2: Click Models Instead of IPS

Fit a cascade-based click model directly:

$$\max_\theta \sum_{\text{logs}} \log P(\text{clicks} | \text{cascade model}(\theta))$$

**Advantage**: Avoids IPS variance  
**Disadvantage**: Identifiability issues (multiple solutions possible)

### Solution 3: Doubly Robust with Cascade Model

Combine cascade-based DM + IPS:

$$\text{DR} = \text{DM}_{\text{cascade}} - \text{IPS}_{\text{cascade correction}}$$

**Advantage**: Low variance + unbiased if either component correct  
**Disadvantage**: Complex to implement

### Solution 4: Online Randomization

The nuclear option: randomize with random ranking probabilities to break cascading.

$$P(\text{Exam}_k) = \text{known}$$ (not confounded by previous relevance)

**Advantage**: Clean propensity estimates  
**Disadvantage**: Harms user experience; requires new data collection

## Real-World Implications

### When You Should Care

- **Web search**: Moderate cascading (users often scan multiple results)
- **E-commerce search**: Strong cascading (users stop after finding good product)
- **Recommendations**: Moderate (depends on context)
- **Ads**: Less cascading (users might ignore ad regardless)

### When You Can Ignore It

- Item-level feedback (not position-based)
- Conversion data (not click data)
- Systems where users always examine all items

## Related Models

### Variants of Cascade

1. **Dependent Click Model**: Adds satisfaction probability
2. **Dynamic Bayesian Network**: Adds abandonment probability
3. **Cascade with attractiveness**: Items attract examination independent of position

### Different User Behaviors

- [[Trust Bias]]: Position affects relevance perception (different from cascade)
- [[Item Selection Bias]]: Hard cutoff (items below fold never seen)
- [[Outlier Bias]]: Distinctive items break cascade (attract examination out of order)

## Connections

- **Extends**: [[Position Bias]] (PBM is a special case)
- **Alternative**: [[Click Models]] provide joint estimation
- **Related**: [[Examination Hypothesis]] (same assumption but different model)
- **Impact on**: [[Inverse Propensity Weighting]] (breaks standard IPS)
- **Solution**: [[Doubly Robust Estimation]], cascade-based click models

## Appears In

- [[Click Models]]
- [[Unbiased Learning to Rank]]
- User behavior modeling in [[Information Retrieval]]
