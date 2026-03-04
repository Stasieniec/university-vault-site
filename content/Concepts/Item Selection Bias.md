---
type: concept
aliases: [top-k bias, item cutoff bias, display limitation]
course: [IR]
tags: [bias, ranking, position-bias]
status: complete
---

# Item Selection Bias

## Definition

**Item Selection Bias** occurs when only a subset of items (e.g., top-k results) are displayed, making items outside the displayed set have **zero examination probability**.

Formally:

$$P(\text{Exam}_k) = \begin{cases} > 0 & \text{if } k \leq k_{\max} \\ 0 & \text{if } k > k_{\max} \end{cases}$$

where $k_{\max}$ is the cutoff (e.g., 10 results per page).

## Intuition

### The Problem

In a top-k ranking display:
- Users can only see and interact with items at ranks $1, ..., k$
- Items at ranks $k+1, k+2, ...$ are completely hidden
- Hidden items have **zero probability of examination** and **zero probability of click**

**Consequence**: We have no evidence whether hidden items are relevant or not.

### Visual Example

```
Page 1: Displaying top 10 results
[1] Item A  ← Examined
[2] Item B  ← Examined
[3] Item C  ← Examined
...
[10] Item J ← Examined

[11] Item K ← NOT visible, zero examination
[12] Item L ← NOT visible, zero examination
...

Users would have to scroll/click "next page" to see item K.
```

### Extreme Form of Position Bias

Item selection bias is **position bias taken to the extreme**:

| Bias Type | Exam(rank 5) | Exam(rank 10) | Exam(rank 11) |
|-----------|--------------|---------------|---------------|
| **Position Bias** | ~70% | ~40% | ~35% |
| **Item Selection** | ~70% | ~40% | **0%** |

With item selection bias, there's a **hard cutoff** at $k_{\max}$.

## Mathematical Formulation

### Click Distribution Under Item Selection Bias

For items within display:
$$P(\text{Click}_{d,k}) = P(\text{Exam}_k) \cdot P(\text{Rel}_d)$$

where $P(\text{Exam}_k) > 0$.

For items outside display:
$$P(\text{Click}_{d,k}) = 0$$

for all $k > k_{\max}$.

### Selection Mechanism

The selection is **deterministic**:
- If $k \leq k_{\max}$: item is shown
- If $k > k_{\max}$: item is hidden

This is NOT random; it's a hard constraint based on the display mechanism.

## Why It's a Problem

### IPS Breaks Down

Standard [[Inverse Propensity Weighting]]:

$$\text{IPS} = \sum_k \frac{\text{Click}_k}{P(\text{Exam}_k)}$$

For hidden items:
$$P(\text{Exam}_k) = 0 \Rightarrow \frac{\text{Click}_k}{0} = \text{undefined}$$

Even if an item is truly relevant, we can never observe a click. IPS cannot estimate its relevance.

### Overlap Violation

A key assumption for IPS is **positivity** (or overlap):

$$P(\text{Exam}_k) > 0 \quad \forall k$$

Item selection bias **violates** this assumption.

### No Counterfactual Evidence

We cannot evaluate how a new policy would perform on hidden items because:
- We have no interaction data
- We can't know if they would be clicked
- They might be highly relevant but unrepresented in logs

## Example: Impact on Learning

### Scenario

```
Logging policy: Shows top 10 results (items 1-10)

Hypothetical new policy: Would rank
[A, X, B, C, ...] 

where X is an item that was hidden (rank > 10)

Question: Would X get clicks?
Answer: No data! Can't know.

Naive IPS: Assign X zero relevance (wrong!)
Correct view: X's relevance is unknown.
```

## Practical Settings

### Common in Practice

1. **Web search**: First page shows 10 results, rest paginated
2. **E-commerce**: Product listings show 20-50 items, pagination for more
3. **Recommendations**: Feed shows 10-20 items, "load more" for rest
4. **Ads**: Top 3-5 ad slots visible, rest below fold

### Severity

Depends on search behavior:
- **Navigational queries** ("find www.amazon.com"): Results 2-10 rarely clicked, item selection bias less harmful
- **Ambiguous queries** ("books"): Many items relevant, selection bias hides good results
- **Tail queries** ("obscure topic"): Limited results, selection bias less relevant

## Solutions

### Solution 1: Stochastic Policies

Show every item with some probability, avoiding hard cutoff:

```
Instead of: Top 10 deterministically

Use: Every item has probability p(k) > 0
  p(1) = 0.9 (show with high prob)
  p(2) = 0.9
  ...
  p(10) = 0.5
  p(11) = 0.1 (show occasionally)
  p(12) = 0.01 (show rarely)
```

**Advantage**: Every item visible with non-zero probability  
**Disadvantage**: Harms user experience (shows worse results)

### Solution 2: Inverse Propensity Weighting with Stochastic Policy

If every item has non-zero visibility probability:

$$P(\text{Exam}_k) > 0 \quad \forall k$$

Then standard IPS works:

$$\text{IPS} = \sum_k \frac{\text{Click}_k}{P(\text{Exam}_k)}$$

**Advantage**: Theoretically sound  
**Disadvantage**: User experience cost

### Solution 3: Doubly Robust Estimation

Use the direct method to estimate relevance for unseen items:

$$\text{DR} = \underbrace{\sum_k \hat{r}_{y[k]}}_{\text{For all items, including hidden}}  - \underbrace{\sum_{k \leq k_{\max}} \frac{(\hat{r}_{y[k]} - \text{Click}_k)}{P(\text{Exam}_k)}}_{\text{IPS correction for observed}}$$

**Advantage**: Handles unseen items via learned model  
**Disadvantage**: Depends on model quality

### Solution 4: Pagination & Multi-Step Exposure

Randomize pagination:

```
Policy A: Top 10 on page 1
Policy B: Items 5-14 on page 1 (offset)
Policy C: Random subset of size 10 on page 1

Collect logs from all → estimate relative propensities
```

**Advantage**: Offline data collection  
**Disadvantage**: Complexity in logging

### Solution 5: Historical Exploration

Use past logs where different systems showed different items:

```
System A: Ranked [A, B, C, D, ...]
System B: Ranked [X, A, C, B, ...]

Item X appeared at position 1 in System B
Item X appeared at position > 10 in System A (not visible)

Use System B data to infer X's relevance
```

**Advantage**: Non-intrusive  
**Disadvantage**: Requires historical diversity

## Identification & Off-Policy Learning

### Can We Infer Hidden Items?

**Difficult question**: Without any interaction data, can we estimate hidden items' relevance?

**Answer**: Only with strong modeling assumptions.

### Extrapolation via Features

If we have item features, we can train a relevance model:

$$\hat{r}_d = f(x_d; \theta)$$

For hidden items:
$$\hat{r}_{hidden} = f(x_{hidden}; \theta)$$

**Assumption**: Feature space generalizes beyond visible items.

**Risk**: If hidden items have unusual feature distributions, extrapolation fails.

## Real-World Implications

### Search Engine Perspective

Web search typically shows 10 results per page:

```
Visible: Results 1-10
Hidden: Results 11-100+

Learning from clicks:
- Results 11+ get zero training signal
- New ranker can't be better than showing top 10
- Innovation limited by what's visible
```

### E-commerce Perspective

Product search shows 24 items per page:

```
Visible: First 24 products
Hidden: 25+

Click data trains on visible products only.
New ranking might prefer items 25-50,
but no evidence to learn from.
```

### Recommendation Systems

Feed shows 10 items before scrolling:

```
Visible: Items 1-10 on screen
Hidden: Items 11+ below fold

Cascade effect: Users don't scroll, so 11+ get zero clicks.
But zero clicks ≠ zero relevance.
```

## Mitigation Strategies

### Strategy 1: Design for Exploration

Show diverse items in top-k:

```
Instead of: Pure ranking [A, B, C, ...]
Use: Diverse ranking [A, X (different), B, Y (different), C, ...]

Expose "hidden" good items within top-k.
```

### Strategy 2: Progressive Disclosure

Allow easy access to more items:

```
Default: Show 10
User can "expand" to see 20, 50, 100
Track clicks at different expansion levels
→ Less selection bias in deeper pages
```

### Strategy 3: Bandit Algorithms

Balance exploitation (show best items) with exploration (show new items):

$$P(\text{show item } k) \propto \text{predicted relevance} + \epsilon \cdot \text{exploration bonus}$$

### Strategy 4: Model-Based Augmentation

Combine click data with learned model:

```
Train relevance model on visible items
Extrapolate to hidden items via features
Use DR estimation to correct

More robust than pure clicks.
```

## Connections

- **Generalization**: Extreme case of [[Position Bias]]
- **Related**: [[Overlap]] assumption in causal inference
- **Solution**: [[Doubly Robust Estimation]]
- **Related problem**: [[Cascading Position Bias]] (stops early)

## Appears In

- [[Unbiased Learning to Rank]]
- Search ranking systems
- E-commerce search
- Recommendation systems
- Production ML systems with limited display slots
