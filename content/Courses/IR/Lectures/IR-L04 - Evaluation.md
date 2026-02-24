---
type: lecture
course: IR
week: 2
lecture: 4
topics:
  - "[[Precision]]"
  - "[[Recall]]"
  - "[[F-Measure]]"
  - "[[MAP]]"
  - "[[NDCG]]"
  - "[[MRR]]"
  - "[[Precision at K]]"
  - "[[Cranfield Paradigm]]"
  - "[[Pooling]]"
status: complete
---

# IR Lecture 4: IR Evaluation

## Overview

IR evaluation measures how effectively a system matches users with relevant information. Since user satisfaction is hard to measure directly, **relevance** is the primary proxy. Evaluation follows the scientific method: design a system, run retrieval, compare results against human judgments.

---

## 1. The Cranfield Paradigm

> [!definition] Cranfield Paradigm
> The standard framework for IR evaluation (Cleverdon, 1960s). Ensures comparability and repeatability using static test collections.

**Components of a test collection:**
1. **Corpus (Documents):** Representative collection of documents
2. **Topics (Queries):** Set of information needs (usually 50+ for statistical significance)
3. **Relevance Judgments (Qrels):** Ground truth — which documents are relevant to which queries

### Depth-k [[Pooling]]

Judging every document in a large corpus for every query is infeasible. Instead:
1. Take the top-$k$ results from multiple different retrieval systems
2. Union these results and remove duplicates
3. Have human judges assess only the pooled documents
4. **Assumption:** Unjudged documents are considered not relevant

> [!warning] Pool Bias
> If a new system retrieves documents no pooled system found, those docs won't have judgments. **Leave-one-out tests**: remove one system from pool, re-evaluate, check if system ranking is stable (using **Kendall's $\tau$** correlation).

---

## 2. Set-Based Metrics

These treat retrieved documents as an unordered set.

> [!formula] Precision and Recall
> $$\text{Precision} = \frac{|\text{Rel} \cap \text{Ret}|}{|\text{Ret}|} = \frac{TP}{TP + FP}$$
> $$\text{Recall} = \frac{|\text{Rel} \cap \text{Ret}|}{|\text{Rel}|} = \frac{TP}{TP + FN}$$
> 
> where $TP$ = relevant retrieved, $FP$ = non-relevant retrieved, $FN$ = relevant not retrieved.

> [!intuition] Precision-Recall Trade-off
> Retrieving more documents increases recall but often decreases precision. The optimal balance depends on the task: legal search needs high recall; web search needs high precision.

> [!formula] F-Measure
> Harmonic mean of Precision and Recall:
> $$F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 \cdot P + R}$$
> 
> - $F_1$: balanced ($\beta = 1$): $F_1 = \frac{2PR}{P + R}$
> - $\beta > 1$: emphasizes recall
> - $\beta < 1$: emphasizes precision

---

## 3. Rank-Based Metrics

IR systems return **ranked** lists — we need metrics that reward relevant documents at the top.

### [[Precision at K]] (P@K)

$$P@K = \frac{\text{relevant documents in top } K}{K}$$

Simple and intuitive. Ignores documents below rank $K$ and doesn't consider order within top $K$.

### [[MAP|Mean Average Precision (MAP)]]

> [!formula] Average Precision (AP)
> $$\text{AP} = \frac{1}{|\text{Rel}|} \sum_{k=1}^{n} P@k \cdot \text{rel}(k)$$
> 
> where $\text{rel}(k) = 1$ if document at rank $k$ is relevant.

$$\text{MAP} = \frac{1}{|Q|} \sum_{j=1}^{|Q|} \text{AP}(q_j)$$

> [!example] Worked Example
> Ranking: [R, N, R, N, R] (R=relevant, N=non-relevant), 3 relevant total.
> - $P@1 = 1/1 = 1.0$ ✓
> - $P@2 = 1/2 = 0.5$ (not relevant, skip)
> - $P@3 = 2/3 = 0.667$ ✓
> - $P@4 = 2/4 = 0.5$ (not relevant, skip)
> - $P@5 = 3/5 = 0.6$ ✓
> 
> $\text{AP} = \frac{1}{3}(1.0 + 0.667 + 0.6) = 0.756$

### [[MRR|Mean Reciprocal Rank (MRR)]]

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Measures where the **first** relevant document appears. Good for navigational queries / QA.

---

## 4. Graded Relevance: DCG and NDCG

Binary relevance (relevant/not relevant) is often too coarse. **Graded relevance** allows degrees: e.g., 0=not relevant, 1=somewhat, 2=highly, 3=perfectly relevant.

### Discounted Cumulative Gain (DCG)

> [!formula] DCG@K
> $$\text{DCG}@K = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$
> 
> where:
> - $\text{rel}_i$ — relevance grade at position $i$
> - $2^{\text{rel}_i} - 1$ — **gain** (exponential: highly relevant docs contribute much more)
> - $\frac{1}{\log_2(i+1)}$ — **discount** (lower positions contribute less)

### Normalized DCG ([[NDCG]])

> [!formula] NDCG@K
> $$\text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$
> 
> where IDCG = DCG of the ideal (perfect) ranking. Normalizes to $[0, 1]$.

> [!example] Worked NDCG Example
> Ranking: [rel=3, rel=2, rel=0, rel=1], $K=4$
> 
> $\text{DCG}@4 = \frac{2^3-1}{\log_2 2} + \frac{2^2-1}{\log_2 3} + \frac{2^0-1}{\log_2 4} + \frac{2^1-1}{\log_2 5}$
> $= \frac{7}{1} + \frac{3}{1.585} + \frac{0}{2} + \frac{1}{2.322} = 7 + 1.893 + 0 + 0.431 = 9.324$
> 
> Ideal: [3, 2, 1, 0]: $\text{IDCG}@4 = 7 + 1.893 + 0.5 + 0 = 9.393$
> 
> $\text{NDCG}@4 = 9.324 / 9.393 = 0.993$

---

## 5. User Browsing Models

More sophisticated metrics model user behavior:

### Rank-Biased Precision (RBP)

> [!formula] RBP
> $$\text{RBP} = (1 - p) \sum_{i=1}^{\infty} p^{i-1} \cdot \text{rel}_i$$
> 
> User views rank 1, continues to next with probability $p$ (persistence). Higher $p$ = more patient user.

### Expected Reciprocal Rank (ERR)

> [!formula] ERR
> $$\text{ERR} = \sum_{r=1}^{n} \frac{1}{r} \prod_{i=1}^{r-1}(1 - R_i) \cdot R_i$$
> 
> where $R_i = \frac{2^{\text{rel}_i} - 1}{2^{\text{rel}_{\max}}}$ is the probability of being satisfied at position $i$.

Models cascade behavior: once a user is satisfied, they stop browsing. A highly relevant document at rank 2 **reduces** the value of a relevant document at rank 3.

---

## 6. Evaluating RAG Systems

For [[Retrieval-Augmented Generation]] systems, evaluation splits into:

**Retrieval component:** Standard IR metrics ([[NDCG]], [[MAP]], [[Recall]])

**Generation component:**
- **Faithfulness / Groundedness:** Is the answer supported by retrieved documents?
- **Answer relevance:** Does the answer address the query?
- **Nugget-based evaluation:** Does the answer contain key information nuggets?
- **LLM-as-a-judge:** Using an LLM to evaluate answer quality

---

## 7. Statistical Significance

> [!warning] Always Test Significance
> A 2% improvement in MAP might be noise. Use statistical tests to confirm results are meaningful.

**Common tests:**
- **Paired t-test:** Compare system A vs system B across queries. $H_0$: no difference.
- **Wilcoxon signed-rank test:** Non-parametric alternative
- **Bootstrap test:** Resample queries, compute metric many times
- **$p < 0.05$**: Standard threshold for significance

**Kendall's $\tau$:** Measures agreement between two system rankings (used for pool reusability).

---

## 8. Summary: Metric Selection Guide

| Scenario | Recommended Metric |
|----------|-------------------|
| Binary relevance, care about top results | [[MAP]] |
| Graded relevance | [[NDCG]] |
| Finding first answer (QA, navigational) | [[MRR]] |
| Quick sanity check | [[Precision at K\|P@10]] |
| User model with persistence | RBP |
| User model with satisfaction | ERR |
