---
type: lecture
course: IR
week: 5
lecture: 10
book_sections: 
  - "LTR for IR 1.2-1.3"
  - "LTR for IR 2-2.2.1"
  - "LTR for IR 2.4.2"
  - "LTR for IR 3"
  - "LTR for IR 4.2"
topics:
  - "[[Learning to Rank]]"
  - "[[Pointwise LTR]]"
  - "[[Pairwise LTR]]"
  - "[[Listwise LTR]]"
  - "[[RankNet]]"
  - "[[LambdaRank]]"
  - "[[ListNet]]"
  - "[[ListMLE]]"
status: complete
---

# Learning to Rank

**Lecturer:** Philipp Hager, University of Amsterdam  
**Date:** March 3rd, 2026

## Overview & Motivation

Modern search engines face a fundamental challenge: how do we combine hundreds or even thousands of signals to rank documents? Airbnb uses >195 features, Bing uses >136 features, Istella uses >220 features, and Yahoo uses >700 features. [[Learning to Rank]] (LTR) is the field that addresses this problem: *automatically constructing a ranking model using training data such that the model can sort new objects according to their degrees of relevance, preference, or importance* (Liu [14]).

Rather than hand-crafting ranking functions, [[Learning to Rank]] treats ranking as a machine learning problem. The key insight is that while we can compute thousands of features for query-document pairs, we need a principled way to combine them into a single ranking. This lecture covers the three main LTR paradigms—**pointwise**, **pairwise**, and **listwise**—and explains why each exists and when to apply them.

---

## Core Concepts: From Signals to Rankings

### Web Search Signals

Search engines leverage multiple categories of signals:

**Query-Document Features:**
- [[TF-IDF]] / [[Vector Space Model]]
- [[BM25]]
- Language models
- Neural semantic matching

**Query-Only Features:**
- Query length
- Query language
- Query type (navigational/informational)
- Search history

**Document-Only Features:**
- Page rank / popularity
- Document length
- Spam detection
- Adult content flags
- Mentioned entities
- URL type
- Last updated

**User/Context Features:**
- User device
- Location
- User device type

The fundamental problem: **How do we combine all these signals into a single ranking?**

### Problem Formulation

The Learning to Rank setup is formalized as:

**Given:**
- Feature vectors for query-document pairs: $\vec{x}_{q,d} \in \mathbb{R}^m$
- Relevance labels: $y_{q,d} \in \{0, 1, 2, 3, 4\}$ (or other ordinal scales)

**Goal:**
- Learn a scoring function $f: \vec{x} \to \mathbb{R}$
- Such that sorting by $f(\vec{x}_{q,d})$ produces the best ranking

**Feature Categories:**
- **Static features**: Document-only and query-only features (unchanged per query)
- **Dynamic features**: Query-document interaction features (computed per query-document pair)

---

## The Main Challenge of Learning to Rank

### Why Can't We Use Ranking Metrics Directly?

The core challenge is that [[Learning to Rank]] metrics like [[NDCG]] and [[MAP]] are **non-differentiable** with respect to model parameters. These metrics rely on the **sorting operation**, which is:
- Non-smooth (sorting operation has no gradient)
- Mostly flat (many models produce the same metric value)
- Discontinuous (small changes in scores don't change the metric)

$$\frac{\partial \text{RR}}{\partial \theta} = \text{???}$$

$$\frac{\partial \text{DCG}}{\partial \theta} = \text{???}$$

> [!warning]
> Ranking metrics produce either zero gradients (flat regions) or discontinuous gradients, making direct optimization impossible with [[Gradient Descent]].

**Solution:** Each LTR family develops a different approximation to overcome this problem.

### Key Ranking Metrics

> [!definition]
> **Reciprocal Rank (RR):** The reciprocal of the rank of the first relevant item:
> $$\text{RR} = \frac{1}{\text{rank}_i}$$
> Higher is better.

> [!definition]
> **Discounted Cumulative Gain (DCG):** The sum of relevance gains discounted by position:
> $$\text{DCG} = \sum_{i=1}^{n} \frac{2^{y_i} - 1}{\log_2(i + 1)}$$
> where $y_i$ is the relevance label at position $i$.

> [!definition]
> **Normalized DCG (NDCG):** DCG divided by the ideal DCG (DCG of a perfect ranking):
> $$\text{NDCG} = \frac{\text{DCG}}{\text{IDCG}}$$

> [!definition]
> **Mean Average Precision (MAP):** The average precision computed over multiple queries.

---

## Pointwise Learning to Rank

### Approach

[[Pointwise LTR]] treats ranking as a **regression or classification problem** on individual query-document pairs, independent of other documents. The model predicts a relevance score for each query-document pair, and documents are ranked by their predicted scores.

> [!intuition]
> The intuition: If we predict the "true" relevance label for each query-document pair, then sorting by predicted score should give us the correct ranking.

### Formulations

**1. Regression (MSE):**

$$L_{\text{mse}} = \sum_{q,d} (y_{q,d} - f(\vec{x}_{q,d}))^2$$

**2. Classification:**

Treat relevance as unordered categories and use cross-entropy loss.

**3. Ordinal Regression:**

Treat relevance as ordered categories (0 < 1 < 2 < 3 < 4).

### Example: Why Lower Loss ≠ Better Ranking

Consider a single query with four documents:

**Scenario 1:** Predictions [0.6, 0.5, 0.5, 0.5] with labels [1, 0, 0, 0]

$$L_{\text{mse}} = (1-0.6)^2 + (0-0.5)^2 + (0-0.5)^2 + (0-0.5)^2 = 0.16 + 0.25 + 0.25 + 0.25 = 0.91$$

Ranking: [doc0 (rel=1), doc1-3 (rel=0)]  
MRR = 1.0 ✓

**Scenario 2:** Predictions [0.2, 0.2, 0.2, 0.1] with labels [1, 0, 0, 0]

$$L_{\text{mse}} = (1-0.2)^2 + (0-0.2)^2 + (0-0.2)^2 + (0-0.1)^2 = 0.64 + 0.04 + 0.04 + 0.01 = 0.73$$

Ranking: [doc0-2 (score 0.2), doc3 (score 0.1)]  
MRR = 0.33 ✗

> [!warning]
> **Fundamental Problem of Pointwise Methods:** A lower pointwise loss does NOT guarantee better ranking, because document scores are **interdependent** in rankings but treated **independently** in the loss.

### Challenges

1. **Class Imbalance**: Irrelevant documents far outnumber relevant ones
2. **Feature Normalization**: Feature distributions vary across queries
3. **Ranking Ignorance**: The loss function ignores that scores are used for sorting

---

## Pairwise Learning to Rank

### Core Idea

[[Pairwise LTR]] shifts the optimization target from individual documents to **pairs of documents**. The intuition: we don't need exact scores, just correct pairwise preferences.

> [!intuition]
> If document $i$ is more relevant than document $j$ ($y_i > y_j$), we want $s_i > s_j$. Minimizing pairwise errors directly addresses the ranking problem.

### General Formulation

For all pairs where $y_i > y_j$:

$$L_{\text{pairwise}}(s, y) = \sum_{y_i > y_j} \varphi(s_i - s_j)$$

where $\varphi$ is a loss function on the score difference.

### RankNet

**RankNet** (Burges et al., 2005) was the first neural ranking model and remains widely used. It uses a neural network to score documents and pairwise comparisons to train it.

#### RankNet Probability Model

RankNet models the probability that document $i$ should rank above document $j$ using the sigmoid function:

$$P(i > j) = \sigma(s_i - s_j) = \frac{1}{1 + e^{-(s_i - s_j)}}$$

> [!formula]
> **Sigmoid mapping:** Maps the score difference to a probability between 0 and 1. The difference $s_i - s_j$ determines how confident we are that $i$ should rank above $j$.

#### Target Probabilities

The target probabilities are derived from relevance labels:
- If $y_i > y_j$: $\bar{P}(i > j) = 1.0$
- If $y_i = y_j$: $\bar{P}(i > j) = 0.5$
- If $y_i < y_j$: $\bar{P}(i > j) = 0.0$

#### RankNet Loss Derivation

Starting from cross-entropy between predicted and target probabilities:

$$L = \sum_{i,j} -\bar{P}_{ij} \log P_{ij} - \bar{P}_{ji} \log P_{ji}$$

For pairs where $y_i > y_j$, the target is $\bar{P}_{ij} = 1, \bar{P}_{ji} = 0$:

$$L = \sum_{y_i > y_j} -\log P_{ij} = \sum_{y_i > y_j} \log(1 + e^{-(s_i - s_j)})$$

> [!formula]
> **RankNet Loss:**
> $$L_{\text{RankNet}}(s, y) = \sum_{y_i > y_j} \log(1 + e^{-(s_i - s_j)})$$
> This is the **logistic loss** on the score difference.

### Other Pairwise Loss Functions

Different pairwise methods use different loss functions $\varphi$:

> [!formula]
> **RankingSVM (Ranking SVM):**
> $$\varphi(z) = \max(0, 1 - z)$$
> Hinge loss with margin 1. Requires $s_i - s_j \geq 1$ when $y_i > y_j$.

> [!formula]
> **RankBoost:**
> $$\varphi(z) = e^{-z}$$
> Exponential loss. Heavily penalizes incorrectly ranked pairs.

> [!formula]
> **RankNet:**
> $$\varphi(z) = \log(1 + e^{-z})$$
> Logistic loss. Smooth and differentiable everywhere.

### The Pairwise Perspective

Pairwise LTR directly minimizes **the number of incorrectly ranked pairs** where $y_i > y_j$ but $s_i < s_j$. This is more aligned with ranking than pointwise regression, but has limitations.

### Key Problem: Not All Pairs Matter Equally

Consider a ranking result: The pairwise loss treats all incorrect pairs equally, but in reality, **top-ranked documents are more important**.

Example: Swapping the #1 and #2 results hurts more than swapping #100 and #101, but pairwise loss treats both as equal.

$$\text{Reducing pairwise errors } 13 \to 11 \text{ can actually DECREASE metrics like MRR and nDCG}$$

> [!warning]
> **Limitation of Pairwise Methods:** Not all pairs contribute equally to ranking quality. Position-sensitive metrics are ignored.

### Why Pairwise Methods Remain Popular

Despite limitations:

1. **Memory Efficiency**: Only two examples needed in memory at a time (crucial for large neural models)
2. **Relative Labels**: Human preferences (pairwise judgments) are often easier to obtain than absolute relevance labels
3. **RLHF Connection**: Pairwise methods naturally extend to [[Reinforcement Learning from Human Feedback]] (RLHF) for LLMs
4. **Contrastive Learning**: Pairwise objectives are fundamental to modern contrastive learning methods

---

## Listwise Learning to Rank

[[Listwise LTR]] considers the **entire ranking** (a list of documents) in the loss function. This directly addresses the ranking problem but is computationally more complex.

### Probabilistic Listwise LTR

#### The Plackett-Luce Model

The Plackett-Luce (PL) model defines the probability of selecting a particular ranking by sequential sampling without replacement:

$$P(d_i) = \frac{e^{s_i}}{\sum_{j=1}^{n} e^{s_j}}$$

This is the softmax function applied to scores.

For a ranking $\pi = (d_2, d_1, d_3)$ with 3 documents, the probability is:

$$P(\pi | s) = \frac{e^{s_2}}{e^{s_1} + e^{s_2} + e^{s_3}} \cdot \frac{e^{s_1}}{e^{s_1} + e^{s_3}} \cdot \frac{e^{s_3}}{e^{s_3}}$$

> [!intuition]
> At each step, we select one document from the remaining documents, with probability proportional to its score (softmax). This generates a distribution over all possible rankings.

> [!warning]
> **Computational Challenge:** The number of possible rankings is $n!$, which grows factorially. Computing the full ranking distribution is infeasible for large $n$.

#### ListNet

**ListNet** (Cao et al., 2007) simplifies the PL model by considering only **top-1 probabilities** instead of full ranking distributions.

The target distribution over documents (based on relevance labels) is:
$$P^*(d_i) = \frac{e^{y_i}}{\sum_{j=1}^{n} e^{y_j}}$$

The predicted distribution is:
$$P(d_i) = \frac{e^{s_i}}{\sum_{j=1}^{n} e^{s_j}}$$

> [!formula]
> **ListNet Loss:**
> $$L_{\text{ListNet}} = -\sum_{i=1}^{n} P^*(d_i) \log P(d_i)$$
> Cross-entropy between the target and predicted softmax over relevance labels and scores.

**Advantage:** Computationally efficient - only computes softmax, not full ranking distribution.

**Limitation:** Only considers top-1 probability, not the full ranking structure.

#### ListMLE

**ListMLE** (Xia et al., 2008) directly maximizes the probability of the **ground-truth permutation** under the PL model.

$$L_{\text{ListMLE}} = -\sum_{i=1}^{n} \log \frac{e^{s_{\pi^*(i)}}}{\sum_{j=i}^{n} e^{s_{\pi^*(j)}}}$$

where $\pi^*(i)$ is the $i$-th element in the ground-truth ranking (in descending order of relevance).

> [!formula]
> **ListMLE Loss Interpretation:** For each position in the ranking, compute the softmax of remaining documents' scores, and maximize the probability of the ground-truth document at that position.

**Advantages:**
- Directly optimizes ranking probability
- Accounts for multiple valid rankings when ties exist in labels

**Computational Complexity:** $O(n \log n)$ for sorting, then $O(n)$ for loss computation.

---

### Metric-Based Listwise LTR

The second approach to listwise LTR directly approximates ranking metrics.

#### LambdaRank

**LambdaRank** (Burges et al., 2006) is a clever technique that scales pairwise gradients by their metric impact.

**Key Observation:**
- Gradient computation doesn't require the loss function itself
- Only the **gradient** with respect to scores matters
- We can scale gradients based on how much swapping two documents changes the ranking metric

$$L_{\text{LambdaRank}}(s, y) = \sum_{y_i > y_j} \log(1 + e^{-(s_i - s_j)}) \cdot \Delta\text{NDCG}(i, j)$$

where $\Delta\text{NDCG}(i, j)$ is the change in NDCG if we swap documents $i$ and $j$.

> [!intuition]
> LambdaRank starts with RankNet gradients (treating pairwise comparisons) but weights them by how much the comparison affects the ranking metric. Pairs that strongly impact NDCG get larger gradients.

**In Practice:** When using multiple additive regression trees (MART), this is called **LambdaMART** and is one of the strongest baseline methods for LTR.

#### ApproxNDCG

**ApproxNDCG** (Qin et al., 2010) directly approximates the ranking function using a sigmoid:

$$\text{rank}(d_i) = 1 + \sum_{j: j \neq i} \mathbb{1}[s_j > s_i] \approx 1 + \sum_{j: j \neq i} \frac{1}{1 + e^{(s_i - s_j)}} = \tilde{\text{rank}}(d_i)$$

The approximate rank is then plugged into DCG:

$$L = -\sum_{d} \frac{2^{\text{label}(d)} - 1}{\log_2(1 + \tilde{\text{rank}}(d))}$$

> [!formula]
> **ApproxNDCG Loss:** Direct differentiable approximation of NDCG using sigmoid smoothing of the ranking function.

**Advantage:** Directly optimizes the metric we care about.

**Limitation:** Assumes metric weights decrease smoothly with rank (not valid for all metrics like fairness metrics).

---

## Key Algorithms & Pseudocode

### RankNet Training Algorithm

```
Algorithm: RankNet Training
Input: Query-document pairs (q, d), features x_{q,d}, labels y_{q,d}
        Neural network f(x; θ) with parameters θ
        Learning rate α
Output: Trained parameters θ

Initialize θ randomly

while not converged:
    for each query q:
        D_q ← candidate documents for query q
        
        for each pair (i, j) where y_i > y_j:
            s_i ← f(x_i; θ)  // score for document i
            s_j ← f(x_j; θ)  // score for document j
            
            // Compute loss
            L ← log(1 + exp(-(s_i - s_j)))
            
            // Compute gradients
            P_ij ← sigmoid(s_i - s_j)
            dL/ds_i ← -(1 - P_ij)
            dL/ds_j ← (1 - P_ij)
            
            // Backpropagate to θ
            dθ ← dθ + dL/ds_i * ds_i/dθ + dL/ds_j * ds_j/dθ
        
        // Update parameters
        θ ← θ - α * dθ

return θ
```

### ListNet Forward Pass

```
Algorithm: ListNet Forward Pass
Input: Relevance labels y for documents 1..n
       Predicted scores s for documents 1..n

// Compute target distribution from labels
P_star ← softmax(y)  // P_star_i = exp(y_i) / sum_j(exp(y_j))

// Compute predicted distribution from scores
P ← softmax(s)  // P_i = exp(s_i) / sum_j(exp(s_j))

// Compute cross-entropy loss
L ← 0
for i in 1..n:
    L ← L - P_star_i * log(P_i)

return L
```

### LambdaRank Gradient Scaling

```
Algorithm: LambdaRank Gradient Computation
Input: Document scores s, relevance labels y, ranking metric M
       Current ranking π sorted by s

// Compute metric importance of each pair
for each pair (i, j) where y_i > y_j:
    // Compute metric change if we swap i and j in current ranking
    π_swap ← swap(π, i, j)
    
    ΔM ← M(π_swap) - M(π)  // metric change
    
    // Gradient scaled by metric impact
    lambda_ij ← log(1 + exp(-(s_i - s_j))) * |ΔM|

// Gradient w.r.t scores
dL/ds_i ← sum over j: lambda_ij * (-dL/d(s_i - s_j))

return dL/ds_i
```

---

## Categorization of LTR Methods: Properties vs. Categories

### Important Caveat

While LTR methods are traditionally categorized as **pointwise, pairwise, or listwise**, this categorization is **not strict**:

> [!warning]
> **There is no consistently-used definition** of these categories across the literature. Some methods are categorized differently by different authors. These are better seen as **properties** (how much context the loss considers) rather than strict categories.

### Properties to Consider

**Scope of the loss:**
- **Pointwise**: Loss considers single documents independently
- **Pairwise**: Loss considers pairs of documents
- **Listwise**: Loss considers entire rankings

**Metric alignment:**
- **Metric-agnostic**: Uses a proxy loss (regression, hinge, logistic)
- **Metric-aligned**: Approximates the target ranking metric

**Scalability:**
- **Memory efficient**: Works with few documents per gradient update (pairwise methods)
- **Computationally intensive**: Requires all documents for each gradient (listwise methods)

---

## Summary & Conclusions

### Pointwise LTR

**Key Idea:** Predict relevance independently for each query-document pair.

**Pros:**
- Simple and intuitive
- Easy to implement

**Cons:**
- Ignores that scores are interdependent
- Lower loss ≠ better ranking
- Doesn't directly optimize ranking quality

### Pairwise LTR

**Key Idea:** Minimize incorrectly ranked document pairs where $y_i > y_j$ but $s_i < s_j$.

**Pros:**
- More aligned with ranking than pointwise
- Memory-efficient for large neural models
- Well-suited to relative preference labels

**Cons:**
- Treats all pairs equally (not position-aware)
- Doesn't directly optimize ranking metrics
- Can degrade top-heavy metrics while improving pairwise accuracy

### Listwise LTR

**Key Idea:** Optimize over entire rankings or ranking metrics.

**Subapproaches:**
- **Probabilistic**: Model distributions over rankings (ListNet, ListMLE)
- **Metric-based**: Directly approximate/optimize ranking metrics (LambdaRank, ApproxNDCG)

**Pros:**
- Directly optimizes ranking quality
- Can be metric-aware

**Cons:**
- Computationally more expensive
- Factorial complexity for full ranking distributions

### The Reality

> [!tip]
> **There is not one LTR method to rule them all.** The choice depends on:
> - Whether you have absolute or relative labels
> - Computational budget (memory, training time)
> - Which ranking metric matters most
> - Whether you need interpretability
> 
> In practice, RankNet and LambdaMART remain strong baselines, while modern approaches blend these ideas with [[Neural Networks]] and [[Transformers]] for neural ranking.

---

## Connections to Modern IR

### Neural Learning to Rank

Modern [[Learning to Rank]] systems combine LTR principles with neural networks:
- **[[Neural Reranking]]**: Apply neural rankers in a multi-stage retrieval system
- **[[Cross-Encoder]]**: End-to-end neural scoring of query-document pairs
- **[[Bi-Encoder]]**: Separate encoding of queries and documents (fast but less precise)
- **[[BERT for IR]]**: Transformer-based ranking models

### Extension to Implicit Feedback

Tomorrow's lecture addresses: **How do we learn from user interactions (clicks, dwell time) instead of labels?** This leads to unbiased learning to rank.

---

## References

[1] Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., & Hullender, G. (2005). Learning to rank using gradient descent. In *ICML* (pp. 89-96).

[2] Cao, Z., Qin, T., Liu, T. Y., Tsai, M. F., & Li, H. (2007). Learning to rank: From pairwise approach to listwise approach. In *ICML* (pp. 129-136).

[3] Chapelle, O., & Chang, Y. (2011). Yahoo! learning to rank challenge overview. In *PMLR*, 14, 1-24.

[4] Cooper, W. S., Gey, F. C., & Dabney, D. P. (1992). Probabilistic retrieval based on staged logistic regression. In *SIGIR* (pp. 198-210).

[5] Cossock, D., & Zhang, T. (2006). Subset ranking using regression. In *COLT* (pp. 605-619).

[6] Crammer, K., & Singer, Y. (2001). Pranking with ranking. In *NIPS*, 14.

[7] Dato, D., MacAvaney, S., Nardini, F. M., Perego, R., & Tonellotto, N. (2022). The istella22 dataset. In *SIGIR* (pp. 3099-3107).

[8] Freund, Y., Iyer, R., Schapire, R. E., & Singer, Y. (2003). An efficient boosting algorithm. *JMLR*, 4, 933-969.

[9] Fuhr, N. (1989). Optimum polynomial retrieval functions. *ACM TOIS*, 7(3), 183-204.

[10] Haldar, M., et al. (2019). Applying deep learning to Airbnb search. In *KDD* (pp. 1927-1935).

[11] Herbrich, R., Graepel, T., & Obermayer, K. (1999). Large margin rank boundaries. In *Advances in Large Margin Classifiers* (pp. 115-132). MIT Press.

[12] Joachims, T. (2002). Optimizing search engines using clickthrough data. In *KDD* (pp. 133-142).

[13] Kenter, T., Borisov, A., Van Gysel, C., Dehghani, M., de Rijke, M., & Mitra, B. (2017). Neural networks for information retrieval. In *SIGIR* (pp. 1403-1406).

[14] Liu, T. Y. (2009). Learning to rank for information retrieval. *Foundations and Trends in IR*, 3(3), 225-331.

[15] Luce, R. D. (2012). *Individual choice behavior: A theoretical analysis*. Courier Corporation.

[16] Nallapati, R. (2004). Discriminative models for information retrieval. In *SIGIR* (pp. 64-71).

[17] Plackett, R. L. (1975). The analysis of permutations. *Journal of the Royal Statistical Society Series C*, 24(2), 193-202.

[18] Qin, T., & Liu, T. Y. (2013). Introducing LETOR 4.0 datasets. *arXiv:1306.2597*.

[19] Qin, T., Liu, T. Y., & Li, H. (2010). A general approximation framework for direct optimization of information retrieval measures. *Information Retrieval*, 13(4), 375-397.

[20] Shashua, A., & Levin, A. (2002). Ranking with large margin principle. In *NIPS* (Vol. 15).

[21] Xia, F., Liu, T. Y., Wang, J., Zhang, W., & Li, H. (2008). Listwise approach to learning to rank. In *ICML* (pp. 1192-1199).
