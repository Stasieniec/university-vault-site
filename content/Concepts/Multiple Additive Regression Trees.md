---
type: concept
aliases: [MART]
course: [IR]
tags: [ltr, ranking, gradient-boosting, exam-topic]
status: complete
---

# Multiple Additive Regression Trees

## Definition

> [!definition] Multiple Additive Regression Trees (MART)
> **MART** is gradient-boosted regression trees applied to ranking. It builds an **additive ensemble** of shallow regression trees, where each new tree is fit to the **negative gradient** (the pseudo-residuals) of the loss with respect to the model's current scores. The final scoring function is the sum of all tree outputs, and it underlies [[LambdaMART]] — the tree-based realization of the [[Listwise LTR|listwise]] [[LambdaRank]] objective.

## Intuition

A single regression tree is a weak learner: it partitions the feature space into a few regions and predicts a constant in each. MART makes it strong by **boosting** — fitting trees sequentially, each one correcting the errors left by the ensemble so far.

The key trick is **gradient boosting**: instead of fixing a specific loss form, MART treats the current ensemble's scores $F(x)$ as parameters and does functional **gradient descent** on the loss. At each round it computes the gradient of the loss w.r.t. each training point's score, and trains the next tree to predict that (negated) gradient. Adding a small step in that direction reduces the loss — exactly like [[Gradient Descent]], but in function space and with trees as the descent direction.

For ranking, the loss is not differentiable (it depends on the metric, e.g. [[NDCG]]). MART sidesteps this: in [[LambdaMART]] the per-document gradient is supplied directly by the LambdaRank "lambdas" (metric-weighted pairwise gradients), so MART never needs an explicit loss — only the gradients.

## Mathematical Formulation

MART builds an additive model after $M$ boosting rounds:

$$F_M(x) = \sum_{m=1}^{M} \nu \, h_m(x), \qquad F_m(x) = F_{m-1}(x) + \nu \, \gamma_m \, h_m(x)$$

where:
- $x$ — feature vector of a query-document pair
- $h_m(\cdot)$ — the $m$-th regression tree (a piecewise-constant function)
- $\gamma_m$ — leaf/step value(s) chosen to minimize the loss along $h_m$
- $\nu \in (0,1]$ — **shrinkage** (learning rate) that scales each tree's contribution
- $F_M(x)$ — final ranking score (documents sorted by descending $F_M$)

At round $m$, each tree is fit to the **negative gradient** of the loss $L$ evaluated at the current scores (the pseudo-residuals):

$$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F = F_{m-1}}$$

where:
- $r_{im}$ — pseudo-residual (descent direction) for training example $i$ at round $m$
- $y_i$ — label / relevance grade for example $i$
- $L(y_i, F(x_i))$ — per-example loss (e.g. squared error for plain regression)

The tree $h_m$ is trained by least-squares to predict $r_{im}$, and within each leaf region $R_{jm}$ the optimal constant is

$$\gamma_{jm} = \arg\min_{\gamma} \sum_{x_i \in R_{jm}} L\!\left(y_i,\, F_{m-1}(x_i) + \gamma\right)$$

For **LambdaMART**, the residual $r_{im}$ is replaced by the LambdaRank lambda $\lambda_i$ (sum of metric-scaled pairwise gradients), so the loss $L$ need never be written down explicitly:

$$\lambda_i = \sum_{j:\, y_i > y_j} -\frac{|\Delta \text{NDCG}_{ij}|}{1 + e^{\,s_i - s_j}} \;+\; \sum_{j:\, y_j > y_i} \frac{|\Delta \text{NDCG}_{ij}|}{1 + e^{\,s_j - s_i}}$$

where:
- $\lambda_i$ — pseudo-gradient for document $i$ (used in place of $r_{im}$)
- $\Delta \text{NDCG}_{ij}$ — change in [[NDCG]] from swapping documents $i$ and $j$
- $s_i = F_{m-1}(x_i)$ — current ensemble score of document $i$

## Key Properties / Variants

- **Additive ensemble of weak learners**: final score is a sum of many shallow trees; trees are fixed once added (no backprop through earlier trees).
- **Loss-agnostic via gradients**: works for any differentiable loss (squared error, logistic) — and even for non-differentiable ranking objectives, by supplying gradients directly (LambdaMART).
- **Regularization knobs**: shrinkage $\nu$ (small values like 0.1 generalize better but need more rounds), tree depth / number of leaves, number of trees $M$, and subsampling of data/features (stochastic gradient boosting).
- **Handles raw features well**: insensitive to feature scaling and monotone transforms; handles mixed numeric features common in [[Learning to Rank]] feature sets.
- **Strong LTR baseline**: [[LambdaMART]] (MART + [[LambdaRank]] gradients) won the Yahoo! Learning to Rank Challenge and remains a hard-to-beat baseline for tabular ranking features, often competitive with neural rankers.
- **Variants**: GBRT/GBDT (the general gradient-boosting family), stochastic gradient boosting (row/column subsampling), and efficient implementations (XGBoost, LightGBM) used to train MART/LambdaMART at scale.

```pseudo
Algorithm: MART (Gradient-Boosted Regression Trees)
───────────────────────────────────────────────────
Input: training data {(x_i, y_i)}, loss L, rounds M,
       shrinkage ν, tree size (max leaves J)

Initialize F_0(x) = argmin_c Σ_i L(y_i, c)   (constant model)

For m = 1 ... M:
    # 1. Pseudo-residuals = negative gradient at current scores
    For each i:
        r_im = -[ ∂L(y_i, F(x_i)) / ∂F(x_i) ]_{F = F_{m-1}}
        # (LambdaMART: set r_im = λ_i, the metric-weighted lambdas)

    # 2. Fit a J-leaf regression tree h_m to the targets r_im
    Fit h_m by least squares to {(x_i, r_im)}

    # 3. Per-leaf optimal step (line search within each region R_jm)
    For each leaf region R_jm:
        γ_jm = argmin_γ Σ_{x_i ∈ R_jm} L(y_i, F_{m-1}(x_i) + γ)

    # 4. Additive update with shrinkage
    F_m(x) = F_{m-1}(x) + ν * Σ_j γ_jm * 1[x ∈ R_jm]

Return F_M(x)   # rank documents by descending F_M
```

## Connections

- Realized in ranking by: [[LambdaMART]] (MART driven by [[LambdaRank]] gradients)
- Optimizes: [[Listwise LTR]] / metric-based objectives such as [[NDCG]]
- Within: [[Learning to Rank]] (a tree-based, non-neural ranker)
- Mechanism: functional [[Gradient Descent]] (boosting in function space)
- Contrasts with: pairwise losses ([[Pairwise LTR]], [[RankNet]]) and neural rankers ([[BERT for IR]], [[Transformers]])

## Appears In

- [[Listwise LTR]]
- [[IR-L10 - Learning to Rank]]
- [[Learning to Rank]]
- [[LambdaMART]]
