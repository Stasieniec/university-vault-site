---
type: concept
aliases: [BPR]
course: [RecSys]
tags: [collaborative-filtering, sequential-rec, evaluation, exam-topic]
status: complete
---

# Bayesian Personalized Ranking (BPR)

## Definition

> [!definition] Bayesian Personalized Ranking (BPR)
> **BPR** [Rendle et al., 2012] is a **pairwise ranking optimization criterion** for learning recommenders from **[[Implicit Feedback]]** (clicks, purchases, views — no explicit ratings). Instead of predicting an absolute score per item, BPR optimizes the **relative order** of item pairs: for a given user, an observed (positive) item should be ranked **above** an unobserved (negative) item. It is a generic objective that can be plugged on top of any scoring model ([[Matrix Factorization|MF]], [[Factorized Personalized Markov Chains (FPMC)|FPMC]], [[GRU4Rec]]), not a model itself.

## Intuition

> [!intuition] Why pairwise, not pointwise?
> With implicit feedback the only signal is "user $u$ interacted with item $i$." A **pointwise** approach (e.g. fit $\hat{x}_{ui}=1$ for observed, $0$ for the rest) is forced to label all non-interacted items as **negative** — but a non-interacted item is really *missing data*, not a confirmed dislike. BPR sidesteps this: it never asserts an absolute target. It only assumes the user **prefers** the item they engaged with **over** an item they did not. This turns the problem into ranking pairs $(i \succ_u j)$, which is exactly what a top-K recommender is graded on by AUC / [[NDCG]].

## Mathematical Formulation

For each user we want positive item $i$ (observed) ranked above negative item $j$ (unobserved). BPR maximizes the posterior probability of the correct pairwise ordering. Define $\hat{x}_{uij} = \hat{x}_{ui} - \hat{x}_{uj}$, the score difference under any scoring model $\hat{x}_{u\cdot}$. The **BPR-OPT** objective (negative log-posterior) is:

$$
\text{BPR-OPT} = \sum_{(u,i,j)\in D_S} \ln \sigma\!\left(\hat{x}_{uij}\right) - \lambda_\Theta \, \lVert \Theta \rVert^2
$$

which is maximized; equivalently the **loss** minimized in practice (the form used for [[GRU4Rec]]) is:

$$
\mathcal{L}_{\text{BPR}} = -\frac{1}{N_S} \sum_{j=1}^{N_S} \ln \sigma\!\left(\hat{r}_{s,i} - \hat{r}_{s,j}\right)
$$

where:
- $\sigma(x) = \dfrac{1}{1+e^{-x}}$ — logistic sigmoid; $\sigma(\hat{x}_{uij})$ is the modeled probability that $i \succ_u j$.
- $\hat{x}_{ui}$ (or $\hat{r}_{s,i}$) — score the model gives the **positive** item $i$ for user/state $u$ (e.g. dot product $P_u^\top Q_i$ in MF).
- $\hat{x}_{uj}$ (or $\hat{r}_{s,j}$) — score for a **sampled negative** item $j$ (an item the user did not interact with).
- $D_S = \{(u,i,j) \mid i \in I_u^+ ,\, j \notin I_u^+\}$ — the training triples; $I_u^+$ is the set of items $u$ engaged with.
- $\lambda_\Theta \lVert\Theta\rVert^2$ — L2 [[Regularization]] on model parameters $\Theta$ (Gaussian prior $\Theta \sim \mathcal{N}(0,\Sigma_\Theta)$).
- $N_S$ — number of negative samples drawn per positive instance.

The gradient w.r.t. parameters $\Theta$ is

$$
\frac{\partial \text{BPR-OPT}}{\partial \Theta} \;=\; \sum_{(u,i,j)} \frac{-e^{-\hat{x}_{uij}}}{1+e^{-\hat{x}_{uij}}}\cdot \frac{\partial \hat{x}_{uij}}{\partial \Theta} \;-\; \lambda_\Theta \Theta,
$$

so the update size automatically **shrinks toward zero once the pair is already correctly and confidently ordered** ($\hat{x}_{uij}\gg 0$) and is largest for violated pairs.

## Key Properties / Variants

- **Optimizes a smooth surrogate for AUC.** The non-smooth pairwise ranking objective $\sum \mathbf{1}[\hat{x}_{ui} > \hat{x}_{uj}]$ (which *is* per-user AUC) is replaced by the differentiable $\ln\sigma(\hat{x}_{uij})$, making it trainable by [[Stochastic Gradient Descent|SGD]].
- **Model-agnostic loss.** Any model that produces $\hat{x}_{ui}$ can be trained with it. The note context shows it used for [[Matrix Factorization|MF]] (BPR-MF), [[Factorized Personalized Markov Chains (FPMC)|FPMC]] (S-BPR), and [[GRU4Rec]]. The discussion of losses notes BPR / BCE / CE are **not model-specific** and interchangeable.
- **Negative sampling is critical.** Enumerating all $(u,i,j)$ triples is infeasible, so negatives $j$ are sampled (typically uniformly). Too few negatives can cause **overconfidence** — a key finding in the BERT4Rec vs SASRec reproducibility study, where increasing negatives sharply changed results.
- **LearnBPR (bootstrap SGD).** The original training algorithm uses **bootstrap sampling of triples with replacement** rather than item-wise iteration, which avoids the slow convergence of sweeping all items per user.
- **Relation to other losses.** Pairwise (BPR) sits between **[[Pointwise Learning to Rank|pointwise]]** losses (BCE on individual items) and **[[Listwise Learning to Rank|listwise]]** losses (LambdaRank, ListNet); contrastive losses like InfoNCE are a related multi-negative generalization.

```pseudo
Algorithm: LearnBPR (Bootstrap SGD for BPR-OPT)
────────────────────────────────────────────────
Initialize parameters Θ randomly
Repeat:
  Draw (u, i, j) from D_S       # u, positive i ∈ I_u⁺, sampled negative j ∉ I_u⁺
  x_uij  ← x̂_ui(Θ) − x̂_uj(Θ)   # score difference
  g      ← σ(−x_uij)            # = e^{−x_uij}/(1+e^{−x_uij}); large when pair is wrong
  Θ ← Θ + α · ( g · ∂x_uij/∂Θ  −  λ_Θ · Θ )   # gradient ASCENT on log-posterior
until convergence
return Θ
```

## Connections

- Trained from: [[Implicit Feedback]] (the setting BPR was designed for)
- Loss family: [[Pairwise Learning to Rank]]; contrasted with [[Pointwise Learning to Rank]] and [[Listwise Learning to Rank]]
- Surrogate for: AUC (per-user pairwise ranking accuracy)
- Applied to models: [[Matrix Factorization]], [[Factorized Personalized Markov Chains (FPMC)]], [[GRU4Rec]]
- Requires: [[Negative Sampling]]
- Uses: [[Regularization]], [[Stochastic Gradient Descent]]
- Sibling losses in [[Sequential Recommendation]]: BCE ([[SASRec]]), CE/MLM ([[BERT4Rec]]); see [[LambdaRank]], [[Contrastive Learning]]
- Core task it serves: [[Top-K Recommendation]]

## Appears In

- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03a - Sequential Recommendation Models]]
