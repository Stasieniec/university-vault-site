---
type: concept
aliases: [RecSys, Recommender Systems]
course: [RecSys]
tags: [collaborative-filtering, evaluation, exam-topic]
status: complete
---

# Recommender System

## Definition

> [!definition] Recommender System (RecSys)
> A **recommender system** is a subclass of **information filtering** systems that provides suggestions for **items** that are most pertinent to a particular **user**. It mitigates **information overload**: it is most useful when a user must choose from a potentially overwhelming number of items a service offers.
>
> Formally, given a set of users $U = \{u_1, \dots, u_n\}$ and a set of items $I = \{i_1, \dots, i_m\}$, the goal is to find the item(s) $i \in I$ of interest for a given user $u \in U$. In most cases, **previous interactions** between (some) users and (some) items are available; sometimes **contextual information** about users, items, and/or interactions is also available.

## Intuition

> [!intuition] Why It Works
> The core bet is that **collective behavior is predictive**: users who agreed in the past tend to agree in the future, and items that co-occur in interaction histories tend to be substitutable or complementary. A recommender exploits the structure in the (sparse) user–item [[User-Item Interaction Matrix]] — either by measuring **similarity** between rows/columns directly (memory-based) or by **fitting a model** that compresses that structure into low-dimensional latent factors (model-based). It then turns predicted preference into a **ranked list** of items per user, which is why ranking metrics, not just classification accuracy, govern evaluation.

## Mathematical Formulation

The canonical model-based formulation is [[Matrix Factorization]]. An $m \times n$ ratings/interaction matrix $R$ is approximately factorized into an $m \times k$ user matrix $U$ and an $n \times k$ item matrix $V$:

$$R \approx U V^{\top}, \qquad \hat{r}_{ij} \approx \bar{u}_i \cdot \bar{v}_j = \sum_{f=1}^{k} u_{if}\, v_{jf}$$

where:
- $R \in \mathbb{R}^{m \times n}$ — observed user–item interaction matrix ($m$ users, $n$ items); most entries are missing.
- $k$ — number of **latent factors** (concepts), with $k \ll \min(m, n)$.
- $\bar{u}_i \in \mathbb{R}^{k}$ — **user factor**: user $i$'s affinity over the $k$ latent concepts.
- $\bar{v}_j \in \mathbb{R}^{k}$ — **item factor**: item $j$'s properties over the same $k$ concepts.
- $\hat{r}_{ij}$ — predicted preference of user $i$ for item $j$, the dot product of the two factors.

Latent factors can be **interpretable**: in the rank-2 example, the two columns correspond to a "history" and a "romance" genre dimension, and a rating reconstructs as (user-affinity-to-history $\times$ item-affinity-to-history) + (user-affinity-to-romance $\times$ item-affinity-to-romance).

The simplest **memory-based** alternative, [[User-based Rating Prediction]], averages the target item's ratings over the user's $k$ nearest neighbors:

$$\hat{r}_{ui} = \frac{1}{|\mathcal{N}_i(u)|} \sum_{v \in \mathcal{N}_i(u)} r_{vi}$$

where $\mathcal{N}_i(u)$ is the set of $k$ nearest neighbors of $u$ who have rated item $i$, and $r_{vi}$ is neighbor $v$'s rating of item $i$.

## Key Properties / Variants

- **Paradigm axes** (a system is a point in this space, not a single label):
  - *Target*: item recommendation (typical) vs. user recommendation (e.g., people-you-may-know).
  - *Signal*: [[Content-Based Filtering]] (content only) vs. [[Collaborative Filtering]] (interactions only) vs. [[Hybrid Recommendation]] (both).
  - *Structure*: [[Sequential Recommendation]] (order matters), [[Session-based Recommendation]] (current session), multi-item / next-basket, and knowledge-graph-based.
- **Collaborative filtering families:**
  - [[Neighborhood-based Collaborative Filtering]] (memory-based): use [[User-Item Interaction|similarity]] between users or items; **simple, efficient, transparent**, but suffers **sparsity, noise, scalability**.
  - [[Model-based Collaborative Filtering]]: train a model (e.g., MF); **scalable**, but **complex, black-box, overfitting-prone** with little data.
- **Beyond linear MF:** neural models capture **non-linear** interactions, **sequential** signals, and **heterogeneous content** (text/image/audio/video). [[Neural Collaborative Filtering]] (He et al., 2017) is canonical; **MF is a special case of NCF** — replace the neural CF layers with element-wise multiplication, fix the output weights to an all-ones unit vector $J_{k\times1}$, and use identity activation, recovering $p_u \cdot q_i$.
- **Training NCF as binary classification:** label $y_{ui}=1$ if relevant else $0$; weighted square loss for [[Explicit Feedback]] or binary cross-entropy for [[Implicit Feedback]]; [[Negative Sampling]] reduces unobserved-instance count.
- **Evaluation:** [[Offline Evaluation]] (historical log data) vs. [[Online Evaluation]] / [[A/B Testing]]. Accuracy metrics split into set-based ([[Recall]] / [[Hit Rate]] — rank-insensitive) and rank-aware ([[MRR]], [[NDCG]]). Two lists with the same relevant items at different positions get identical recall but different MRR — motivating rank-aware metrics. [[Beyond-Accuracy Metrics]] ([[Diversity]], [[Fairness in Recommendation]], [[Novelty]]) also matter.
- **No universal winner:** best model depends on problem formulation, domain, and available context; hybrids often win. **Reproducibility is fragile** (Dacrema et al., 2019): always tune baselines, count parameters for fairness, never tune on the test set.

```pseudo
Algorithm: Generic Top-N Recommendation Pipeline
──────────────────────────────────────────────
Input: users U, items I, interaction matrix R (sparse), target user u
Train:  fit model M on R           # e.g., factorize R ≈ U Vᵀ, or train NCF
Score:  for each candidate item i in I not yet interacted by u:
            ŝ(u,i) ← M.predict(u, i)   # e.g., ū_u · v̄_i
Rank:   sort candidate items by ŝ(u,i) descending
Return: top-N items as the recommendation list for u
```

## Connections

- Methods: [[Collaborative Filtering]], [[Content-Based Filtering]], [[Hybrid Recommendation]]
- Model-based core: [[Matrix Factorization]], [[Neural Collaborative Filtering]]
- Memory-based core: [[Neighborhood-based Collaborative Filtering]], [[User-based Rating Prediction]]
- Data: [[User-Item Interaction Matrix]], [[Explicit Feedback]], [[Implicit Feedback]], [[Cold Start Problem]]
- Specializations: [[Sequential Recommendation]], [[Session-based Recommendation]], [[Generative Recommendation]], [[LLM-based Recommendation]]
- Evaluation: [[Offline Evaluation]], [[Online Evaluation]], [[A/B Testing]], [[Recall]], [[MRR]], [[NDCG]], [[Beyond-Accuracy Metrics]]
- Related field: [[Information Retrieval]] (recommendation is a form of information filtering)

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
