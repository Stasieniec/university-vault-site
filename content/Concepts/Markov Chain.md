---
type: concept
aliases: [k-order Markov Chain, Transition Matrix]
course: [RecSys]
tags: [sequential-rec, collaborative-filtering, exam-topic]
status: complete
---

# Markov Chain

## Definition

> [!definition] Markov Chain (for Sequential Recommendation)
> A **Markov Chain** models the probability of the **next item** in a chronologically ordered sequence of interactions $\langle i_1, i_2, \dots, i_n \rangle$ conditioned on the **recent history**. A **k-order Markov Chain** assumes the next item depends only on the **last $k$ items** (the *Markov assumption*: the future is independent of the distant past given the recent past). In its simplest form (first-order, $k=1$) the model is fully described by a **[[Transition Matrix]]** $T$ whose entry $T_{ij}$ is the probability of transitioning from item $i$ to item $j$. This is the earliest family of [[Sequential Recommendation]] methods [Shani et al., 2005], motivated by the fact that [[Matrix Factorization]] ignores interaction order.

## Intuition

> [!intuition] Order matters; MF throws it away
> Standard [[Collaborative Filtering]] / [[Matrix Factorization]] treats a user's interactions as an **unordered set** — it cannot tell that "phone → phone case" is sensible but "phone case → phone" is not, nor that Star Wars IV precedes V precedes VI. A Markov Chain fixes this by directly modeling **item-to-item transitions**: if many users who interacted with $i$ next interacted with $j$, then $T_{ij}$ is high and $j$ is recommended after $i$. The "memory" is deliberately short ($k$ items) because conditioning on the *entire* history is statistically infeasible — there are too few sequences that match a long context exactly.

## Mathematical Formulation

The general next-item objective over the full history is

$$P(i_{n+1} \mid i_1, \dots, i_n)$$

The **k-order Markov assumption** truncates the conditioning context to the last $k$ items:

$$P(i_{n+1} \mid i_1, \dots, i_n) \;\approx\; P(i_{n+1} \mid i_{n-k+1}, \dots, i_n)$$

For a **first-order** chain ($k=1$) the model reduces to a single **[[Transition Matrix]]** estimated by **counting consecutive co-occurrences** (an n-gram / maximum-likelihood estimate):

$$T_{ij} = \hat{P}(i_{n+1}=j \mid i_n=i) = \frac{\text{count}(i \rightarrow j)}{\sum_{j'} \text{count}(i \rightarrow j')}$$

where:
- $\langle i_1, \dots, i_n \rangle$ — the chronologically ordered interaction (or basket) sequence
- $k$ — the **order** (memory length); $k=1$ gives a first-order chain, the standard case
- $T \in \mathbb{R}^{|I| \times |I|}$ — the **global transition matrix** over the item catalog $I$; rows = "from" item, columns = "to" item; each row sums to 1
- $\text{count}(i \rightarrow j)$ — number of observed transitions from item $i$ directly to item $j$ in the training data
- the denominator (the support, often shown as a "#" column) is the total number of transitions observed *out of* item $i$

The chain is **non-parametric**: it stores raw transition counts rather than learned embeddings, so it does not scale capacity with a dimensionality hyperparameter (unlike factorized models).

## Key Properties / Variants

- **Global, non-personalized.** A single matrix $T$ is shared by **all users** — the same transition behavior applies to everyone. This is the chief limitation that [[FPMC|Factorized Personalized Markov Chains (FPMC)]] addresses with per-user matrices.
- **First-order only by default.** Captures only the immediately preceding item's influence; higher $k$ increases context but worsens sparsity exponentially.
- **Data sparsity is the central problem** — most item pairs are never observed consecutively, so $T_{ij}=0$ for almost all $(i,j)$. Mitigations:
  - **Skipping** — $\langle x_1, x_2, x_3 \rangle$ lends some likelihood to $\langle x_1, x_3 \rangle$ (allow gaps in the sequence).
  - **Clustering** — treat similar contexts as equivalent, $\langle x, y, z \rangle \approx \langle w, y, z \rangle$.
  - **Mixture modeling** — interpolate transition estimates across several orders $k$ (mix n-gram models).
- **Personalization via factorization.** Per-user transition matrices $T^{(u)} \in \mathbb{R}^{|I|\times|I|}$ are far too sparse to estimate directly; [[FPMC]] factorizes the user × from-item × to-item **transition cube** into shared latent factors, combining a long-term [[Matrix Factorization]] term with a short-term factorized first-order transition.
- **Where it wins / loses.** On **sparse** data or with a tight compute budget, simple Markov-chain models (and FPMC) can **outperform** deep models like [[GRU4Rec]]; with abundant data, deep [[Sequential Recommendation]] models dominate because Markov chains see only first-order dependencies.

Estimating the first-order transition matrix:

```pseudo
Algorithm: Fully-Parametrized First-Order Markov Chain
──────────────────────────────────────────────────────
Input: training sequences {<i_1,...,i_n>} for all users
Initialize count[i][j] = 0 for all item pairs (i,j)

For each user sequence <i_1, ..., i_n>:
  For t = 1 .. n-1:
    count[i_t][i_{t+1}] += 1        # observe transition i_t -> i_{t+1}

For each from-item i:
  total = sum_j count[i][j]          # support (# of transitions out of i)
  For each to-item j:
    T[i][j] = count[i][j] / total    # MLE transition probability (0 if total = 0)

Recommend after observing item i_n:
  return argmax_j  T[i_n][j]         # (or top-K ranking of row i_n)
```

## Connections

- Special case / motivates: [[FPMC]] — personalized + factorized first-order Markov chain (adds a long-term [[Matrix Factorization]] term)
- Core object: [[Transition Matrix]]
- Family: [[Sequential Recommendation]] / [[Next-Item Prediction]]
- Contrasts with: [[Matrix Factorization]] and [[Collaborative Filtering]] (order-agnostic; treat interactions as an unordered set)
- Superseded by (deep methods): [[GRU4Rec]], [[SASRec]], [[BERT4Rec]]
- Same name, different setting: the [[Markov Decision Process]] / [[Markov Chain]] state-transition formalism in [[Reinforcement Learning]]

## Appears In

- [[RS-L03a - Sequential Recommendation Models]]
