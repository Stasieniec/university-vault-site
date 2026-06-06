---
type: concept
aliases: []
course: [RecSys]
tags: [evaluation, fairness, exam-topic]
status: complete
---

# Popularity Bias

## Definition

> [!definition] Popularity Bias
> **Popularity bias** is the tendency of a recommender system to over-favour a small number of mainstream / frequently-interacted-with items at the expense of niche, [[Long Tail|long-tail]] items. Two things compound: (1) interaction data is itself **long-tailed** — a few items absorb most of the feedback; (2) because the recommendation list (top-K) is **limited**, the algorithm *amplifies* this skew, pushing popular items even harder and leaving most of the catalogue unexposed. It is the canonical source of **item-side unfairness** and a driver of low [[Catalog Coverage|catalogue coverage]] and low [[Novelty]] / [[Diversity]].

## Intuition

> [!intuition] Why the tail collapses
> Logged feedback is collected *through* a recommender that already preferred popular items, so popular items accrue even more interactions — a feedback loop. A model trained to maximise [[NDCG|accuracy]] learns that "predict popular" is a cheap way to be right on average, since popular items are the safe bet for most users. With only K slots per user, marginal long-tail items never make the cut, so their exposure (and future data) shrinks toward zero. The same small set is shown to everyone (low coverage), narrowing taste over time into a [[Filter Bubble]]. Crucially, popularity bias is *not* the same as a popular item genuinely being relevant — it is the **systematic over-representation beyond what relevance justifies**.

## Mathematical Formulation

The bias surfaces at three points: the **data**, the **model**, and the **decoding**. The shared object is item **exposure** — the (position-discounted) attention an item or group receives in served lists, computed by a browsing model that decays with rank (logarithmic / geometric / cascade). Item fairness then measures how far exposure deviates from a target. Two evaluation lenses from RS-L02:

> [!formula] Catalogue Coverage and Group Exposure Parity
> $$\text{Catalog Coverage} = \frac{|\{\, i \in \mathcal{I} : i \text{ recommended to some user} \,\}|}{|\mathcal{I}|}$$
> $$\text{DP} = \frac{\text{Exposure}(G_{\text{pop}})}{\text{Exposure}(G_{\text{tail}})}, \qquad \text{MinMaxRatio} = \frac{\min_{g \in G} \text{Exposure}(g)}{\max_{g \in G} \text{Exposure}(g)}$$
>
> where:
> - $\mathcal{I}$ — full item catalogue
> - $G_{\text{pop}}, G_{\text{tail}}$ — item groups split by popularity (head vs long tail)
> - $\text{Exposure}(g)$ — total position-discounted attention to group $g$, summed over served lists
> - **Catalogue Coverage** $\downarrow$ under popularity bias (most of $\mathcal{I}$ never shown)
> - **DP** $\gg 1$ under popularity bias (head gets far more exposure than tail); statistical parity wants DP $\approx 1$
> - **MinMaxRatio** $\to 0$ as the worst-off (tail) group is starved; $\uparrow$ (toward 1) is fairer

The standard **in-processing** countermeasure re-weights the loss so under-exposed groups count more, e.g. Inverse Propensity Scoring (IPS), which weights a group by the reciprocal of its summed popularity:

> [!formula] Popularity Debiasing via Re-weighted / Regularized Loss
> $$\mathcal{L} = \sum_{g \in G} w_g \, \mathcal{L}_g, \qquad w_g \propto \frac{1}{\sum_{i \in g} \text{pop}(i)} \qquad\text{(IPS-style)}$$
> $$\mathcal{L} = \mathcal{L}_{\text{relevance}} + \lambda \, \mathcal{L}_{\text{fairness}}$$
>
> where:
> - $w_g$ — weight on group $g$'s loss; rarer (tail) groups get up-weighted
> - $\text{pop}(i)$ — interaction count / popularity of item $i$
> - $\mathcal{L}_{\text{fairness}}$ — penalty on exposure imbalance (e.g. squared gap between group exposures)
> - $\lambda$ — trade-off knob: larger $\lambda$ buys fairness at the cost of accuracy (Utility Loss)

In **generative recommendation** (RS-L04) the bias re-emerges at *decoding* as **amplification bias**: in autoregressive [[Beam Search]] over [[Semantic ID|Semantic IDs]], popular code prefixes win every step and long-tail items are pruned before they are ever scored:
$$p_\theta(\mathbf{z}_i \mid \mathbf{x}) = \prod_{\ell=1}^{L} p_\theta(z_{i,\ell} \mid \mathbf{x}, z_{i,<\ell})$$
where a popular shared prefix $(z_1, z_2, \dots)$ dominates the product, so the top-$B$ beam collapses into one "family" of head items. With [[Atomic Item IDs|atomic IDs]] the same effect appears directly as popularity bias in the softmax over the catalogue.

## Key Properties / Variants

- **Data-level (cause):** the long-tail interaction distribution (RS-L02 slide 40) — a few popular items, a heavy tail of rarely-touched items.
- **Model-level (amplification):** accuracy-optimised models reproduce and *exacerbate* the skew because predicting popular items is a low-risk way to maximise hit-rate / NDCG.
- **Decoding-level (GenRec):** *amplification bias* + *homogeneity* in beam search — top results share a popular prefix, so the list is near-duplicates of head items (RS-L04 slides 49–50).
- **Distinct from cold start:** popularity bias starves items *with little data*; an item can be valid/decodable yet still never surface because the generator was trained only on clicked (popular) items — "fragile cold-start."
- **Two-sided harm:** item/provider side (under-exposed providers lose revenue, may leave the platform) and user side (low novelty/diversity, filter bubbles, dissatisfaction).
- **Mitigation by pipeline stage** (FairDiverse framing):
  - *Pre-processing* — debias the logged data / re-sample the tail before training.
  - *In-processing* — re-weight or re-sample under-exposed groups; add a fairness regulariser $\lambda \mathcal{L}_{\text{fairness}}$ (FOCF, IPS, FairDual).
  - *Post-processing* — re-rank the output list to inject tail items ([[Maximal Marginal Relevance (MMR)|MMR]], CP-Fair, P-MMF).
  - *Decoding-time (GenRec)* — temperature / sampling, diverse beam search, or reward diversity/validity in [[Group Relative Policy Optimization|GRPO]]; or fix it at the tokenizer so popular items don't all collapse onto one prefix (LETTER).
- **Evaluation caveat:** offline accuracy metrics (Recall@K, NDCG@K) *reward* popularity bias — surfacing a good but unseen tail item counts as "wrong" because it isn't the logged click, so benchmarks under-credit exactly the novelty we want.

Greedy mitigation by post-hoc re-ranking (MMR-style, trading relevance for spread):

```pseudo
Algorithm: Diversity / Tail-aware Re-ranking (post-processing)
──────────────────────────────────────────────────────────────
Input: candidate list C scored by relevance s(i); selected set S = {}
Loop until |S| = K:
  for each i in C \ S:
    mmr(i) = λ·s(i) − (1−λ)·max_{j in S} sim(i, j)
    (optionally subtract β·popularity(i) to up-weight the tail)
  i* = argmax_i mmr(i)
  S ← S ∪ {i*}
Return S      # spreads exposure across items/prefixes, raising coverage
```

## Connections

- Causes / is grounded in: [[Long Tail]], [[Long-Tail Distribution]], [[Implicit Feedback]]
- Type of: [[Item Fairness]], [[Fairness in Recommendation]] (item-side)
- Hurts these beyond-accuracy metrics: [[Catalog Coverage]], [[Novelty]], [[Diversity]], [[Serendipity]]
- Related biases: [[Position Bias]] (exposure decays with rank), [[Exposure Fairness]]
- Leads to: [[Filter Bubble]], [[Echo Chamber]]
- Trades off against: [[NDCG]] / accuracy (Utility Loss when debiasing)
- Mitigated with: [[Inverse Propensity Weighting]] / IPS, [[Maximal Marginal Relevance (MMR)]], [[Bayesian Personalized Ranking (BPR)]] negative sampling choices
- Re-emerges in: [[Generative Recommendation]] decoding (amplification bias) via [[Beam Search]] over [[Semantic ID|Semantic IDs]] vs [[Atomic Item IDs]]

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L04 - Generative Recommendation]]
