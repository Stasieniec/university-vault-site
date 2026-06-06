---
type: concept
aliases: [Diversity Score, Intra-List Distance, ILD]
course: [RecSys]
tags: [evaluation, exam-topic]
status: complete
---

# Diversity

## Definition

> [!definition] Diversity
> **Diversity** is a [[Beyond-Accuracy Metrics|beyond-accuracy]] objective that measures the **dissimilarity or variety** of the items inside a single recommendation list. It quantifies how spread-out a top-K list is, using either **custom equations** over item attributes (e.g. category counts) or a **pairwise similarity/distance metric** between items. A perfectly accurate recommender that returns ten near-identical action films has high accuracy but near-zero diversity; recommending fruit, bread, drink, and snacks in a grocery basket has high diversity.
> 
> Diversity is one of several beyond-accuracy factors alongside [[Serendipity]], [[Novelty]], [[Coverage]], and [[Fairness in Recommendation]]. Low diversity can **reduce user engagement** (recommending only the same category narrows choice and bores the user).

## Intuition

> [!intuition] Why accuracy is not enough
> Accuracy-based metrics (Recall, [[NDCG]], [[MRR]]) only reward predicting the exact item a user clicks next. But a list of ten thrillers can be "accurate" while being useless as a set: the user already has a thriller, they want *options*. Diversity is a **list-level** (set-level) property, not an item-level one — you cannot tell whether a list is diverse by looking at any single recommendation, only at how its items relate to each other.
>
> There are two granularities worth keeping distinct: **individual-level diversity** (variety *within one user's* list — the metrics below) and **system-level diversity** like [[Catalog Coverage]] (variety *across all users*, i.e. how much of the catalogue is ever exposed). They are different: every user could get a perfectly diverse 5-category list while the system still only ever recommends the same 5 categories (high individual diversity, low coverage).

## Mathematical Formulation

The course gives four representative individual-level diversity metrics. The two flagged in the title (Diversity Score and Intra-List Distance) are the attribute-based and distance-based forms respectively.

> [!formula] Diversity Score (DS, ↑)
> $$\mathrm{DS} = \frac{\text{number of recommended categories}}{\text{number of recommended items}}$$
> 
> where:
> - numerator — count of *distinct* categories appearing in the recommended list
> - denominator — total number of recommended items (the list length $N$)
> - DS $\to 1$ when every item is a different category (maximally varied); DS $\to 1/N$ when all items share one category. (Liang et al., 2021)

> [!formula] Intra-List Distance (ILD, ↑)
> $$\mathrm{ILD} = \frac{1}{\binom{N}{2}} \sum_{i=1}^{N} \sum_{j=i+1}^{N} \mathrm{distance}(\text{item}_i, \text{item}_j)$$
> 
> where:
> - $N$ — number of items in the recommendation list
> - $\binom{N}{2} = \frac{N(N-1)}{2}$ — number of unordered item pairs (the normalizer, averaging over all pairs)
> - $\mathrm{distance}(\cdot,\cdot)$ — any item–item distance (e.g. $1 - \cos$ similarity of feature/embedding vectors, or a categorical Hamming distance)
> - higher average pairwise distance $\Rightarrow$ a more diverse list. (Cen et al., 2020)

> [!formula] Category-distribution diversity: Entropy and Gini
> Let $p(i)$ be the fraction of recommended items belonging to unique category $i$, over $N$ unique categories.
> $$\mathrm{Entropy} = -\sum_{i=1}^{N} p(i)\,\log_2 p(i) \qquad (\uparrow)$$
> $$\mathrm{Gini} = 1 - \sum_{i=1}^{N} p(i)^2 \qquad (\uparrow)$$
> 
> where:
> - $\mathrm{Entropy}$ — [[Shannon Entropy]] of the category distribution; maximized when categories are equiprobable
> - $\mathrm{Gini}$ — the Gini–Simpson / $(1 - \text{HHI})$ form: $\mathrm{Gini}=0$ means no diversity (one category), $\mathrm{Gini}=1$ means maximum diversity (items spread evenly across categories). Note this is the *diversity* reading of Gini (↑ is better), distinct from the *inequality* Gini used for fairness.

## Key Properties / Variants

- **Set-based, not rank-based.** Plain ILD/DS/Entropy ignore *where* in the list an item sits. Rank- and relevance-aware variants exist (Vargas & Castells, 2011) that weight pairs by position.
- **Two metric families.** Attribute/category based (DS, Entropy, Gini — need item metadata/categories) vs. distance based (ILD — needs an item similarity function or embeddings). Choice depends on what side information you have.
- **Diversity vs. accuracy trade-off.** Highly accurate recommenders tend to surface popular/similar items, lowering diversity and driving the **[[Filter Bubble]]** (the per-user category distribution narrows over time to a single peak). This is a core *multi-objective evaluation* tension — but it is not universal: Yin et al. show diversity and accuracy can improve together in [[Sequential Recommendation]].
- **In [[Generative Recommendation]] diversity is a decoding problem.** Because items get [[Semantic IDs]] where similar items share leading codebook tokens, [[Beam Search]] locks onto one popular prefix (e.g. `(12, 48, ·)`) and the top-$B$ list collapses into near-duplicates ("homogeneity" + popularity amplification). Diversity is injected two ways:
  - **Decoding time:** temperature/sampling, *diverse beam search* (penalize groups for repeating earlier choices), or post-hoc re-rank with **[[Maximal Marginal Relevance (MMR)]]**.
  - **Training time:** reward diversity inside [[GRPO]] (penalize look-alike candidates in the group), or fix it at the tokenizer so popular items don't all share a prefix (LETTER's diversity regularizer for balanced code usage).
- **MMR re-ranking** (the canonical post-processing greedy method, Carbonell & Goldstein, 1998): build the list one item at a time, each step preferring an item *unlike* those already chosen — trading marginal relevance against marginal novelty.

```pseudo
Algorithm: MMR re-ranking for diversity
─────────────────────────────────────────
Input: candidate set C with relevance rel(·); already-selected set S = {}
       similarity sim(·,·); trade-off λ ∈ [0,1]
Loop until list length reached:
    for each d in C \ S:
        score(d) = λ · rel(d) − (1−λ) · max_{s in S} sim(d, s)
    pick d* = argmax score(d); move d* from C to S
return S   # high λ → accuracy; low λ → diversity
```

- **Tooling.** The course's FairDiverse toolkit reports Gini Index and Entropy as its diversity metrics, and supports post-processing re-rankers that re-balance lists under diversity/fairness constraints, paying a measurable **Utility Loss** ($\text{Utility}_{\text{ori}} - \text{Utility}_{\text{fair}}$) in accuracy.

## Connections

- Sibling beyond-accuracy metrics: [[Serendipity]], [[Novelty]], [[Coverage]], [[Catalog Coverage]]
- Contrasted with accuracy metrics: [[NDCG]], [[Recall]], [[MRR]]
- System-level counterpart: [[Long Tail]] exposure, [[Catalogue Coverage]]
- Trade-off partner / failure mode: [[Filter Bubble]], [[Echo Chamber]], [[Popularity Bias]]
- Related fairness goal: [[Fairness in Recommendation]], [[Item Fairness]]
- Achieved via: [[Maximal Marginal Relevance (MMR)]], [[Beam Search]], [[GRPO]]
- Built on: [[Shannon Entropy]]
- Relevant settings: [[Sequential Recommendation]], [[Generative Recommendation]], [[Semantic IDs]]

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L04 - Generative Recommendation]]
