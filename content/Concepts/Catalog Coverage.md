---
type: concept
aliases: [Coverage, Catalogue Coverage]
course: [RecSys]
tags: [evaluation, fairness, exam-topic]
status: complete
---

# Catalog Coverage

## Definition

> [!definition] Catalog Coverage
> **Catalog coverage** is a beyond-accuracy metric that assesses the **percentage of unique items in the entire catalog that are actually recommended** to users across all recommendation lists. It indicates the **breadth of exposure** a system provides: how much of the item inventory ever reaches a user. A system that always recommends a small set of popular items has **low coverage**, leaving a large portion of the catalog completely unexposed.

## Intuition

> [!intuition] How much of the shelf do users ever see?
> Accuracy metrics ([[Recall]], [[Precision]], [[NDCG]]) only ask whether the items shown were *correct*. They are blind to the items that were *never shown at all*. A recommender can score perfectly on accuracy while recommending the same handful of blockbuster items to every user.
>
> Coverage takes the **provider / catalog side** view: pool together every item that appears in *any* user's top-K list, and ask what fraction of the whole catalog that represents. High coverage means the system distributes exposure broadly; low coverage means most of the [[Long Tail]] stays invisible. This directly connects to [[Popularity Bias]] and [[Item Fairness]] — a popularity-skewed model amplifies the long-tail distribution and exposes only the popular head.

## Mathematical Formulation

The simplest (and lecture-presented) form is **item-space coverage**: the size of the union of all recommended items relative to the catalog size.

$$\text{Catalog Coverage} = \frac{\left| \bigcup_{u \in U} \text{TopK}_u \right|}{|I|} = \frac{\text{Number of unique recommended items}}{\text{Total number of items in the catalog}}$$

where:
- $U$ — the set of all users
- $\text{TopK}_u$ — the top-$K$ recommendation list produced for user $u$
- $\bigcup_{u \in U} \text{TopK}_u$ — the set of **distinct** items recommended to *at least one* user (a union, so an item recommended to many users is counted once)
- $I$ — the full item catalog; $|I|$ is its cardinality
- range: $[0, 1]$ — higher ($\uparrow$) is better; $1.0$ means every catalog item was recommended to someone

## Key Properties / Variants

- **Direction:** higher is better ($\uparrow$). It is a *beyond-accuracy* objective, often in tension with accuracy (see [[Diversity]] vs accuracy trade-off below).
- **Union, not sum:** because of the union over users, an item recommended to 1 user and an item recommended to 1,000,000 users contribute identically. Plain catalog coverage therefore measures *whether* an item is exposed, not *how often* — it can hide severe exposure inequality among the items that *are* covered.
- **Distinguish from [[Diversity]]:** diversity (e.g. [[Intra-List Distance]], category [[Entropy]], Gini) is measured *within a single list* — variety for one user. Coverage is a *system-level / aggregate* property across *all* lists. A system can give each user a diverse list yet still cover only a small slice of the catalog (every user gets the same diverse set).
- **Sensitivity to $K$ and to $|U|$:** larger top-$K$ lists or more users mechanically increase the union and inflate coverage; comparisons must fix $K$ and the user set.
- **Common related variants** (beyond the strict item-space form above):
  - *Prediction coverage* — fraction of items for which the model can produce a score/prediction at all (relevant to [[Cold Start]] items that have no interactions).
  - *Gini / entropy of the recommendation-frequency distribution* — captures *how evenly* exposure is spread across the covered items (the "how often" that plain coverage ignores), linking coverage to exposure-inequality measures.

```pseudo
Algorithm: Catalog Coverage (item-space)
─────────────────────────────────────────
Input: catalog I, users U, model, list size K
recommended_set ← ∅            # union accumulator
for each user u in U:
    TopK_u ← recommend(model, u, K)   # top-K items for u
    recommended_set ← recommended_set ∪ TopK_u
coverage ← |recommended_set| / |I|
return coverage
```

> [!warning] Coverage ≠ fair exposure
> A high coverage number does **not** guarantee fair exposure. Because coverage uses a *set union*, it only checks that each item was shown at least once; it says nothing about the **distribution** of attention. Two systems with identical 100% coverage can differ enormously: one may show every item roughly equally, another may show one item to almost everyone and the rest to a single user each. To capture *how much* exposure each item receives — accounting for the decreasing attention users pay at deeper rank positions (a browsing model) — use exposure-based [[Item Fairness]] metrics (MinMaxRatio, Max-Min Fairness) rather than coverage alone.

## Connections

- Beyond-accuracy sibling of: [[Diversity]], [[Novelty]], [[Serendipity]]
- Contrasts with: [[Recall]], [[Precision]], [[NDCG]] (accuracy-only metrics ignore unexposed items)
- Caused/worsened by: [[Popularity Bias]], the [[Long Tail]] distribution, [[Cold Start]] items
- Provider-side concern within: [[Fairness in Recommendation]], specifically [[Item Fairness]] / [[Exposure Fairness]]
- Within-list counterpart: [[Intra-List Distance]], [[Entropy]] (per-list variety vs system-wide breadth)
- Trade-off context: [[Filter Bubble]] / [[Echo Chamber]] formation when coverage collapses

## Appears In

- [[RS-L02 - Evaluation Beyond Accuracy]]
