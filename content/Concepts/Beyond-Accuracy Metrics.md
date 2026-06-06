---
type: concept
aliases: [Beyond-Accuracy Evaluation]
course: [RecSys]
tags: [evaluation, fairness, exam-topic]
status: complete
---

# Beyond-Accuracy Metrics

## Definition

> [!definition] Beyond-Accuracy Metrics
> **Beyond-accuracy metrics** evaluate quality factors of a recommendation list that go *beyond the mere correctness* (relevance) of items. Accuracy metrics ([[Recall]], [[Precision at K|Precision]], [[NDCG]], [[MRR]], [[MAP]]) only ask *"are the recommended items relevant?"*. Beyond-accuracy metrics instead ask whether the list is **diverse**, **serendipitous**, **novel**, has good catalog **coverage**, and is **fair** to both users and item providers. They are essential for real-world recommendation quality, where always returning the most accurate (often most popular, most similar) items can hurt engagement, reinforce [[Filter Bubble|filter bubbles]], and produce unfair outcomes.

## Intuition

> [!intuition] Why correctness is not enough
> A movie recommender that returns ten films from the same franchise can be perfectly *accurate* (every film is relevant) yet useless — the user sees no variety and quickly disengages. Likewise, a news feed that only re-shows articles you have already read scores high on relevance but offers zero **novelty**. And a job recommender that maximizes click accuracy may funnel high-paying ads mostly to men, an unacceptable **fairness** failure even at high accuracy.
>
> The core tension: maximizing accuracy tends to recommend **popular and mutually similar** items. This narrows [[Diversity]], starves the [[Long Tail|long tail]] of exposure, and amplifies bias. Beyond-accuracy metrics quantify these orthogonal axes so they can be measured and traded off explicitly (multi-objective evaluation).

## Mathematical Formulation

The metrics below operate on a recommendation list $R = (i_1, \dots, i_N)$ (typically the [[Top-K Recommendation|top-K]]) per user, often aggregated over the user set $U$.

> [!formula] Intra-List Distance (Diversity, ↑)
> $$\text{ILD} = \frac{1}{\binom{N}{2}} \sum_{i=1}^{N} \sum_{j=i+1}^{N} \text{distance}(\text{item}_i, \text{item}_j)$$
>
> where:
> - $N$ — length of the recommendation list
> - $\text{distance}(\cdot,\cdot)$ — a dissimilarity measure between two items (e.g. $1 - \cos$ of content/embedding vectors)
> - $\binom{N}{2}$ — number of unordered item pairs (normalizer)
>
> Higher average pairwise distance ⇒ more diverse list. Related list-level diversity scores: **Diversity Score** $\text{DS} = \frac{\#\text{recommended categories}}{\#\text{recommended items}}$; **Entropy** $-\sum_i p(i)\log_2 p(i)$ and **Gini** $1 - \sum_i p(i)^2$ over the category distribution $p(i)$ of the list.

> [!formula] Serendipity (↑) and Catalog Coverage (↑)
> $$\text{Serendipity} = \frac{|R_{\text{unexpected}} \cap R_{\text{useful}}|}{|R|} \qquad\qquad \text{Coverage} = \frac{|\,\text{unique recommended items}\,|}{|\,\text{catalog}\,|}$$
>
> where:
> - $R_{\text{unexpected}}$ — items in $R$ dissimilar to what the user liked in the past (the *surprise* component)
> - $R_{\text{useful}}$ — items in $R$ that are actually relevant/useful to the user
> - Serendipity counts recommendations that are **both surprising and useful**; novelty is the surprise component alone.
> - Coverage measures the **breadth** of the catalog ever exposed — low coverage means only a few popular items are shown, leaving the long tail unexposed.

> [!formula] Fairness — User Group Fairness (UGF, ↓) and Exposure
> $$\text{UGF} = \left| \frac{1}{|Z_1|} \sum_{i \in Z_1} M(W_i) \;-\; \frac{1}{|Z_2|} \sum_{i \in Z_2} M(W_i) \right|$$
>
> where:
> - $Z_1, Z_2$ — two user groups (e.g. advantaged vs disadvantaged, by gender/region/activity)
> - $M(W_i)$ — a quality metric (e.g. F1@10, NDCG) for user $i$
> - UGF = absolute gap in average quality between groups; **lower is better** (0 = no disparity).
>
> **Item-side fairness** instead measures *exposure*. Exposure is computed via a **browsing model** that decays with rank position $k$, e.g. logarithmic $\frac{1}{\log_2(k+1)}$, geometric $\gamma^{k}$, or cascade. Group exposure $\epsilon(g)$ then feeds parity/utility metrics (below).

## Key Properties / Variants

- **Diversity** (↑): dissimilarity/variety within a list. Metrics: Diversity Score (categories/items), [[Intra-List Distance]] (ILD), category [[Entropy]], Gini index $1-\sum_i p(i)^2$ (here 0 = single category, 1 = even spread; ↑ better).
- **Serendipity** (↑): surprising *and* useful. **Novelty** (↑): items unknown to / unseen by the user (the surprise alone). **Coverage** (↑): fraction of the catalog ever recommended.
- **Fairness is two-sided**:
  - *User fairness* — equal recommendation quality across user groups (UGF, ↓).
  - *Item / provider fairness* — fair distribution of **exposure (attention)** across item groups, depends on a position-decaying browsing model.
- **Item-fairness metric taxonomy** (choose by *goal* × *#groups*):
  - **Statistical parity** (comparable exposure regardless of merit): Demographic Parity $\text{DP}=\epsilon(G_0)/\epsilon(G_1)$, MinMaxRatio $\frac{\min_g \epsilon(g)}{\max_g \epsilon(g)}$ (↑), Max-Min Fairness $\text{MMF}=\min_g \epsilon(g)/\text{Weight}(g)$ (↑).
  - **Equality of opportunity** (exposure proportional to utility/merit): Exposed Utility Ratio, Realized Utility Ratio (uses actual CTR), Expected Exposure Loss $\|\epsilon-\epsilon^*\|_2^2$ (↓), Inequity of Amortized Attention $\sum_i |A_i - R_i|$ (↓, $L_1$ between attention and predicted relevance).
  - Two groups → DP, EUR, RUR; multiple groups → MinMaxRatio, MMF, EEL, IAA.
- **Multi-objective trade-offs**: Diversity/Fairness/Efficiency frequently trade off against accuracy (over-optimizing accuracy → filter bubbles, [[Popularity Bias]], unfair exposure). But trade-offs are not universal — some settings achieve **win-win** (e.g. diversity + accuracy improving together in [[Sequential Recommendation]]).
- **Utility Loss**: the accuracy cost paid for a fairness re-ranking, $\text{Utility}_{\text{ori}} - \text{Utility}_{\text{fair}}$ (sum of relevance over the original vs fair list).
- **Interventions** to optimize these metrics span three stages: *pre-processing* (debias data), *in-processing* (fairness/diversity term in the loss, e.g. $\mathcal{L}=\mathcal{L}_{\text{relevance}}+\lambda\mathcal{L}_{\text{fairness}}$, or re-weighting under-performing groups), and *post-processing* (re-rank the output list under fairness/diversity constraints, e.g. greedy [[Maximal Marginal Relevance (MMR)|MMR]]-style swaps). The FairDiverse toolkit (Xu et al., 2025) standardizes these for both search and recommendation.

## Connections

- Complements (does not replace): [[Recall]], [[Precision at K]], [[NDCG]], [[MRR]], [[MAP]] — accuracy/ranking metrics
- Core sub-concepts: [[Diversity]], [[Novelty]], [[Serendipity]], [[Coverage]], [[Fairness in Recommendation]]
- Diversity measures: [[Intra-List Distance]], [[Entropy]], [[Maximal Marginal Relevance (MMR)]]
- Fairness sides: [[User Fairness]], [[Item Fairness]], [[Provider Fairness]], [[Exposure Fairness]], [[Algorithmic Fairness]]
- Caused/measured failure modes: [[Popularity Bias]], [[Long Tail]], [[Filter Bubble]], [[Echo Chamber]], [[Position Bias]]
- Evaluated under: [[Offline Evaluation]], [[Online Evaluation]], [[A/B Testing]]
- Trade-offs arise in: [[Top-K Recommendation]], [[Collaborative Filtering]], [[Sequential Recommendation]]

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
