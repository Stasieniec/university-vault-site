---
type: concept
aliases: [Item Fairness, User Fairness, Provider Fairness, RecSys Fairness]
course: [RecSys]
tags: [evaluation, fairness, exam-topic]
status: complete
---

# Fairness in Recommendation

## Definition

> [!definition] Fairness in Recommendation
> A recommender is a **multi-stakeholder** system: it must serve **users** (consumers) and **items/providers** (e.g. artists, sellers, job candidates). **Fairness in recommendation** is the absence of systematic, unjustified disadvantage to a protected group on either side, arising from biased data or the ranking process [Ekstrand et al., 2022]. Two complementary sides:
> - **User fairness** — recommendation **quality (accuracy)** should not differ across user groups (grouped by gender, region, activity level).
> - **Item / provider fairness** — **exposure (attention)** should be distributed fairly across item groups (grouped by popularity, category, brand), not merely allocated to popular items.

## Intuition

> [!intuition] Why ranking amplifies tiny biases
> Two mechanisms make recommendation unfair even from "neutral" data:
> 1. **Long-tail exacerbation.** Interaction data follows a steep popularity curve. Because the top-K list is *limited*, the algorithm keeps re-recommending already-popular items, starving the long tail of any exposure (a feedback loop / [[Filter Bubble]]).
> 2. **Position bias.** Users pay sharply decreasing attention to deeper ranks (a [[Position Bias|browsing model]]). So a **small** difference in relevance becomes a **large** difference in exposure. In the canonical job-seeker example [Singh & Joachims, 2018], a **0.03** gap in average relevance between two candidate groups produced a **0.32** gap in average exposure (probability of interview).
>
> Fairness metrics therefore compare **exposure** (or accuracy) across groups, not raw relevance.

## Mathematical Formulation

Item fairness rests on **exposure**: an item's exposure is the attention it accrues, weighted by a browsing model $b(\cdot)$ that decays with rank position $k$ (Logarithmic, Geometric, or Cascade decay). For a group $g$,

$$\text{Exposure}(g) \;=\; \sum_{i \in g}\; b\big(\text{rank}(i)\big), \qquad b(k)=\tfrac{1}{\log_2(k+1)} \;\;(\text{logarithmic model})$$

where:
- $b(k)$ — position discount; same shape as the [[NDCG]] discount, encoding decreasing user attention at deeper ranks.
- $\text{rank}(i)$ — position of item $i$ in the recommendation list.

**User-side fairness — User Group Fairness (UGF, $\downarrow$ better)** [Li et al., 2021]:

$$\text{UGF} \;=\; \left| \frac{1}{|Z_1|}\sum_{i \in Z_1} M(W_i) \;-\; \frac{1}{|Z_2|}\sum_{i \in Z_2} M(W_i) \right|$$

where:
- $Z_1, Z_2$ — two user groups (e.g. advantaged vs. disadvantaged).
- $M(W_i)$ — a quality metric for user $i$'s recommendation list $W_i$ (e.g. F1@10, [[NDCG]]).
- UGF is the **absolute gap** in mean performance between groups; $0$ = perfectly fair.

**Item-side fairness goals** (choose by goal × #groups). *Statistical parity* — equal exposure regardless of merit:

$$\text{DP} = \frac{\text{Exposure}(G_0)}{\text{Exposure}(G_1)}, \qquad \text{MinMaxRatio} = \frac{\min_{g\in G}\text{Exposure}(g)}{\max_{g\in G}\text{Exposure}(g)}, \qquad \text{MMF} = \min_{g\in G}\frac{\text{Exposure}(g)}{\text{Weight}(g)}$$

where:
- DP (Demographic Parity, two groups) — exposure ratio; $1.0$ = parity.
- MinMaxRatio ($\uparrow$, multi-group) — worst-to-best exposure ratio.
- MMF (Max-Min Fairness, $\uparrow$, multi-group) — weight-normalized exposure of the **most disadvantaged** group; $\text{Weight}(g)$ = group size or a quality-based value.

*Equality of opportunity* — exposure should be **proportional to utility/merit** $Y(G)$ (relevance offline):

$$\text{EUR} = \frac{\epsilon(G^+)/Y(G^+)}{\epsilon(G^-)/Y(G^-)}, \qquad \text{EEL} = \lVert \epsilon - \epsilon^* \rVert_2^2, \qquad \text{IAA} = \sum_{i=1}^{n} |A_i - R_i|$$

where:
- $\epsilon(G)$ — exposure of group $G$; $Y(G)$ — utility (summed relevance); $G^+,G^-$ — advantaged/disadvantaged. EUR/RUR target a ratio of $1.0$ (RUR replaces $\epsilon$ with realized click-through $\Gamma(G)$).
- EEL ($\downarrow$, Expected Exposure Loss) — squared distance between system exposure $\epsilon$ and **target** exposure $\epsilon^*$.
- IAA ($\downarrow$, Inequity of Amortized Attention) — $L_1$ distance between attention $A_i$ and predicted relevance $R_i$ per item.

## Key Properties / Variants

- **Goal × #groups taxonomy** (which metric to report):
  - *Statistical parity* → DP, MinMaxRatio, MMF.
  - *Equality of opportunity* → EUR, RUR, EEL, IAA.
  - *Two groups* → DP, EUR, RUR. *Multiple groups* → MinMaxRatio, MMF, EEL, IAA.
- **Three intervention stages** (where you inject fairness):
  - **Pre-processing** — debias the data before training (causal, probabilistic mapping). *Not supported for the recommendation task in FairDiverse.*
  - **In-processing** — modify the training objective:
    - *Re-weight / re-sample*: up-weight the disadvantaged group's loss. Weighted loss $\mathcal{L} = w_{\text{maj}}\mathcal{L}_{\text{maj}} + w_{\text{min}}\mathcal{L}_{\text{min}}$ with $w_{\text{min}}\!\uparrow$ (e.g. inverse-propensity weighting, dual-mirror descent for MMF).
    - *Regularizer*: add a fairness penalty, $\mathcal{L} = \mathcal{L}_{\text{relevance}} + \lambda\,\mathcal{L}_{\text{fairness}}$, where $\lambda$ trades accuracy for fairness.
    - *Prompt-based*: fairness-aware prompts for [[LLM-based Recommendation|LLM-based]] recommenders.
  - **Post-processing** — re-rank the output list to meet constraints (greedy/knapsack heuristics; learning-based fair scores). Cost is measured by **Utility Loss** $= \text{Utility}_{\text{ori}} - \text{Utility}_{\text{fair}}$.
- **Trade-offs.** Fairness vs. [[Diversity|accuracy]] is a multi-objective problem: constraints can lower accuracy and even **reinforce stereotypes** if low-utility items are force-promoted. But win-win cases exist (e.g. repeat-biased next-basket methods can be both more accurate and more item-fair).
- **Societal stakes.** Under-exposure (providers leave the platform), eroded user trust, [[Echo Chamber|echo chambers]]/polarization, and economic inequality (e.g. fewer high-paying job ads shown to women).

```pseudo
Algorithm: In-processing fair training (re-weight) vs Post-processing re-rank
─────────────────────────────────────────────────────────────────────────────
# (A) In-processing: fairness embedded in the loss
for each batch:
    L_relevance = ranking_loss(scores, labels)        # e.g. BPR / BCE
    L_fairness  = group_disparity(Exposure(groups))   # e.g. MMF gap
    L = L_relevance + λ · L_fairness                   # λ = fairness strength
    θ ← θ − α ∇θ L

# (B) Post-processing: re-rank a fixed relevance list under a fairness goal
ranked ← sort items by relevance                       # original list
re_ranked ← greedy_select(ranked, constraint=MMF/DP)   # inject minority items
report  Utility_Loss = Utility(ranked) − Utility(re_ranked)
        + fairness metric (DP / MMF / EEL ...)
```

## Connections

- Side of: [[Beyond-Accuracy Metrics]] (alongside [[Diversity]], [[Serendipity]], [[Novelty]], [[Coverage]])
- Root causes: [[Popularity Bias]], [[Long Tail]], [[Position Bias]]
- Exposure depends on: [[NDCG]]-style position discount / browsing model
- Two sides: [[User Fairness]], [[Item Fairness]] / [[Provider Fairness]] / [[Exposure Fairness]]
- General lens: [[Algorithmic Fairness]], [[Fairness in Ranking]]
- Downstream harms: [[Filter Bubble]], [[Echo Chamber]]
- Measured offline vs. online: [[Offline Evaluation]], [[A/B Testing]]
- Optimized via: Multi-Objective Recommendation trade-offs

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
