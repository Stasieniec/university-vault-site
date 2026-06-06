---
type: concept
aliases: []
course: [RecSys]
tags: [evaluation, generative-rec, exam-topic]
status: complete
---

# Novelty

## Definition

> [!definition] Novelty
> **Novelty** is a beyond-accuracy evaluation criterion measuring the degree to which recommended items are **unknown to the user** and **different from what the user has seen before**. A novel recommendation exposes the user to something they would not have encountered or sought out on their own. It directly trades off against **repetition**: pure accuracy-optimal systems re-recommend familiar/popular items, which is correct but stale, whereas novelty rewards **discovery**. It is one of the standard beyond-accuracy objectives alongside [[Diversity]], [[Serendipity]], and [[Coverage]].

## Intuition

> [!intuition] Repetition vs. Discovery
> Users like consuming what they already know (a favorite song on repeat, re-reading a known author), but a recommender that *only* surfaces the already-known fails its purpose: in news recommendation, repeatedly suggesting articles the user has already read offers no new information and engagement decays; in music, only ever suggesting known songs/artists blocks discovery and users get bored.
>
> Novelty is the dimension that captures "have I seen this before?" — distinct from its neighbors:
> - **[[Diversity]]** = items in the list are different *from each other* (intra-list).
> - **Novelty** = items are different *from the user's own past* (or globally unpopular).
> - **[[Serendipity]]** = items are novel *and* relevant *and* surprising (the useful subset of novelty).
>
> So serendipity ⊂ novelty: a novel item can still be useless, but a serendipitous one must first be novel.

## Mathematical Formulation

The lecture defines novelty conceptually (RS-L02 slide 36) as items being unknown/unseen by the user. The standard operational measure (Vargas & Castells, RecSys 2011), used to make it computable, is **self-information / popularity-based novelty**: a recommended item is novel in proportion to how *unlikely* it is to be already known, which is estimated from its global popularity.

$$\text{Novelty}(R_u) = \frac{1}{|R_u|} \sum_{i \in R_u} -\log_2 p(i), \qquad p(i) = \frac{|\{u' : i \in I_{u'}\}|}{|U|}$$

where:
- $R_u$ — the list of items recommended to user $u$
- $p(i)$ — the probability that a randomly chosen user has interacted with item $i$ (its **popularity**); estimated as the fraction of users who consumed $i$
- $-\log_2 p(i)$ — the **self-information** (surprisal) of item $i$: popular items ($p(i) \to 1$) contribute $\approx 0$ novelty, rare long-tail items ($p(i) \to 0$) contribute large novelty
- $|U|$ — number of users; $|R_u|$ — recommendation list length

An alternative *user-relative* (unseen) formulation scores novelty by dissimilarity to the user's own history $I_u$ rather than to global popularity:

$$\text{Novelty}(R_u) = \frac{1}{|R_u|} \sum_{i \in R_u} \min_{j \in I_u} \text{distance}(i, j)$$

so an item far from everything the user has already consumed counts as novel. Both forms are **higher-is-better** (↑).

## Key Properties / Variants

- **Two reference points.** *Global novelty* (self-information, $-\log_2 p(i)$) measures novelty w.r.t. the whole user population; *personalized/unseen novelty* measures it w.r.t. the individual's history $I_u$. The lecture's verbal definition ("unknown to *the user*") is the personalized notion; the popularity form is the common computable proxy.
- **Long-tail link.** Novelty mechanically rewards items in the **[[Long-Tail Distribution|long tail]]** of the popularity curve. Pushing novelty therefore tends to fight **[[Popularity Bias]]** and improve **[[Catalogue Coverage]]** — though novelty (per-list) and coverage (catalog-wide) are not identical.
- **Accuracy trade-off.** Novelty conflicts with accuracy metrics like [[NDCG]]/[[Recall]]: the highest-probability next item is usually a familiar/popular one. RS-L02 frames this as a multi-objective trade-off; over-optimizing accuracy narrows recommendations into a **[[Filter Bubble]]**.
- **Relation to repetition research.** In sequential / next-basket recommendation the explore-vs-repeat balance is exactly this tension; repeat-biased methods can score higher on accuracy while explore-biased methods raise novelty.
- **Novelty in generative recommendation (RS-L04).** GenRec exposes a *structural* novelty problem at decode time:
  - **Beam search homogeneity / amplification bias** — items with similar [[Semantic IDs|semantic IDs]] share opening codes, so [[Beam Search]] locks onto a popular prefix and the top-$B$ list collapses into near-duplicates (low novelty), pruning long-tail prefixes early.
  - **Decode-time novelty fixes** — inject randomness (temperature/sampling), diverse beam search, or post-hoc [[Maximal Marginal Relevance (MMR)|MMR]] re-ranking.
  - **Train-time novelty fixes** — reward novelty/diversity inside the **[[Group Relative Policy Optimization|GRPO]]** group (penalize look-alike candidates), or shape the tokenizer (LETTER) so popular items do not all collapse onto one prefix.
  - **Evaluation caveat (RS-L04 slide 58).** Offline metrics (Recall@K, NDCG@K) credit only the exact logged click, so a genuinely novel-but-good recommendation the user never saw is scored as *wrong* — benchmarks **under-credit the very novelty GenRec is built to provide**.

```pseudo
Compute popularity-based novelty of a recommendation list R_u
─────────────────────────────────────────────────────────────
Precompute p(i) = (#users who interacted with i) / |U|   for all items i

novelty(R_u):
    s ← 0
    for each item i in R_u:
        s ← s + ( -log2( p(i) ) )      # self-information; rarer ⇒ larger
    return s / |R_u|                    # mean over the list (↑ better)
```

## Connections

- Beyond-accuracy sibling metrics: [[Diversity]], [[Serendipity]], [[Coverage]], [[Catalogue Coverage]]
- Driven by / fights: [[Long-Tail Distribution]], [[Popularity Bias]]
- Trades off against: [[NDCG]], [[Recall]] (accuracy); contributes to forming a [[Filter Bubble]] / [[Echo Chamber]] when ignored
- Multi-objective evaluation context: [[Beyond-Accuracy Metrics]], [[Fairness in Recommendation]]
- Generative-rec mechanisms affecting novelty: [[Beam Search]], [[Semantic IDs]], [[Group Relative Policy Optimization]], [[Maximal Marginal Relevance (MMR)]]
- Measured via: [[Shannon Entropy]] (self-information underlies the popularity form)

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L04 - Generative Recommendation]]
