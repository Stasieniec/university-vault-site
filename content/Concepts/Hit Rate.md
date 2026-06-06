---
type: concept
aliases: [HR@K, Hit Ratio, Hit-rate]
course: [RecSys]
tags: [evaluation, collaborative-filtering, sequential-rec, exam-topic]
status: complete
---

# Hit Rate

## Definition

> [!definition] Hit Rate (HR@K)
> **Hit Rate** (also Hit Ratio, HR@K) is a **set-based / non-ranking accuracy metric** for [[Top-K Recommendation|top-K recommendation]]. It is the **fraction of users for whom at least one relevant item appears in their top-K list**. A user is scored a binary "hit" (1) if any relevant item makes it into the top-K, and a "miss" (0) otherwise; HR@K averages this indicator over all users.
> 
> Because it only asks "was *there* a relevant item in the cutoff?", HR@K ignores **how many** relevant items were retrieved and **where** in the list they sit — it is rank-insensitive within the top-K.

## Intuition

> [!intuition] Did we get *anything* right?
> Hit Rate answers the most forgiving question an evaluator can ask: *for each user, did we surface even one good item?* It is essentially **per-user recall thresholded at "≥ 1 hit"**, then averaged across users.
> 
> Worked example from lecture (Hit Rate @ 3, three users): User 1 has a relevant item in their top-3 → hit; User 2 has none → miss; User 3 has one → hit. So $\text{HR@3} = 2/3 \approx 0.67$.
> 
> This makes it a natural fit for **next-item / sequential recommendation**, where each test instance has exactly one ground-truth "next item": HR@K then literally means "did the held-out next item land in the top-K?" — the leave-one-out hit rate reported for SASRec, GRU4Rec, and BERT4Rec (e.g. HR@10, HR@100).

## Mathematical Formulation

> [!formula] Hit Rate at K
> $$\text{HR@K} = \frac{1}{|U|} \sum_{u \in U} \mathbf{1}\!\left( \text{Rel}_u \cap \text{TopK}_u \neq \emptyset \right)$$
> 
> where:
> - $U$ — the set of (test) users; $|U|$ — number of users evaluated
> - $\text{Rel}_u$ — the set of relevant (ground-truth) items for user $u$
> - $\text{TopK}_u$ — the $K$ highest-ranked items the system recommends to user $u$
> - $\mathbf{1}(\cdot)$ — indicator function, returning $1$ if the intersection is non-empty (≥ 1 relevant item in the top-K), else $0$
> - $\text{Rel}_u \cap \text{TopK}_u \neq \emptyset$ — the "hit" condition for user $u$

In the common **leave-one-out** sequential setting, $\text{Rel}_u = \{i_{n+1}\}$ (a single held-out next item), so the indicator reduces to $\mathbf{1}(i_{n+1} \in \text{TopK}_u)$ and HR@K equals the proportion of test users whose true next item is recovered in the top-K.

## Key Properties / Variants

- **Range and direction:** $\text{HR@K} \in [0,1]$, higher is better (↑). HR is monotonically **non-decreasing in $K$**: enlarging the cutoff can only add hits.
- **Relation to [[Recall]]:** HR@K is the binarized, per-user version of recall. When each user has exactly **one** relevant item, HR@K and Recall@K coincide. With multiple relevant items, recall measures the *fraction* retrieved while HR only checks for *at least one* — so HR ≥ Recall in that regime.
- **Rank-insensitive:** like [[Recall]], it cannot distinguish a list with the relevant item at rank 1 from one with it at rank K. Lecture motivates rank-aware metrics ([[MRR]], [[NDCG]]) precisely because set-based metrics such as HR give the **same score** to two lists that place relevant items at different positions.
- **Family:** one of the accuracy-based **non-ranking / position-agnostic** metrics — grouped with [[Precision]], [[Recall]], and [[F1-Score]] — as opposed to ranking metrics ([[NDCG]], [[MRR]], [[MAP]], AUC).
- **Reported standard:** HR appears as a core accuracy metric in evaluation toolkits (e.g. FairDiverse lists MRR, Hit Ratio (HR), and NDCG), typically alongside NDCG@K.

```pseudo
Function: HitRate@K
──────────────────────────────────────────────
Input: users U, relevant sets Rel_u, ranked recs for each u, cutoff K
hits ← 0
for each user u in U:
    topK ← first K items of ranked_recs[u]
    if (Rel_u ∩ topK) is non-empty:
        hits ← hits + 1
return hits / |U|
```

## Connections

- Binarized form of: [[Recall]] (per-user, thresholded at ≥ 1 hit)
- Sibling non-ranking metrics: [[Precision]], [[F1-Score]]
- Contrasted with rank-aware metrics: [[MRR]], [[NDCG]], [[MAP]]
- Evaluation context: [[Top-K Recommendation]], [[Offline Evaluation]], [[Beyond-Accuracy Metrics]]
- Reported for: [[SASRec]], [[GRU4Rec]], [[BERT4Rec]] (leave-one-out next-item HR@K)

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03a - Sequential Recommendation Models]]
