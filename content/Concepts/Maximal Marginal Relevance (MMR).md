---
type: concept
aliases: [MMR, Maximal Marginal Relevance]
course: [RecSys]
tags: [evaluation, sequential-rec, generative-rec, exam-topic]
status: complete
---

# Maximal Marginal Relevance (MMR)

## Definition

> [!definition] Maximal Marginal Relevance
> **MMR** is a **greedy post-hoc re-ranking** method that builds a recommendation (or retrieval) list **one item at a time**, at each step picking the item that maximizes a linear trade-off between its **relevance** to the query/user and its **redundancy** (maximum similarity) with the items **already selected**. It is the classic technique for injecting [[Diversity]] into a ranked list without retraining the underlying scorer — originally proposed by Carbonell & Goldstein (SIGIR 1998).

## Intuition

> [!intuition] Relevant but not redundant
> A pure relevance ranking tends to stack near-identical items at the top (all five "action thrillers"). MMR fixes this **at decoding/re-ranking time**: after you have picked a few items, the *marginal* value of yet another near-duplicate is low. MMR re-scores each remaining candidate as "how relevant is it" **minus** "how similar is it to what I already chose," so the next pick is relevant *and* adds something new.
>
> In the [[RS-L04 - Generative Recommendation]] decoding pipeline this is exactly the "re-rank afterwards" knob: after [[Beam Search]] returns look-alike items sharing a [[Semantic ID]] prefix (e.g. all `(12, 48, ·)`), MMR rebuilds the list preferring items *unlike* those already chosen, so the user actually has a choice.

## Mathematical Formulation

At each step, given the query/user representation $q$, the set of candidate items $R$, and the set of **already-selected** items $S$, the next item is:

$$\text{MMR} = \arg\max_{d_i \in R \setminus S} \Big[\; \lambda \cdot \text{Sim}_1(d_i, q) \;-\; (1-\lambda)\cdot \max_{d_j \in S}\,\text{Sim}_2(d_i, d_j) \;\Big]$$

where:
- $d_i$ — a candidate item not yet selected ($d_i \in R \setminus S$)
- $q$ — the query / user-history representation (relevance target)
- $\text{Sim}_1(d_i, q)$ — relevance score of candidate $i$ (e.g. the recommender's predicted score, or the generative model's log-likelihood $\log p_\theta(\mathbf{z}_i \mid \mathbf{x})$)
- $\text{Sim}_2(d_i, d_j)$ — similarity between candidate $i$ and an already-selected item $j$ (e.g. cosine similarity of item embeddings, or shared category)
- $\max_{d_j \in S}\,\text{Sim}_2(d_i, d_j)$ — the **redundancy** term: the candidate's similarity to its *nearest* already-chosen item (its worst-case overlap)
- $\lambda \in [0, 1]$ — trade-off knob: $\lambda = 1$ recovers pure relevance ranking; $\lambda = 0$ ranks purely for novelty/diversity; intermediate $\lambda$ balances the two
- $S$ — the partial output list built so far (starts empty)

The first item is the most relevant one (the max over $S = \emptyset$ contributes nothing); every later item is chosen by the full criterion above.

## Key Properties / Variants

- **Greedy, not globally optimal:** MMR is a heuristic. It maximizes marginal gain step by step and does *not* solve the combinatorial "best diverse set of size K" problem (which is NP-hard, related to max-sum/max-min dispersion). It runs in roughly $O(K \cdot |R|)$ similarity evaluations.
- **Post-processing intervention:** in the [[RS-L02 - Evaluation Beyond Accuracy]] taxonomy MMR is a **heuristic post-processing re-ranker** — it operates on the output ranked list and never touches training. This is the same "greedy search to re-rank items" family as the heuristic methods in the FairDiverse toolkit.
- **Accuracy–diversity trade-off:** raising diversity (lowering $\lambda$) typically costs a little top-1 accuracy but improves [[Intra-List Distance]] / [[Catalog Coverage]] / [[Novelty]]. This is the canonical Diversity-vs-Accuracy trade-off; $\lambda$ is the explicit dial.
- **Choice of $\text{Sim}_2$ defines "diversity":** embedding cosine gives content diversity; category/genre overlap gives categorical diversity. The metric you optimize against (e.g. [[Intra-List Distance|ILD]]) should match the similarity you penalize.
- **MMR in GenRec decoding:** one of two decoding-time diversity remedies in generative recommendation (the other being sampling / temperature and diverse beam search). It complements, rather than replaces, fixing diversity at the tokenizer (so popular items do not collapse onto a shared prefix) or rewarding diversity in RL fine-tuning.
- **Relation to fairness re-ranking:** structurally identical to exposure-fairness re-rankers that add a per-step penalty/score to the relevance score; MMR's penalty targets *redundancy*, fairness re-rankers' penalty targets *under-exposed groups*.

Greedy MMR re-ranking, building a list of size $K$:

```pseudo
Algorithm: MMR Re-Ranking
─────────────────────────────────────────────
Input:  candidate set R, query q, λ ∈ [0,1], list size K
Output: ordered list S

S ← []                                  # selected items (in order)
while |S| < K and R is not empty:
    for each d_i in R:                  # score every remaining candidate
        rel_i ← Sim1(d_i, q)
        if S is empty:
            red_i ← 0
        else:
            red_i ← max over d_j in S of Sim2(d_i, d_j)   # redundancy
        mmr_i ← λ * rel_i − (1 − λ) * red_i
    d* ← argmax_{d_i in R} mmr_i
    append d* to S
    remove d* from R
return S
```

## Connections

- Adds: [[Diversity]] (the objective MMR optimizes for) — measured by [[Intra-List Distance]], [[Novelty]], [[Catalog Coverage]]
- Trades against: [[Top-K Recommendation]] accuracy ([[NDCG]] / [[Recall]]) — the Diversity-vs-Accuracy trade-off
- Applied as: a heuristic post-processing **re-ranking** step (see [[Reranking]])
- Used in GenRec to fix homogeneous [[Beam Search]] output over [[Semantic ID|Semantic IDs]]
- Sibling re-rankers: exposure-based [[Fairness in Recommendation]] post-processors (same per-step penalty structure)

## Appears In

- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L04 - Generative Recommendation]]
