---
type: concept
aliases: [Content-based Recommendation, Content-Based Filtering]
course: [RecSys]
tags: [collaborative-filtering, exam-topic]
status: complete
---

# Content-Based Recommendation

## Definition

> [!definition] Content-Based Recommendation
> **Content-based recommendation** (a.k.a. **content-based filtering**) recommends items to a user based on the **content/attributes of the items** the user has previously liked, rather than on the collective interaction data of other users. Each item is described by a **content profile** (features extracted from text, audio, images, metadata, etc.), and each user is described by a **user profile** built from the content of items they interacted with. Recommendation reduces to finding items whose content profile best **matches** the user's profile.
>
> It is one axis of the **content-vs-collaborative-vs-hybrid** paradigm: where [[Collaborative Filtering]] uses *who liked what*, content-based filtering uses *what the item is*.

## Intuition

> [!intuition] "More of what you already liked"
> If a user repeatedly listens to acoustic folk tracks, describe those tracks by their content (genre tags, audio features, lyrics), average them into a profile of the user's taste, and then recommend other tracks whose content is closest to that profile. No other users are needed — the system reasons purely about item attributes and the individual's own history. This is why content-based methods shine in domains like **news** (recency-driven, content-rich text) where fresh items have no interaction history yet.

## Mathematical Formulation

Represent every item $i$ by a content **feature vector** $\bar{c}_i \in \mathbb{R}^d$ (e.g., a [[TF-IDF]] vector over the words/tags describing the item). Build the **user profile** $\bar{p}_u$ as a (rating-weighted) aggregate of the content of items the user has interacted with:

$$\bar{p}_u = \frac{\sum_{i \in I_u} r_{ui}\,\bar{c}_i}{\sum_{i \in I_u} |r_{ui}|}$$

Score a candidate item by the **cosine similarity** between user profile and item content:

$$\hat{s}(u, i) = \cos(\bar{p}_u, \bar{c}_i) = \frac{\bar{p}_u \cdot \bar{c}_i}{\lVert \bar{p}_u \rVert \, \lVert \bar{c}_i \rVert}$$

where:
- $\bar{c}_i$ — content feature vector of item $i$ (e.g., [[TF-IDF]] weights over the terms describing item $i$; $d$ = vocabulary/feature size)
- $I_u$ — set of items user $u$ has interacted with (the user's history)
- $r_{ui}$ — feedback signal of $u$ on $i$ (explicit rating, or implicit $1$); positive feedback pulls the profile toward $\bar{c}_i$
- $\bar{p}_u$ — user profile, a weighted centroid of liked items' content in the same feature space
- $\hat{s}(u,i)$ — match score; items are ranked by $\hat{s}$ and the [[Top-K Recommendation|top-K]] are returned

This profile-then-cosine recipe is the classic linear instance (the [[Rocchio Algorithm]] applied to recommendation). More generally, the matcher can be any **classifier/regressor** $f_\theta(\bar{c}_i)$ trained **per user** on their history (e.g., naive Bayes, k-NN, or a neural model), predicting how relevant item content is to that user.

## Key Properties / Variants

- **No reliance on other users.** Scores depend only on the target user's own history + item content, which gives it complementary strengths to [[Collaborative Filtering]].
- **Item cold start is solved.** A brand-new item with **zero interactions** can still be recommended the moment its content features are known — crucial for news/music where the catalog churns fast.
- **User cold start persists.** A new user with an empty $I_u$ has no profile, so the method cannot personalize yet.
- **Feature engineering is the bottleneck.** Quality depends entirely on how well $\bar{c}_i$ captures what matters. Text → [[TF-IDF]] / [[Word Embeddings]]; audio/image/video → learned embeddings. Neural encoders let the model ingest **heterogeneous content** (text, images, audio) without hand-crafted features.
- **Over-specialization / filter bubble.** Because it only retrieves "more of the same," it has **low [[Diversity]]/[[Serendipity]]** — it struggles to surface genuinely novel items outside the user's established profile.
- **Transparent & explainable.** Recommendations are justifiable via shared content features ("recommended because it shares genre/keywords with X").
- **Hybrid use.** Usually combined with CF in a [[Hybrid Recommendation]] to get the best of both: content for cold items, collaborative signal for popular ones.

```pseudo
Algorithm: Content-Based Recommendation
────────────────────────────────────────
# 1. Item profiles (offline)
for each item i in catalog:
    c_i ← extract_features(i)        # e.g. TF-IDF over text, or learned embedding

# 2. User profile (per user u)
p_u ← weighted_mean({ r_ui * c_i  for i in history(u) })

# 3. Scoring & ranking (online, for target user u)
for each candidate item i not in history(u):
    score[i] ← cosine(p_u, c_i)       # or f_θ(c_i) for a trained matcher
return top-K items by score[i]
```

## Connections

- Paradigm sibling of: [[Collaborative Filtering]] (interaction-based) — content-based is the *content* axis
- Combined with CF in: [[Hybrid Recommendation]]
- Linear instance is essentially: [[Rocchio Algorithm]] (profile + cosine matching)
- Feature representations: [[TF-IDF]], [[Word Embeddings]], [[Vector Space Model]]
- Addresses item side of: [[Cold Start Problem]]
- Weakness motivates: [[Diversity]], [[Serendipity]], [[Novelty]] (beyond-accuracy)
- Ranked output evaluated with: [[Recall]], [[MRR]], [[NDCG]]

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
