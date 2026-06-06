---
type: concept
aliases: [Online Evaluation, Offline Evaluation]
course: [RecSys]
tags: [evaluation, exam-topic]
status: complete
---

# Online and Offline Evaluation

## Definition

> [!definition] Online and Offline Evaluation
> **Evaluation** measures the **quality** and **effectiveness** of a recommender system in order to (i) **identify** strengths/weaknesses of an algorithm, (ii) **compare** algorithms, and (iii) **guide** design and optimization. Two main paradigms (plus simulation as a third) answer the question differently:
> - **Offline evaluation** uses **pre-collected historical (log) data** — ratings, purchase history, click logs. The model is trained on a training split and scored against held-out interactions with offline metrics ([[Recall]], [[NDCG]], [[MRR]], …). No deployed system or real users are needed.
> - **Online evaluation** assesses a recommender **in real time with actual user interactions**, by **deploying it in a live environment**. The canonical method is **[[A/B Testing]]**: randomly split traffic into a **control** group (current system) and one or more **test** groups (new system), run for a period, and compare a business objective via **significance testing**.

## Intuition

> [!intuition] Cheap proxy vs. real ground truth
> Offline evaluation is a **fast, cheap, repeatable proxy**: you can screen hundreds of algorithm variants on a fixed log without touching production. But the log is a **frozen snapshot of past behavior under the old system**, so it cannot capture how users would react to *new* recommendations, and it cannot measure business outcomes like satisfaction, revenue, or engagement.
>
> Online evaluation is the **real ground truth** — it observes actual user response to the live system — but it is **slow, expensive, and risky** (a bad test group can lose revenue or hurt user experience). The practical workflow is therefore a funnel: prune aggressively offline, then validate the few survivors online with A/B testing. **Simulation** sits between the two, replacing real users with a learned user-choice model when online testing is too costly or when log data is too sparse/biased.

## Mathematical Formulation

The core of online evaluation is **[[A/B Testing]]**: estimate the difference in an objective metric $M$ between the test variant $B$ and the control variant $A$, where users are randomly assigned to groups $\mathcal{G}_A$ and $\mathcal{G}_B$.

$$\widehat{\Delta} = \bar{M}_B - \bar{M}_A = \frac{1}{|\mathcal{G}_B|}\sum_{u \in \mathcal{G}_B} M_u \;-\; \frac{1}{|\mathcal{G}_A|}\sum_{u \in \mathcal{G}_A} M_u$$

A two-sample statistic decides whether $\widehat{\Delta}$ is **statistically meaningful** rather than random variation:

$$z = \frac{\bar{M}_B - \bar{M}_A}{\sqrt{\dfrac{s_A^2}{|\mathcal{G}_A|} + \dfrac{s_B^2}{|\mathcal{G}_B|}}}$$

where:
- $M_u$ — the objective measured per user (e.g. conversion, revenue, clicks/engagement)
- $\bar{M}_A, \bar{M}_B$ — mean objective in control / test groups
- $\mathcal{G}_A, \mathcal{G}_B$ — randomly assigned, (typically) equal-sized user groups
- $s_A^2, s_B^2$ — sample variances of the objective in each group
- $z$ — test statistic; large $|z|$ ⇒ reject the null that $\Delta = 0$ ⇒ the difference is significant

Offline evaluation, by contrast, scores a ranking against a held-out set with a fixed metric. A representative example is **[[Recall]]** on the held-out relevant items:

$$\mathrm{Recall} = \frac{\#\{\text{correctly recommended relevant items}\}}{\#\{\text{relevant items}\}}$$

where the "relevant items" come entirely from the **logged** historical interactions, never from live user response.

## Key Properties / Variants

- **Offline — pros:** does **not** require a deployed system or real users; **fast and convenient** for testing many algorithms; reproducible on a fixed dataset (the default setting for most RecSys research).
- **Offline — cons:** relies on **pre-collected historical data** that may not reflect real-time behavior (e.g. the last coffee machine in a purchase history may be obsolete later); **cannot measure business metrics** such as user satisfaction or revenue; the log can be **biased** (e.g. [[Position Bias]] — attention concentrated on top results), so offline scores can be misleading.
- **Online — pros:** measures **real user response** in a live environment; supports **continuous monitoring and improvement**; can directly optimize business objectives (conversion, engagement, revenue).
- **Online — cons:** **risky and costly** (a poor variant harms revenue/UX); needs enough traffic and time to reach **statistical significance**; slower iteration than offline.
- **A/B testing essentials:** (1) define the **objective**; (2) **randomly** assign users to control vs. test group(s); (3) run long enough to collect data for **significance testing**; promote the winner only if results are significant and promising, otherwise explore alternatives.
- **Simulation (third paradigm):** replaces real users with a learned **user-choice model** built from a user–item rating matrix populated by logged data. Used when online testing is **too costly/risky** or when historical data is **insufficient or biased** (synthesize an unbiased dataset for unbiased evaluation). Findings from semi-synthetic data should still be validated with real-world testing. Frameworks: RecoGym, RecSim.

A/B test decision loop:

```pseudo
Algorithm: A/B Test (Online Evaluation)
────────────────────────────────────────
Define objective metric M (e.g. conversion, revenue, engagement)
Randomly split live traffic:
    control group G_A  ← current system  (variant A)
    test group(s) G_B  ← new system(s)   (variant B)

Loop while not enough data:
    serve A to users in G_A, serve B to users in G_B
    log M_u for each user u

Compute  Δ̂ = mean(M | G_B) - mean(M | G_A)
Run significance test on Δ̂   (e.g. z-test, p-value)
if Δ̂ > 0 and significant:
    promote / further optimize variant B
else:
    keep A, explore alternative strategies
```

## Connections

- Measured with: [[Recall]], [[Precision]], [[NDCG]], [[MRR]], [[MAP]], [[Hit Ratio]], [[Beyond-Accuracy Metrics]]
- Online method: [[A/B Testing]]
- Confounded by: [[Position Bias]], [[Popularity Bias]] (sources of log bias motivating simulation / debiasing)
- Related families: [[Offline Evaluation]], [[Online Evaluation]] (sibling aliases of this note), [[Cranfield Paradigm]] (offline-evaluation tradition in [[Information Retrieval]])
- Goal of recommendation evaluated: [[Top-K Recommendation]] for a [[Recommender System]]
- Beyond accuracy: [[Fairness in Recommendation]], [[Diversity]] also evaluated offline/online

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
