---
type: concept
aliases: []
course: [RecSys]
tags: [evaluation, fairness, exam-topic]
status: complete
---

# A/B Testing

## Definition

> [!definition] A/B Testing
> **A/B testing** is the standard method for **[[Online Evaluation]]** of a recommender system: a live experiment that compares a new system (variant) against the currently deployed one (control) by **randomly splitting live traffic** into groups and measuring a business objective on each. The **control group** receives the current production system; one or more **test groups** receive the new version(s). Observed differences are then subjected to **significance testing** to decide whether the new system is genuinely better or the gap is just random variation.

## Intuition

> [!intuition] Why split live traffic instead of replaying logs?
> [[Offline Evaluation]] reuses pre-collected historical logs — it is fast and cheap and lets you screen many algorithms, but it cannot capture real-time user behavior and cannot measure true **business metrics** like satisfaction, conversion, or revenue (the logged "best" item may already be stale). A/B testing puts both systems in front of **real users at the same time**. Because users are assigned **randomly**, the two groups are statistically equivalent in expectation, so any systematic difference in the outcome metric is **causally attributable** to the system change rather than to confounds (seasonality, user-mix, etc.). The price is that experimenting on live users is **risky and costly** (a bad variant can lose revenue or harm experience) and you must run long enough to accumulate the sample size needed for a statistically meaningful conclusion.

## Mathematical Formulation

The experiment estimates the difference in a chosen metric between the test variant $B$ and the control $A$. With $n_A$ control users and $n_B$ test users, and per-user outcome $y_{u}$ (e.g. conversion 0/1, revenue, or engagement), the group means are

$$\hat{\mu}_A = \frac{1}{n_A}\sum_{u \in A} y_u, \qquad \hat{\mu}_B = \frac{1}{n_B}\sum_{u \in B} y_u, \qquad \widehat{\Delta} = \hat{\mu}_B - \hat{\mu}_A$$

To decide whether $\widehat{\Delta}$ is real, test the null hypothesis $H_0: \mu_A = \mu_B$ against $H_1: \mu_A \neq \mu_B$ using a two-sample statistic:

$$z = \frac{\hat{\mu}_B - \hat{\mu}_A}{\sqrt{\dfrac{\hat{\sigma}_A^2}{n_A} + \dfrac{\hat{\sigma}_B^2}{n_B}}}$$

where:
- $\hat{\mu}_A, \hat{\mu}_B$ — observed mean of the objective metric in control and test groups
- $\widehat{\Delta}$ — measured treatment effect (lift) of the new system
- $\hat{\sigma}_A^2, \hat{\sigma}_B^2$ — sample variances of the outcome within each group
- $n_A, n_B$ — number of users randomly assigned to each group
- $z$ — test statistic compared against a threshold; if the associated $p$-value $< \alpha$ (e.g. $\alpha = 0.05$) we reject $H_0$ and declare the difference statistically significant

The randomized assignment is what licenses the causal reading: each user $u$ is independently routed by $g(u) \sim \text{Uniform}\{A, B\}$ (often via a hash of the user ID), making the groups exchangeable so that $\mathbb{E}[\widehat{\Delta}]$ equals the true effect of the system change.

## Key Properties / Variants

- **Procedure (as taught):**
  1. **Define the objective** — pick the specific metric to compare: conversion rate, revenue generated, user engagement, or another business-specific KPI.
  2. **Identify test groups** — **randomly** assign users to a **control group** (current system) and one or more **test group(s)** (new version(s)).
  3. **Run the test** — serve each group its variant for some time (this can be risky on live users) and collect enough data for **significance testing**.
  4. **Decide** — if the new algorithm shows promising, statistically significant results it can be further optimized and rolled out to production; otherwise explore alternatives.
- **A/B/n testing:** more than two arms (control + several test variants) compared simultaneously; needs correction for multiple comparisons.
- **Sample size / run time:** must be large/long enough that the test has the statistical power to detect the expected lift; too short → underpowered, noisy conclusions.
- **Cost vs. risk trade-off:** unlike offline evaluation, A/B testing exposes real users to a possibly worse system, so it can lose revenue or hurt user experience — motivating cheaper proxies first.
- **Relation to the evaluation taxonomy:** A/B testing is the concrete realization of **online evaluation**, sitting alongside **offline evaluation** (historical logs) and **simulation** (learned user-choice models such as Recogym / Recsim) as the three evaluation paradigms. Simulation is often used precisely to avoid the cost and risk of online A/B tests.
- **What it can and cannot measure:** it directly captures real-world business metrics (satisfaction, CTR, revenue) that offline metrics cannot, but it cannot cheaply screen many candidate algorithms — that is what offline evaluation is for.

```pseudo
Algorithm: A/B Test of a New Recommender
─────────────────────────────────────────
Define objective metric y (e.g. conversion, revenue, engagement)
Choose significance level α and target effect size; derive required n

For each incoming user u (live traffic):
    g ← random_assign(u)          # uniform / hashed → {A, B}
    if g == A: serve CONTROL system, log y_u
    else:      serve TEST system,    log y_u

After collecting n_A, n_B users:
    μ_A, μ_B ← group means of y
    z ← (μ_B − μ_A) / sqrt(σ²_A/n_A + σ²_B/n_B)
    p ← significance_test(z)
    if p < α and μ_B > μ_A:
        ship TEST (optionally optimize further)
    else:
        keep CONTROL / explore alternatives
```

## Connections

- Realizes: [[Online Evaluation]] (the canonical online-evaluation method)
- Contrasted with: [[Offline Evaluation]] (historical logs, cheap, biased), and simulation-based evaluation
- Measures: business KPIs beyond accuracy — complements [[Recall]], [[MRR]], [[NDCG]] used offline
- Confounded by: [[Position Bias]] and other logged-data biases (a reason offline metrics and A/B outcomes can disagree)
- Stakes: fairness/diversity effects (e.g. [[Filter Bubble]], [[Popularity Bias]]) may only surface on real users over time

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
