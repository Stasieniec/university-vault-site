---
type: concept
aliases: [ULTR]
course: [IR]
tags: [unbiased-ltr, evaluation, neural-ir, exam-topic]
status: complete
---

# Unbiased Learning to Rank

## Definition

> [!definition] Unbiased Learning to Rank (ULTR)
> **Unbiased Learning to Rank** is the family of methods that train a ranking model from **implicit feedback** (clicks) while correcting for the systematic biases that clicks carry — above all [[Position Bias]]. The goal is to recover the **true relevance** signal $\text{Rel}_d$ from observed clicks, which are a *biased* function of relevance, position, and presentation.
>
> The defining contract: produce an estimator of ranking quality (e.g. DCG) whose **expectation over clicks equals the true relevance-based metric**, so that minimising the empirical loss optimises the model toward genuine relevance rather than toward "whatever the logging system already ranked highly."

## Intuition

Clicks are cheap and abundant, but they lie. A user clicks an item only if they **examine** it and find it **relevant** (the [[Examination Hypothesis]]). Because users scan top-down, items at high ranks are examined far more often, so they collect clicks regardless of true relevance. Training naively on raw clicks therefore teaches the model to reproduce the **logging policy's ranking**, not relevance — a self-reinforcing feedback loop.

ULTR breaks the loop by treating clicks as observations drawn through a known **bias model** (the propensities). If we know *how* a click was distorted, we can mathematically *undo* the distortion. The workhorse is [[Inverse Propensity Weighting]]: a click at a rarely-examined low rank is worth more relevance evidence than a click at rank 1, because it survived a much harsher examination filter.

Two complementary stances:
- **Counterfactual / offline**: reuse historical logs from a fixed logging policy and reweight ([[Counterfactual Learning to Rank]]).
- **Online**: actively intervene (randomise) to *measure* propensities, then learn.

## Mathematical Formulation

**Click generation (PBM).** Under the [[Examination Hypothesis]] with the [[Position-Based Click Model]], a click on document $d$ shown at rank $k$ factorises:

$$P(C_{d,k}=1) = \underbrace{P(E_k=1)}_{\text{examination / propensity}} \cdot \underbrace{P(R_d=1)}_{\text{true relevance}}$$

where:
- $C_{d,k}$ — observed click on $d$ at rank $k$ (binary)
- $E_k$ — examination event, depends only on rank $k$ (position bias)
- $R_d$ — true relevance of $d$ to the query, independent of rank
- $p_k := P(E_k=1)$ — the **propensity** at rank $k$

**Unbiased relevance estimate (IPW).** Dividing the click by its propensity removes the position-dependent factor in expectation:

$$\widehat{\text{Rel}}_d = \frac{C_{d,k}}{p_k}, \qquad \mathbb{E}[\widehat{\text{Rel}}_d] = \frac{P(E_k)\,P(R_d)}{p_k} = P(R_d)$$

**IPW additive ranking loss.** A model $f_\theta$ producing scores is trained on the clicked documents only, each weighted by its inverse propensity. For an additive metric (rank-based loss $\Delta$, e.g. the rank of $d$ in $f_\theta$'s ranking):

$$\mathcal{L}_{\text{IPW}}(f_\theta) = \frac{1}{|\mathcal{Q}|}\sum_{q}\sum_{d:\,C_{d}=1} \frac{\Delta\big(d \mid f_\theta(q)\big)}{p_{k(d)}}$$

where:
- $\mathcal{Q}$ — set of logged query sessions
- $C_d=1$ — documents that received a click in that session
- $k(d)$ — the rank at which $d$ was displayed by the logging policy
- $\Delta(d\mid f_\theta(q))$ — loss for placing the relevant $d$ at the rank assigned by $f_\theta$ (e.g. a [[Pairwise Learning to Rank]] hinge / [[LambdaRank]]-style $\Delta$NDCG term)
- $p_{k(d)}$ — examination propensity at the **displayed** rank

> [!formula] Unbiasedness of the IPW loss
> $$\mathbb{E}_{C}\big[\mathcal{L}_{\text{IPW}}(f_\theta)\big] = \frac{1}{|\mathcal{Q}|}\sum_q \sum_{d:\,R_d=1} \Delta\big(d \mid f_\theta(q)\big)$$
>
> The expectation over clicks reduces to the **ideal full-information loss** computed on truly relevant documents — the $p_k$ exactly cancels the examination factor, so $\theta$ is optimised toward true relevance.

> [!warning] Positivity is mandatory
> Every displayed rank must have $p_k > 0$, otherwise the weight $1/p_k$ is undefined and that position contributes no gradient. Top-$k$ logging (ranks beyond $k$ never shown) violates this; it forces either stochastic logging policies, propensity clipping, or a [[Doubly Robust Estimation]] fallback.

## Key Properties / Variants

- **Bias–variance tension.** IPW is unbiased but high-variance: small $p_k$ at deep ranks inflates weights and amplifies click noise. Propensity **clipping** $\min(1/p_k, \tau)$ trades a little bias for large variance reduction.
- **Where the propensities come from.**
  - *Online randomisation* (gold standard): RandTop-$k$ or RandPair swaps measure $p_k$ directly.
  - *Intervention harvesting*: mine naturally-occurring rank swaps in historical A/B logs.
  - *Jointly learned (Dual Learning)*: estimate the propensity model and the ranker together via EM-style alternation, no interventions needed.
- **User-model variants beyond PBM.** PBM assumes examination depends only on rank. Richer biases need richer models: [[Cascading Position Bias]] (sequential scan that stops on satisfaction), [[Trust Bias]] (top items clicked even when irrelevant), [[Item Selection Bias]] (items never shown get no feedback), [[Outlier Bias]] (visually distinctive items over-examined).
- **Doubly Robust ULTR.** Combine a learned relevance/imputation model $\hat r$ with the IPW correction: unbiased if *either* the propensities *or* the imputation model is correct, and lower variance than pure IPW ([[Doubly Robust Estimation]]).
- **Relation to causal/off-policy RL.** ULTR is off-policy evaluation: the logging policy is the behaviour policy, the new ranker is the target policy, and IPW is [[Importance Sampling]] over rankings.

```pseudo
Algorithm: Counterfactual ULTR with IPW (offline)
──────────────────────────────────────────────────
Input: click logs {(q, ranking y0, clicks c)}, propensities p_k
Output: ranking model f_θ

# 1) Estimate / load propensities
Estimate p_k for each rank k          # randomization OR intervention harvesting
                                       # OR jointly via EM (Dual Learning)

# 2) Train ranker on IPW-weighted clicks
Initialize θ
Loop until converged:
  Sample session (q, y0, c) from logs
  ranking ← f_θ(q)                     # current model's ranking
  L ← 0
  For each clicked d in session:
    k ← rank of d in displayed y0
    w ← min(1 / p_k, τ)                # inverse propensity, clipped
    L ← L + w · Δ(d | ranking)         # pairwise / ΔNDCG loss term
  θ ← θ − α · ∇_θ L                     # gradient step (SGD)

# 3) Offline evaluation (unbiased DCG estimate)
DCG_IPW(f) ← Σ_sessions Σ_{d: c_d=1}  c_d / (p_{k(d)} · log2(rank_f(d)+1))
return f_θ
```

## Connections

- Built on: [[Examination Hypothesis]], [[Position-Based Click Model]], [[Click Models]]
- Core correction: [[Inverse Propensity Weighting]] / [[Importance Sampling]]
- Offline framing: [[Counterfactual Learning to Rank]]
- Variance reduction: [[Doubly Robust Estimation]]
- Biases addressed: [[Position Bias]], [[Cascading Position Bias]], [[Trust Bias]], [[Item Selection Bias]], [[Outlier Bias]]
- Loss machinery inherited from: [[Learning to Rank]], [[Pairwise Learning to Rank]], [[LambdaRank]]

## Appears In

- [[IR-L11 - Unbiased Learning to Rank]]
- [[Learning to Rank]]
- [[Counterfactual Learning to Rank]]
- [[Cascading Position Bias]]
- [[Doubly Robust Estimation]]
- [[Examination Hypothesis]]
- [[Inverse Propensity Weighting]]
- [[Item Selection Bias]]
- [[Outlier Bias]]
- [[Position Bias]]
- [[Trust Bias]]
