---
type: concept
aliases: [PBM]
course: [IR]
tags: [click-models, unbiased-ltr, user-behavior, exam-topic]
status: complete
---

# Position-Based Click Model

## Definition

> [!definition] Position-Based Click Model (PBM)
> The **Position-Based Click Model** is the simplest practical [[Click Models|click model]]. It assumes a user clicks a document $d$ shown at rank $k$ **iff** the user **examines** rank $k$ **and** the document is **relevant** — and crucially that examination depends **only on the rank $k$**, never on the document, the query, or the surrounding results. It is the operational instantiation of the [[Examination Hypothesis]] in which the examination probability is a per-rank constant.

## Intuition

Think of each result position as having a fixed "visibility" determined purely by where it sits on the page. Rank 1 is almost always looked at; rank 8 is looked at far less. The PBM bakes this into a single number per rank, the **propensity** $P(\text{Exam}_k)$, and then treats *whether the user clicks given that they looked* as a clean measurement of relevance.

This factorization is what makes the model so useful: the two latent causes of a click — *did they see it?* (position) and *did they like it?* (relevance) — are assumed **independent**. Once you know the per-rank examination probabilities, an observed click becomes a noisy but **unbiased-after-reweighting** signal of relevance, which is exactly what [[Inverse Propensity Weighting|IPW]] exploits.

The contrast to keep in mind: PBM says examination of rank $k$ is the *same* regardless of what is above it. The [[cascade model]] / [[Cascading Position Bias]] says the opposite — examination depends on the relevance of everything above $k$.

## Mathematical Formulation

A click random variable $C_{d,k} \in \{0,1\}$ for document $d$ at rank $k$ factors into two independent Bernoulli events:

$$P(C_{d,k} = 1) \;=\; \underbrace{P(E_k = 1)}_{\text{examination (position only)}} \;\cdot\; \underbrace{P(R_d = 1 \mid q)}_{\text{relevance (position-free)}}$$

where:
- $C_{d,k}$ — observed click on document $d$ displayed at rank $k$
- $E_k \in \{0,1\}$ — **latent** examination event for rank $k$; $P(E_k=1)$ is the **propensity** $\theta_k$
- $R_d \in \{0,1\}$ — **latent** relevance of $d$ to query $q$; $P(R_d=1\mid q) = \gamma_d$
- $\theta_k = P(E_k=1)$ — depends **only** on $k$ (the defining PBM assumption), monotonically decreasing in $k$
- $\gamma_d = P(R_d=1\mid q)$ — depends **only** on $(d,q)$, never on $k$

So the per-(document, rank) click probability is simply the product $\theta_k \gamma_d$. The latent variables $E_k$ are unobserved; only $C_{d,k}$ is observed, which is what forces an EM-style inference.

### Likelihood and EM estimation

Given a click log of sessions $s$, each presenting document $d_s$ at rank $k_s$ with observed click $c_s$, the data log-likelihood is:

$$\mathcal{L}(\theta, \gamma) = \sum_{s} \Big[ c_s \log(\theta_{k_s}\gamma_{d_s}) + (1-c_s)\log(1 - \theta_{k_s}\gamma_{d_s}) \Big]$$

Because $E_k$ is latent, parameters are fit by **Expectation-Maximization**:

- **E-step** — for a *non-click* ($c_s = 0$) we infer the posterior that the rank was nonetheless examined (the click failed because the doc was irrelevant):
$$P(E_{k_s}=1 \mid c_s=0) = \frac{\theta_{k_s}(1-\gamma_{d_s})}{1-\theta_{k_s}\gamma_{d_s}}, \qquad P(R_{d_s}=1\mid c_s=0) = \frac{(1-\theta_{k_s})\gamma_{d_s}}{1-\theta_{k_s}\gamma_{d_s}}$$
(for a click $c_s=1$ both $E_{k_s}=1$ and $R_{d_s}=1$ are certain).
- **M-step** — re-estimate each parameter as the average of its inferred posterior over the relevant sessions:
$$\theta_k \leftarrow \frac{\sum_{s:\,k_s=k}\big[c_s + (1-c_s)P(E_{k}=1\mid c_s=0)\big]}{|\{s : k_s = k\}|}, \qquad \gamma_d \leftarrow \frac{\sum_{s:\,d_s=d}\big[c_s + (1-c_s)P(R_{d}=1\mid c_s=0)\big]}{|\{s : d_s = d\}|}$$

Iterating E and M to convergence yields the propensities $\theta_k$ used downstream.

## Key Properties / Variants

- **Two latent factors, one product** — the entire model is $P(C_{d,k}) = \theta_k\gamma_d$; everything else (EM, IPW) is bookkeeping on top of this factorization.
- **Propensities for counterfactual LTR** — the fitted $\theta_k$ are exactly the inverse weights used by [[Inverse Propensity Weighting]]: a click at rank $k$ counts as $1/\theta_k$ units of relevance evidence, debiasing [[Counterfactual Learning to Rank]] objectives.
- **Estimating $\theta_k$ without full EM** — propensities can also be recovered by **result randomization** (swap a document across ranks and watch how its click rate scales) or **intervention harvesting** from naturally occurring rank changes, avoiding the joint EM fit.
- **Identifiability caveat** — if documents rarely change rank, the data cannot separate "clicked because examined" from "clicked because relevant"; multiple $(\theta,\gamma)$ explain the log equally well. Randomization breaks this degeneracy.
- **Position-only assumption is the weakness** — PBM ignores that earlier results affect later examination. When users scan-and-stop, the [[cascade model]] / [[Cascading Position Bias]] is the correct alternative.
- **Does not model trust or outlier effects** — top ranks attracting *extra* clicks ([[Trust Bias]]) and visually distinctive items grabbing attention ([[Outlier Bias]]) both violate PBM's clean factorization and require extended models.

```pseudo
Algorithm: PBM Parameter Estimation via EM
──────────────────────────────────────────────
Input: click log {(d_s, k_s, c_s)} over sessions s
Initialize θ_k, γ_d ∈ (0,1) for all ranks k, docs d

Repeat until convergence:
  # E-step: posteriors over latent E, R for non-clicks
  For each session s:
    if c_s == 1:
      P(E=1) ← 1 ;  P(R=1) ← 1
    else:                                  # c_s == 0
      denom    ← 1 - θ_{k_s} * γ_{d_s}
      P(E=1)   ← θ_{k_s} * (1 - γ_{d_s}) / denom
      P(R=1)   ← (1 - θ_{k_s}) * γ_{d_s} / denom

  # M-step: re-estimate as posterior averages
  For each rank k:
    θ_k ← mean over {s : k_s = k} of [ c_s + (1-c_s)*P(E=1)_s ]
  For each doc d:
    γ_d ← mean over {s : d_s = d} of [ c_s + (1-c_s)*P(R=1)_s ]

Return propensities {θ_k}  →  feed as 1/θ_k weights to IPW
```

## Connections

- Instantiates: [[Examination Hypothesis]] (examination probability made a per-rank constant)
- Member of: [[Click Models]] (simplest member of the family)
- Quantifies: [[Position Bias]] via the propensities $\theta_k$
- Feeds: [[Inverse Propensity Weighting]] and [[Counterfactual Learning to Rank]] / [[Unbiased Learning to Rank]]
- Contrasted with: [[cascade model]] / [[Cascading Position Bias]] (examination depends on items above)
- Violated by: [[Trust Bias]], [[Outlier Bias]], [[Surrounding Item Bias]]
- Robust alternative when assumptions break: [[Doubly Robust Estimation]]

## Appears In

- [[Examination Hypothesis]]
- [[Inverse Propensity Weighting]]
- [[Outlier Bias]]
- [[Unbiased Learning to Rank]]
- [[IR-L11 - Unbiased Learning to Rank]]
