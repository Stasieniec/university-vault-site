---
type: concept
aliases: [RWR]
course: [RL]
tags: [policy-gradient, algorithm, offline-rl, exam-topic]
status: complete
---

# Reward-Weighted Regression

## Definition

> [!definition] Reward-Weighted Regression (RWR)
> **Reward-Weighted Regression** is a policy-search method that turns policy improvement into a **weighted supervised-regression** problem: fit a new policy to the observed actions, where each $(s,a)$ sample is weighted by a monotonic, non-negative transformation of its return. High-return behavior is imitated strongly; low-return behavior is down-weighted. There are **no policy gradients** and **no value bootstrapping** — just maximum-likelihood fitting against reward-weighted targets, solved (originally) in closed form for linear/Gaussian policies.

## Intuition

> [!intuition] Imitate the good bits, weighted by how good they were
> Plain behavioral cloning imitates **all** logged actions equally, so it can never beat a mediocre dataset. RWR instead imitates actions **in proportion to how much reward they earned**. Conceptually it is "weighted [[REINFORCE]] without the gradient": rather than nudging $\log\pi$ up by a reward-scaled step, RWR directly *re-fits* the whole policy to a dataset where each action's influence is its reward weight $w = f(R)$.
>
> The trick that makes this an *EM* procedure rather than a hack: treating reward as a fictitious "success" signal lets us cast policy optimization as inference. Maximizing expected reward becomes maximizing a likelihood lower bound, and the maximization step is an ordinary weighted regression — exactly the kind of stable, well-understood objective that supervised learning excels at.

## Mathematical Formulation

RWR optimizes the expected-reward objective by an **Expectation–Maximization** lower bound. Treating an exponentiated reward as an unnormalized "improper" probability of a binary success event, one maximizes a weighted log-likelihood of the policy.

> [!formula] Reward-Weighted Regression Update
> Collect samples $\{(s_i, a_i, R_i)\}_{i=1}^N$ under the current policy $\pi_{\theta_{\text{old}}}$, then set
> $$\theta_{\text{new}} = \arg\max_\theta \; \sum_{i=1}^{N} w_i \, \log \pi_\theta(a_i \mid s_i),
> \qquad w_i = f(R_i)$$
> with the canonical exponential weighting
> $$w_i = \exp\!\Big(\tfrac{1}{\beta}\, R_i\Big) .$$
>
> where:
> - $\pi_\theta(a\mid s)$ — parametric policy being fit (the **actor**)
> - $R_i$ — return (or, in operational-space control, the per-step reward / advantage) of sample $i$
> - $w_i = f(R_i)$ — non-negative, monotonically increasing **reward weight**; the data sample's importance in the regression
> - $\beta > 0$ — temperature controlling greediness: $\beta \to 0$ concentrates all weight on the best samples (near-greedy), large $\beta$ flattens weights toward uniform behavioral cloning
> - $f$ — any monotone non-negative transform (exponential, or a shifted/normalized affine map of returns)

For a **Gaussian policy** $\pi_\theta(a\mid s) = \mathcal{N}(a \mid \phi(s)^\top\! \theta,\, \Sigma)$ the weighted log-likelihood maximization has a **closed-form weighted least-squares** solution (the original operational-space-control setting of Peters & Schaal, 2007):

> [!formula] Closed-form Gaussian / linear RWR
> $$\theta_{\text{new}} = \big(\Phi^\top W \Phi\big)^{-1}\, \Phi^\top W\, A$$
>
> where:
> - $\Phi$ — design matrix of features $\phi(s_i)$ stacked over samples
> - $A$ — matrix of observed actions $a_i$ stacked over samples (the regression targets)
> - $W = \mathrm{diag}(w_1, \dots, w_N)$ — diagonal matrix of reward weights
> - the result is **weighted [[Ordinary Least Squares|OLS]]**: the policy that best regresses observed actions onto features, with each sample weighted by its reward

Iterating *(sample → reweight → weighted regression)* monotonically improves a lower bound on expected reward, analogous to how EM iterates over a fixed objective.

## Key Properties / Variants

- **No gradients, no bootstrapping.** Unlike [[REINFORCE]] or [[Actor-Critic]], the M-step is a plain weighted regression, so it is stable and avoids step-size tuning for the policy update (the closed-form linear case has *no* learning rate at all).
- **EM / inference-as-control view.** RWR is policy search cast as probabilistic inference: reward is treated as evidence for a "success" variable, and the EM maximization step is the weighted likelihood fit. This is the same lineage as later methods that re-derive maximum-entropy objectives.
- **Linear-policy limitation.** The clean closed-form holds for linear/Gaussian policies. For deep policies the M-step becomes a few SGD steps of weighted log-likelihood instead of a single solve, which loses the closed-form guarantee.
- **Temperature $\beta$ trades greed vs. coverage.** Small $\beta$ behaves near-greedily (imitate only the very best samples), large $\beta$ degenerates toward uniform behavioral cloning. This is the same exponentiated-advantage knob seen in [[Maximum Entropy RL|max-entropy]] / KL-regularized policy updates.
- **Precursor to return-conditioned methods.** RWR (and the related PoWER algorithm) is one of the early "imitate the good trajectories" ideas that the [[Decision Transformer]] and [[Upside-Down RL]] later revisit with deep sequence models — but RWR *weights* by return rather than *conditioning* on a desired return.
- **Offline-friendly.** Because it only needs logged $(s,a,R)$ tuples and a supervised fit, RWR is naturally applicable to [[Offline Reinforcement Learning]].

```pseudo
Algorithm: Reward-Weighted Regression (RWR)
─────────────────────────────────────────────
Initialize policy parameters θ
Choose weighting f (e.g., w = exp(R / β))

Loop until converged:
  # E-step: gather experience under current policy
  Sample rollouts {(s_i, a_i, R_i)} from π_θ
  Compute reward weights w_i ← f(R_i)

  # M-step: weighted maximum-likelihood fit
  θ ← argmax_θ  Σ_i w_i · log π_θ(a_i | s_i)
      # Gaussian/linear policy ⇒ closed-form weighted least squares:
      #   θ ← (Φᵀ W Φ)⁻¹ Φᵀ W A,   W = diag(w_i)
      # Deep policy ⇒ a few SGD steps on the weighted NLL
return θ
```

## Connections

- Weighted, gradient-free cousin of: [[REINFORCE]], [[Policy Gradient Methods]]
- Solves an [[Ordinary Least Squares|OLS]]-style weighted regression in the linear/[[Gaussian Policy|Gaussian]] case
- Inference / EM and temperature view shared with: [[Maximum Entropy RL]]
- Same "imitate the good bits" family as: [[Decision Transformer]], [[Upside-Down RL]]
- Applicable to: [[Offline Reinforcement Learning]]
- Actor side of: [[Actor-Critic]] when an advantage replaces the raw return weight

## Appears In

- [[RL-L11 - SAC, Decision Transformer & Diffuser]]
