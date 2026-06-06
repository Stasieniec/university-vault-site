---
type: concept
aliases: [Compatible Features, Compatibility Conditions]
course: [RL]
tags: [policy-gradient, actor-critic, exam-topic]
status: complete
---

# Compatible Function Approximation

## Definition

> [!definition] Compatible Function Approximation
> In an [[Actor-Critic]] method, the critic $\hat{q}_w(s,a)$ is an **approximation** of the true $q^\pi(s,a)$, so naively plugging it into the [[Policy Gradient Theorem]] introduces **bias**. **Compatible function approximation** specifies the conditions under which a *learned* critic can replace the true action-value function with **no bias** in the policy gradient. The critic must (1) be **linear** in the policy's score features $\nabla_\theta \log \pi_\theta(a\mid s)$ and (2) be fit to **minimise mean-squared error** against $q^\pi$. When both hold, $\nabla_\theta J$ computed with $\hat{q}_w$ exactly equals the one computed with $q^\pi$.

## Intuition

> [!intuition] Why bias can vanish
> The policy gradient only ever "sees" the critic through the inner product $\nabla_\theta \log \pi_\theta(a\mid s)\, \hat{q}_w(s,a)$. The critic does **not** need to be globally correct — it only needs to be correct in the subspace spanned by the score functions $\nabla_\theta \log \pi_\theta$.
>
> If the critic is *linear in exactly those score features*, then any approximation error is, by construction, **orthogonal** to the score functions. Orthogonal error contributes nothing to the projected inner product, so it cancels in expectation. This is the same orthogonality logic that makes least-squares residuals perpendicular to the regression features — here the "features" are the policy's own score functions.

## Mathematical Formulation

A critic $\hat{q}_w(s,a)$ is **compatible** with policy $\pi_\theta$ if it satisfies two conditions.

**Condition 1 — Compatible features (the gradient must match):**
$$\nabla_w \hat{q}_w(s,a) = \nabla_\theta \log \pi_\theta(a\mid s)$$

This forces a **linear** critic in the score features:
$$\hat{q}_w(s,a) = w^\top \nabla_\theta \log \pi_\theta(a\mid s)$$

**Condition 2 — Critic minimises mean-squared error** against the true value:
$$\varepsilon(w) = \mathbb{E}_{s\sim\mu^\pi,\, a\sim\pi_\theta}\!\left[\big(q^\pi(s,a) - \hat{q}_w(s,a)\big)^2\right], \qquad w^\star = \arg\min_w \varepsilon(w)$$

**Result — unbiased policy gradient.** If both hold, then
$$\nabla_\theta J(\theta) = \mathbb{E}_{s\sim\mu^\pi,\, a\sim\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a\mid s)\, \hat{q}_{w^\star}(s,a)\right] = \mathbb{E}\!\left[\nabla_\theta \log \pi_\theta(a\mid s)\, q^\pi(s,a)\right]$$

where:
- $\hat{q}_w(s,a)$ — the parametrised critic, parameters $w$
- $\nabla_\theta \log \pi_\theta(a\mid s)$ — the **score function** of the policy (the Log-derivative trick term that appears in every policy gradient)
- $\mu^\pi$ — the On-Policy Distribution of states under $\pi_\theta$
- $q^\pi(s,a)$ — the true action-value function being approximated
- $w^\star$ — critic weights at the MSE minimum

**Why it works (proof sketch).** At the minimiser $w^\star$, the gradient of the MSE is zero:
$$\nabla_w \varepsilon(w^\star) = \mathbb{E}\!\left[\big(q^\pi - \hat{q}_{w^\star}\big)\, \nabla_w \hat{q}_{w^\star}(s,a)\right] = 0$$
Substituting Condition 1, $\nabla_w \hat{q}_{w^\star} = \nabla_\theta \log \pi_\theta$, gives
$$\mathbb{E}\!\left[\big(q^\pi - \hat{q}_{w^\star}\big)\, \nabla_\theta \log \pi_\theta(a\mid s)\right] = 0$$
i.e. the approximation error is **orthogonal** to the score functions. Adding this zero to the gradient lets us swap $q^\pi \to \hat{q}_{w^\star}$ with no change.

## Key Properties / Variants

- **Two conditions, both required**: linear-in-score-features critic **and** MSE-optimal weights. Drop either and the substitution is biased.
- **Subspace, not global, accuracy**: the critic need not approximate $q^\pi$ well everywhere — only its projection onto the score-function subspace matters.
- **Baselines are free under compatibility**: subtracting any state-dependent [[Baseline]] $b(s)$ (e.g. $\hat{v}(s)$) leaves the gradient unbiased because $\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a\mid s)\, b(s)] = 0$. Combining a compatible critic with a value baseline yields an unbiased [[Advantage Function]] estimate $\hat{A}(s,a) = \hat{q}_w(s,a) - b(s)$.
- **Limited expressiveness**: a critic linear in score features is weak. In practice (deep actor-critic, A2C/A3C/PPO) the compatibility conditions are **relaxed** — a nonlinear neural critic is used, trading exact unbiasedness for representational power. Compatible FA is mainly the **theoretical guarantee** that an unbiased actor-critic *can* exist.
- **Connection to Natural Policy Gradient**: with a compatible critic $\hat{q}_w = w^\top \nabla_\theta \log \pi_\theta$, the MSE-optimal weights $w^\star$ are exactly the [[Natural Policy Gradient]] direction, $w^\star = F^{-1}\nabla_\theta J$, where $F$ is the [[Fisher Information Matrix]]. So the compatible critic's parameters *are* the natural gradient.

```pseudo
Algorithm: Compatible Actor-Critic (one update)
────────────────────────────────────────────────
Given policy π_θ, compatible critic q̂_w(s,a) = wᵀ ∇_θ log π_θ(a|s)

Loop:
  Sample s ~ μ^π, a ~ π_θ(·|s); observe target for q^π(s,a)
  # Critic step: drive MSE to its minimum
  feat ← ∇_θ log π_θ(a|s)          # compatible features
  err  ← q^π_target(s,a) − wᵀ feat
  w    ← w + β · err · feat         # ⇒ at convergence, error ⟂ feat
  # Actor step: unbiased because critic is compatible + MSE-optimal
  θ    ← θ + α · feat · (wᵀ feat)   # = ∇_θ log π_θ · q̂_w
```

## Connections

- Makes unbiased: [[Policy Gradient Theorem]], [[Actor-Critic]]
- Critic feature is the: [[Log-derivative trick]] score function $\nabla_\theta \log \pi_\theta$
- Equivalent weights to: [[Natural Policy Gradient]] via [[Fisher Information Matrix]]
- Pairs with: [[Baseline]], [[Advantage Function]]
- State weighting: [[On-Policy Distribution]]
- Relaxed by deep methods: [[A2C]], [[A3C]], [[Proximal Policy Optimization]]
- Foundational for: [[REINFORCE]] (the special case with no critic)

## Appears In

- [[RL-L10 - Advanced Policy Search]]
