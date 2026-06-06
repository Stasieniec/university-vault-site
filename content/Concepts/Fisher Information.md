---
type: concept
aliases: [Fisher Information Matrix]
course: [RL]
tags: [policy-gradient, optimization, deep-rl, exam-topic]
status: complete
---

# Fisher Information

## Definition

> [!definition] Fisher Information (Matrix)
> The **Fisher Information Matrix** $F(\theta)$ measures how much information an observable sample carries about the parameter $\theta$ of a probability distribution $p_\theta$. In RL, $p_\theta = \pi_\theta(a \mid s)$ is the policy, and $F(\theta)$ quantifies the **sensitivity of the policy distribution to changes in $\theta$**. It is defined as the expected outer product of the score (the gradient of the log-likelihood), and equivalently as the local curvature (Hessian) of the [[Natural Policy Gradient|KL]] divergence around $\theta$. It is the [[Steepest Descent|Riemannian metric tensor]] that turns vanilla gradients into [[Natural Gradient|natural gradients]].

## Intuition

> [!intuition] A Ruler for Distribution Space
> The vanilla gradient $\nabla_\theta J$ asks "which way moves the *parameters* fastest?" — but that depends entirely on how you happened to parameterize the policy ($\sigma$ vs $\log\sigma$ vs $\sigma^2$ give wildly different steps). The Fisher matrix re-scales the geometry so the question becomes "which way moves the *policy distribution* fastest?", measured by KL divergence.
>
> Think of $F(\theta)$ as a curvature-aware ruler: along a direction where a tiny parameter nudge swings the action distribution a lot (high information), $F$ has a **large** eigenvalue, so the natural gradient takes a **small** step there; along a flat direction where parameters barely matter, $F$ is small and the step is large. This makes the resulting update **invariant** to reparameterization.

## Mathematical Formulation

The **score function** is the gradient of the log-likelihood, $\nabla_\theta \log \pi_\theta(a \mid s)$. The Fisher Information Matrix is the covariance of the score (its mean is zero):

$$F(\theta) = \mathbb{E}_{s \sim d^\pi,\; a \sim \pi_\theta(\cdot \mid s)}\!\left[ \nabla_\theta \log \pi_\theta(a \mid s)\, \nabla_\theta \log \pi_\theta(a \mid s)^{\top} \right]$$

where:
- $F(\theta)$ — Fisher Information Matrix, a $d \times d$ matrix ($d$ = number of policy parameters), symmetric and positive semi-definite
- $\nabla_\theta \log \pi_\theta(a \mid s)$ — the **score** vector (also the [[Log-derivative trick|log-derivative]] used in [[Policy Gradient Theorem|policy gradients]])
- $d^\pi$ — the [[On-Policy Distribution|state distribution]] induced by following $\pi_\theta$
- $\mathbb{E}[\cdot\,]$ — expectation under the policy's own samples (the outer product averages over actions drawn from $\pi_\theta$)

Under regularity conditions, the score has zero mean, $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a\mid s)] = 0$, so $F$ is exactly the **covariance of the score**. There is also an equivalent **negative-Hessian** form:

$$F(\theta) = -\,\mathbb{E}_{a \sim \pi_\theta}\!\left[ \nabla_\theta^2 \log \pi_\theta(a \mid s) \right]$$

**Connection to KL divergence** — $F$ is the second-order (Hessian) term of the KL divergence between nearby policies:

$$D_{\mathrm{KL}}\big(\pi_\theta \,\|\, \pi_{\theta + \Delta\theta}\big) \approx \tfrac{1}{2}\, \Delta\theta^{\top} F(\theta)\, \Delta\theta$$

where:
- $\Delta\theta$ — a small parameter displacement
- the first-order term vanishes because KL is minimized (zero) at $\Delta\theta = 0$, leaving $F$ as the local quadratic metric

This is precisely why $F$ appears in the [[Natural Policy Gradient]]: preconditioning by $F^{-1}$ yields the steepest-ascent direction in distribution (KL) space rather than raw parameter space:

$$\tilde{\nabla}_\theta J(\theta) = F(\theta)^{-1}\, \nabla_\theta J(\theta)$$

## Key Properties / Variants

- **Positive semi-definite & symmetric.** Eigenvalues $\geq 0$; large eigenvalues mark "informative" directions where the policy is sensitive to $\theta$.
- **Reparameterization invariance.** The natural gradient $F^{-1}\nabla_\theta J$ produces the *same policy update* under any smooth reparameterization, unlike the vanilla gradient.
- **Curvature = information.** $F$ is simultaneously (i) the score covariance, (ii) the negative expected Hessian of the log-likelihood, and (iii) the Hessian of KL divergence — three identities for the same object.
- **Cramér–Rao link.** In statistics, $F^{-1}$ lower-bounds the variance of any unbiased estimator of $\theta$; high information ⇒ tighter achievable estimates.
- **Computational cost.** Forming and inverting $F$ explicitly is $O(d^3)$ — prohibitive for neural policies. Practical schemes avoid this:

```pseudo
Algorithm: Fisher-Vector Product via Conjugate Gradient
─────────────────────────────────────────────────────────
Goal: compute natural gradient  x = F^{-1} g   without forming F

Input: vanilla gradient g = ∇_θ J(θ), damping λ
Define Fisher-vector product (no explicit F):
  FVP(v):
    # use the KL-Hessian identity: F v = ∇_θ ( (∇_θ D_KL)^T v )
    kl   ← mean KL( π_old || π_θ ) over sampled states
    grad ← ∇_θ kl
    return ∇_θ ( grad · v ) + λ·v          # add damping λI for stability

# Solve  F x = g  iteratively (≈20 CG steps), never inverting F
x ← ConjugateGradient(FVP, b = g, iters = 20)

return x        # x ≈ F^{-1} g  =  natural gradient direction
```

- **Approximations.** *Diagonal Fisher* $F \approx \mathrm{diag}(\mathbb{E}[g \odot g])$ is $O(d)$ but crude; *K-FAC* (Kronecker-factored) gives a structured, accurate approximation for neural nets; adding damping $F + \lambda I$ improves conditioning.
- **Empirical vs true Fisher.** The "true" Fisher samples actions $a \sim \pi_\theta$; the *empirical Fisher* uses the actions actually taken from data. They coincide only when the model fits the data well, and behave differently as optimizers.

## Connections

- Preconditioner for: [[Natural Policy Gradient]], [[Natural Gradient]]
- Built from: [[Log-derivative trick]] (the score function), [[Policy Gradient Theorem]]
- Geometry of: [[Steepest Descent]] in distribution space (vs parameter space)
- Underlies: [[Trust Region Policy Optimization (TRPO)]] (KL-constrained step), approximated away by [[PPO]]
- Contrast with: vanilla [[Gradient Ascent]] / [[Stochastic Gradient Descent]] (identity metric)
- Adaptive optimizers ([[Adam]], [[AdaGrad]]) loosely mimic diagonal curvature scaling

## Appears In

- [[Natural Policy Gradient]]
- [[RL-L10 - Advanced Policy Search]]
- [[Trust Region Policy Optimization (TRPO)]]
