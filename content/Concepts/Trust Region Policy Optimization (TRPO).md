---
type: concept
aliases: [TRPO, Trust Region Policy Optimization]
course: [RL]
tags: [policy-gradient, deep-rl, optimization, exam-topic]
status: complete
---

# Trust Region Policy Optimization (TRPO)

## Definition

> [!definition] Trust Region Policy Optimization
> **TRPO** is an on-policy policy gradient algorithm that improves the policy by **maximizing a surrogate objective** (the expected advantage under the new policy, weighted by an importance ratio) **subject to a hard constraint** that the new policy stays close to the old one in **average KL divergence**. The KL constraint defines a "trust region" inside which the surrogate is a reliable approximation of the true performance, guaranteeing **monotonic (non-decreasing) policy improvement** in the idealized case.

## Intuition

Vanilla [[Policy Gradient Methods|policy gradient]] takes a fixed-size step in **parameter space** ($\theta \leftarrow \theta + \alpha \nabla_\theta J$). The problem: a small change in $\theta$ can cause a huge change in the actual policy distribution, collapsing performance. Because policy gradient is on-policy, one bad update poisons all future data — there is no recovery from off-policy replay.

TRPO fixes this by measuring step size in **policy space** instead of parameter space. It asks: *"How far can I move the policy and still trust my data-derived estimate of improvement?"* The answer is bounded by the KL divergence between old and new policy. Stay inside the trust region $\overline{D}_{\text{KL}} \le \delta$, and the surrogate objective faithfully predicts the true return; the update is then guaranteed not to make things worse.

> [!intuition] Why a Constraint Instead of a Penalty
> Theory (the MM / minorize-maximize bound) actually suggests a KL **penalty** with a coefficient tied to the max advantage. But that coefficient forces tiny, conservative steps. TRPO instead uses a **hard KL constraint** $\delta$ as a robust, tunable proxy that allows much larger practical steps while keeping the trust-region guarantee.

## Mathematical Formulation

**The surrogate objective.** TRPO maximizes the expected advantage of the new policy $\pi_\theta$, with state visitation taken from the **old** policy $\pi_{\theta_\text{old}}$ and actions reweighted by an importance ratio:

$$L_{\theta_\text{old}}(\theta) = \mathbb{E}_{s \sim \rho^{\pi_{\theta_\text{old}}},\, a \sim \pi_{\theta_\text{old}}} \left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_\text{old}}(a \mid s)} A^{\pi_{\theta_\text{old}}}(s, a) \right]$$

**The constrained optimization problem** solved at each iteration:

$$\max_{\theta}\; L_{\theta_\text{old}}(\theta) \qquad \text{subject to} \qquad \overline{D}_{\text{KL}}\!\left(\pi_{\theta_\text{old}} \,\|\, \pi_\theta\right) \le \delta$$

where:
- $\dfrac{\pi_\theta(a\mid s)}{\pi_{\theta_\text{old}}(a\mid s)}$ — **importance ratio** correcting for the fact that data was collected under $\pi_{\theta_\text{old}}$ but we score $\pi_\theta$
- $A^{\pi_{\theta_\text{old}}}(s,a)$ — [[Advantage Function|advantage]] under the old policy (estimated in practice by [[Generalized Advantage Estimation|GAE]])
- $\rho^{\pi_{\theta_\text{old}}}$ — discounted state-visitation distribution under the old policy
- $\overline{D}_{\text{KL}}(\pi_{\theta_\text{old}} \| \pi_\theta) = \mathbb{E}_{s \sim \rho^{\pi_{\theta_\text{old}}}}\!\big[ D_{\text{KL}}(\pi_{\theta_\text{old}}(\cdot\mid s) \,\|\, \pi_\theta(\cdot\mid s)) \big]$ — **average** KL over visited states (the max-KL version is intractable)
- $\delta$ — trust-region radius (typical: $\delta \approx 10^{-2}$)

**Solving it (the natural-gradient connection).** Linearize the objective and quadratically approximate the constraint around $\theta_\text{old}$:

$$\max_{\Delta\theta}\; g^\top \Delta\theta \quad \text{s.t.} \quad \tfrac{1}{2}\,\Delta\theta^\top F\, \Delta\theta \le \delta$$

where $g = \nabla_\theta L_{\theta_\text{old}}(\theta)|_{\theta_\text{old}}$ is the policy gradient and $F$ is the [[Fisher Information Matrix]] (the Hessian of the KL constraint). The closed-form solution is the [[Natural Policy Gradient|natural gradient]] step scaled to fill the trust region:

$$\theta_\text{new} = \theta_\text{old} + \sqrt{\frac{2\delta}{g^\top F^{-1} g}}\; F^{-1} g$$

where:
- $F^{-1}g$ — the **natural gradient** direction (search direction $x$ obtained by solving $Fx = g$ with conjugate gradient, never forming $F$ explicitly)
- $\sqrt{2\delta / (g^\top F^{-1} g)}$ — **maximal step size** along $F^{-1}g$ that keeps the quadratic KL at exactly $\delta$

> [!formula] Monotonic Improvement Bound
> TRPO is derived from a guarantee on the true return $\eta$:
> $$\eta(\pi_\theta) \ge L_{\theta_\text{old}}(\theta) - C \cdot \overline{D}_{\text{KL}}^{\max}(\pi_{\theta_\text{old}} \| \pi_\theta), \qquad C = \frac{4\varepsilon\gamma}{(1-\gamma)^2}$$
> with $\varepsilon = \max_{s,a}|A^{\pi_{\theta_\text{old}}}(s,a)|$. Maximizing the right-hand side (a **minorant** of the true return) cannot decrease $\eta$ — this is the monotonic-improvement property. TRPO replaces the penalty $C\,\overline{D}_{\text{KL}}$ with the trust-region constraint $\delta$ for larger, practical steps.

## Key Properties / Variants

- **On-policy.** Fresh trajectories are collected from $\pi_{\theta_\text{old}}$ each iteration; the importance ratio is only a local correction valid near $\theta_\text{old}$ (inside the trust region).
- **Second-order / natural-gradient method.** The Fisher matrix $F$ is the curvature of policy space; TRPO is essentially [[Natural Policy Gradient]] with an automatically chosen step size plus safeguards.
- **Hessian-free.** Conjugate gradient solves $Fx = g$ using only **Fisher-vector products** ($Fv$ via a double back-prop through the KL), avoiding the $O(d^3)$ cost of inverting $F$.
- **Backtracking line search** enforces the *true* (non-approximated) constraint and a real improvement in the surrogate — protecting against errors from the quadratic approximation.
- **Monotonic improvement guarantee** (in the exact setting), unlike vanilla policy gradient which can collapse.
- **Cost / drawbacks:** the CG + line search machinery is complex and per-update expensive; it does not naturally share parameters between policy and value networks, and is awkward with architectures involving heavy noise/dropout.
- **Successor — [[PPO|Proximal Policy Optimization]]:** replaces the hard KL constraint with a **clipped** surrogate ratio (or a soft KL penalty), recovering most of TRPO's stability with first-order SGD and far simpler code. PPO is now the default; TRPO is the theoretical anchor.

```pseudo
Algorithm: Trust Region Policy Optimization (TRPO)
──────────────────────────────────────────────────
Input: initial policy θ, value/critic params w, trust radius δ
Loop for each iteration:
  1. Collect trajectories by running π_θ_old (θ_old ← θ) in the environment
  2. Estimate advantages  Â_t   (e.g., GAE with the critic V_w)
  3. Compute policy gradient   g = ∇_θ L(θ) |_θ_old
       L(θ) = mean_t [ (π_θ(a_t|s_t) / π_θ_old(a_t|s_t)) · Â_t ]
  4. Solve  F x = g  by conjugate gradient    (x ≈ F⁻¹ g, natural-grad direction)
       using Fisher-vector products  F v = ∇_θ ( (∇_θ KL)·v )   (no explicit F)
  5. Compute max step:  β = sqrt( 2δ / (xᵀ F x) )
  6. Backtracking line search over j = 0,1,2,...:
       θ_try = θ_old + (α^j) · β · x          (α ∈ (0,1), e.g. 0.5)
       accept first θ_try with  KL(π_θ_old ‖ π_θ_try) ≤ δ
                               and  L(θ_try) > L(θ_old)
       θ ← θ_try   (if none accepted, keep θ_old)
  7. Fit critic: minimize  Σ_t ( V_w(s_t) − R̂_t )²
```

## Connections

- Builds directly on: [[Natural Policy Gradient]], [[Fisher Information Matrix]], [[Policy Gradient Theorem]]
- Uses: [[Advantage Function]], [[Generalized Advantage Estimation]], [[Importance Sampling]] (the policy ratio)
- Simplified successor: [[PPO]] (clipped surrogate, first-order)
- Family: [[Actor-Critic]], [[Policy Gradient Methods]], [[Deep Reinforcement Learning]]
- Contrast: vanilla policy gradient / [[REINFORCE]] (parameter-space step, no improvement guarantee)

## Appears In

- [[RL-L10 - Advanced Policy Search]]
- [[Advantage Function]]
- [[Generalized Advantage Estimation]]
- [[Natural Policy Gradient]]
