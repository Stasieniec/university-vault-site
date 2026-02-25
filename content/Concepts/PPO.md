---
type: concept
aliases: [Proximal Policy Optimization, PPO]
course: [RL]
tags: [policy-gradient, deep-rl, exam-topic]
status: complete
---

# Proximal Policy Optimization (PPO)

> [!definition] PPO
> A [[Policy Gradient Methods|policy gradient]] algorithm that constrains the policy update to stay close to the current policy, preventing destructively large updates. It uses a **clipped surrogate objective** as a simpler alternative to trust-region methods (TRPO).

## Intuition

Standard [[Policy Gradient Methods]] can take steps that are too large, causing the policy to collapse or oscillate. PPO addresses this by clipping the probability ratio between old and new policies, ensuring conservative updates without needing complex second-order optimization.

## Clipped Surrogate Objective

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ — probability ratio between new and old policy
- $\hat{A}_t$ — estimated advantage at time $t$
- $\epsilon$ — clipping hyperparameter (typically 0.1–0.2)
- The $\min$ operation takes the more pessimistic (conservative) bound

> [!intuition] Why Clipping Works
> If the ratio $r_t$ moves too far from 1 (policy changed a lot), the clip cuts off the objective's gradient, stopping the update. This means the policy can improve but can't change drastically in a single step.

## Algorithm Sketch

```pseudo
Algorithm: PPO (Clip version)
──────────────────────────────
For each iteration:
  1. Collect T timesteps of data using current policy π_θ_old
  2. Compute advantages Â_t (e.g., GAE)
  3. For K epochs over the collected data:
     - Compute r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
     - Compute L^CLIP(θ)
     - Update θ via gradient ascent on L^CLIP
  4. θ_old ← θ
```

## Key Properties

- **Simple to implement** compared to TRPO (no conjugate gradient, no KL constraint)
- **Empirically strong**: works well across many domains (Atari, MuJoCo, robotics)
- **Sample efficient**: reuses data across K epochs per collection phase
- **Monotonic improvement guarantee** (approximate): clipping prevents catastrophic updates

## Connections

- Extension of [[REINFORCE]] and [[Policy Gradient Theorem]]
- Uses [[Actor-Critic]] framework (policy + value function)
- Alternative to TRPO (Trust Region Policy Optimization)
- Often combined with [[Generalized Advantage Estimation]] (GAE)

## Appears In

- RL course Week 5–6 (policy gradient methods)
