---
type: concept
aliases: [Twin Delayed DDPG, Twin Delayed Deep Deterministic Policy Gradient]
course: [RL]
tags: [policy-gradient, deep-rl, actor-critic, exam-topic]
status: complete
---

# TD3

## Definition

> [!definition] TD3 (Twin Delayed DDPG)
> **TD3** is an **off-policy, actor-critic** algorithm for **continuous control** that fixes the systematic **overestimation bias** and brittle training of [[DDPG]]. It keeps DDPG's deterministic actor $\mu_\phi(s)$ and [[Deterministic Policy Gradient|DPG]] update, but adds three tricks: (1) **clipped double Q-learning** — two critics, take the **minimum** as the target; (2) **delayed policy updates** — update the actor (and targets) less often than the critics; (3) **target policy smoothing** — add clipped noise to the target action so the critic cannot exploit sharp peaks.

## Intuition

> [!intuition] Why the min of two critics?
> In value-based RL with [[Function Approximation]], the **max** operator (and the implicit max in DDPG's greedy actor) combined with noisy Q-estimates produces a **positive bias**: $\mathbb{E}[\max_a \hat{Q}] \ge \max_a \mathbb{E}[\hat{Q}]$. The actor then chases these inflated values, and the error compounds through [[Bootstrapping]]. TD3's answer is to train **two independent critics** and form the bootstrap target from $\min(Q_1, Q_2)$. The min is a pessimistic estimate, so any upward error in one critic is suppressed — trading a small *under*-estimation (harmless) for removing the dangerous *over*-estimation.
>
> **Delayed updates**: a moving actor chasing a still-noisy critic destabilises learning. Updating the actor only every $d$ critic steps lets the value estimate settle first (lower variance per actor update).
>
> **Target smoothing** is a regulariser: similar actions should have similar values, so we fit the critic to a small neighbourhood around the target action rather than a single point, preventing overfitting to narrow Q-function spikes.

## Mathematical Formulation

**Target action with clipped smoothing noise:**

$$\tilde{a} = \mu_{\phi'}(s') + \epsilon, \qquad \epsilon \sim \text{clip}\big(\mathcal{N}(0,\sigma^2),\, -c,\, +c\big)$$

**Clipped double-Q target (shared by both critics):**

$$y = r + \gamma \, \min_{i=1,2} Q_{\theta'_i}\big(s', \tilde{a}\big)$$

**Critic loss (each critic $i$, regressed to the same $y$):**

$$L(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \Big[\big(Q_{\theta_i}(s,a) - y\big)^2\Big]$$

**Delayed actor update (DPG, using only $Q_{\theta_1}$), applied every $d$ steps:**

$$\nabla_\phi J(\phi) = \mathbb{E}_{s \sim \mathcal{D}}\Big[\nabla_a Q_{\theta_1}(s,a)\big|_{a=\mu_\phi(s)} \, \nabla_\phi \mu_\phi(s)\Big]$$

**Polyak (soft) target updates, also every $d$ steps:**

$$\theta'_i \leftarrow \tau\theta_i + (1-\tau)\theta'_i, \qquad \phi' \leftarrow \tau\phi + (1-\tau)\phi'$$

where:
- $\mu_\phi(s)$ — deterministic actor; $\mu_{\phi'}$ — its target network
- $Q_{\theta_1}, Q_{\theta_2}$ — twin critics; $Q_{\theta'_1}, Q_{\theta'_2}$ — their target networks
- $\min_{i=1,2} Q_{\theta'_i}$ — clipped double-Q: the pessimistic bootstrap that curbs overestimation
- $\epsilon \sim \text{clip}(\mathcal{N}(0,\sigma^2), -c, c)$ — target policy smoothing noise, clipped to $[-c,c]$
- $\sigma$ — smoothing noise std; $c$ — noise clip; $d$ — policy/target update delay (typically $d=2$)
- $\gamma$ — discount factor; $\tau \ll 1$ — Polyak averaging rate; $\mathcal{D}$ — replay buffer
- $y$ — TD target; note it is shared by both critics even though only $Q_{\theta_1}$ drives the actor

## Key Properties / Variants

- **Three improvements over [[DDPG]]**: clipped double Q (bias), delayed actor/target updates (variance/stability), target policy smoothing (regularisation). DDPG = TD3 minus these three.
- **Off-policy**: trains from a [[Experience Replay|replay buffer]] $\mathcal{D}$; behaviour = $\mu_\phi(s)$ + exploration noise $\mathcal{N}(0,\sigma_{\text{explore}})$. Inherits DPG's lack of [[Importance Sampling]] weights.
- **Two noises, different roles**: *exploration* noise is added to actions taken in the environment; *smoothing* noise is added only to the **target** action when computing $y$.
- **Both critics regress to the same target $y$** (the min), but only **one** critic ($Q_{\theta_1}$) is used in the actor gradient — this avoids the actor exploiting the pessimistic min.
- **Deterministic, not stochastic**: unlike [[Soft Actor-Critic (SAC)]], TD3 keeps a deterministic policy with **no** entropy term; exploration is purely injected noise.

```pseudo
Algorithm: TD3 (Twin Delayed DDPG)
──────────────────────────────────────────────────────────
Init critics Q_θ1, Q_θ2 and actor μ_φ with random params
Init targets θ'1 ← θ1, θ'2 ← θ2, φ' ← φ
Init replay buffer D
for t = 1 .. T:
  Select action with exploration noise:
    a = μ_φ(s) + ε,  ε ~ N(0, σ_explore)
  Execute a, observe r, s'; store (s,a,r,s') in D
  Sample mini-batch of N transitions (s,a,r,s') from D

  # ---- Target with smoothing + clipped double-Q ----
  ã ← μ_φ'(s') + clip(N(0,σ²), -c, +c)
  y ← r + γ · min( Q_θ'1(s',ã), Q_θ'2(s',ã) )

  # ---- Update BOTH critics toward the same y ----
  θ_i ← argmin_θ_i  (1/N) Σ (Q_θ_i(s,a) - y)²   for i = 1,2

  # ---- Delayed actor + target updates (every d steps) ----
  if t mod d == 0:
    update φ by deterministic policy gradient:
      ∇_φ J = (1/N) Σ ∇_a Q_θ1(s,a)|_{a=μ_φ(s)} ∇_φ μ_φ(s)
    Polyak update targets:
      θ'_i ← τ θ_i + (1-τ) θ'_i   for i = 1,2
      φ'   ← τ φ   + (1-τ) φ'
```

## Connections

- Fixes overestimation in: [[DDPG]] (TD3 is the direct successor)
- Built on: [[Deterministic Policy Gradient]], [[Actor-Critic]], [[Policy Gradient Methods]]
- Reuses machinery from: [[Q-Learning]] (clipped double-Q is a double-estimator fix), [[Target Network]], [[Experience Replay]]
- Contrast with: [[Soft Actor-Critic (SAC)]] (stochastic, max-entropy; SAC also uses the min of two critics)
- Avoids variance of: [[Importance Sampling]] (off-policy via DPG, no ratios)
- Continuous-control sibling under: [[Deep Reinforcement Learning]]

## Appears In

- [[Deterministic Policy Gradient]]
- [[RL-L10 - Advanced Policy Search]]
- [[RL-L11 - SAC, Decision Transformer & Diffuser]]
