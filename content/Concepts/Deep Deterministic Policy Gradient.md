---
type: concept
aliases: [DDPG]
course: [RL]
tags: [policy-gradient, deep-rl, actor-critic, exam-topic]
status: complete
---

# Deep Deterministic Policy Gradient (DDPG)

## Definition

> [!definition] Deep Deterministic Policy Gradient
> **DDPG** is an **off-policy**, **model-free** [[Actor-Critic]] algorithm for **continuous action spaces**. It scales the [[Deterministic Policy Gradient]] to deep neural networks by combining it with the two stabilization tricks from [[Deep Q-Network (DQN)]]: an [[Experience Replay|replay buffer]] and (slowly-updated) [[Target Network|target networks]]. A deterministic actor $\mu_\theta(s) \to a$ is trained by following the gradient of a learned critic $Q_w(s,a)$, while the critic is trained by [[Q-Learning|Q-learning]].

## Intuition

In continuous control the greedy step of Q-learning, $\arg\max_a Q(s,a)$, is intractable — you cannot enumerate infinitely many actions. DDPG sidesteps this by maintaining a **deterministic actor** $\mu_\theta(s)$ that is trained to *output* the maximizing action directly. The critic supplies the gradient direction $\nabla_a Q$ telling the actor "which way to nudge the action to increase value," and the actor moves that way via the chain rule.

Because the actor is deterministic, the [[Deterministic Policy Gradient]] needs **no [[Importance Sampling|importance weights]]** — so DDPG can be fully off-policy, reusing a [[Experience Replay|replay buffer]] of past transitions. But a deterministic policy explores nothing on its own, so exploration is injected by adding noise to the actor's output during data collection. Naively bootstrapping a deep $Q_w$ against itself diverges (the [[Deadly Triad]]); DDPG borrows DQN's replay + target networks to stabilize it.

## Mathematical Formulation

**Actor (deterministic policy gradient).** The deterministic actor is updated to maximize the critic's value of its own action, via the chain rule through the action:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \mathcal{D}}\Big[ \nabla_a Q_w(s,a)\big|_{a=\mu_\theta(s)} \; \nabla_\theta \mu_\theta(s) \Big]$$

where:
- $\mu_\theta(s)$ — deterministic actor network mapping state to a single action
- $Q_w(s,a)$ — critic network estimating the action-value
- $\nabla_a Q_w(s,a)|_{a=\mu_\theta(s)}$ — direction in action space that increases value (the "pathwise"/chain-rule gradient)
- $\mathcal{D}$ — replay buffer; expectation over states drawn from it (off-policy)

**Critic (Q-learning / TD target with target networks).** The critic minimizes a one-step TD error against a bootstrapped target computed from **target** actor and critic networks $\mu_{\theta^-}, Q_{w^-}$:

$$L(w) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\Big[\big(Q_w(s,a) - y\big)^2\Big], \qquad y = r + \gamma\, Q_{w^-}\big(s', \mu_{\theta^-}(s')\big)$$

where:
- $y$ — TD target; the next action is supplied by the **target actor** $\mu_{\theta^-}(s')$ (no $\max$ needed — the actor *is* the argmax)
- $w^-, \theta^-$ — slowly tracking target-network parameters
- $\gamma$ — [[Discount Factor|discount factor]]

**Exploration noise.** Actions are collected by perturbing the deterministic actor:

$$a_t = \mu_\theta(s_t) + \mathcal{N}_t$$

where $\mathcal{N}_t$ is exploration noise (original DDPG used temporally-correlated Ornstein–Uhlenbeck noise; Gaussian noise also works in practice).

**Soft target updates.** Target networks are not copied periodically (as in DQN) but tracked smoothly with Polyak averaging:

$$\theta^- \leftarrow \tau\,\theta + (1-\tau)\,\theta^-, \qquad w^- \leftarrow \tau\,w + (1-\tau)\,w^-, \qquad \tau \ll 1$$

## Key Properties / Variants

- **Off-policy + continuous actions**: reuses a replay buffer; the deterministic actor replaces the intractable continuous-action $\max$ of Q-learning.
- **No importance weights**: inherited from the [[Deterministic Policy Gradient]] — the off-policy objective integrates over states only, not actions.
- **Stabilization from DQN**: [[Experience Replay]] decorrelates samples; [[Target Network|target networks]] (soft-updated with $\tau$) provide stable TD targets and break the [[Deadly Triad]].
- **Brittle / sample-greedy**: as a near-pure Q-maximizer, DDPG is sensitive to hyperparameters and Q-function overestimation. Discussed in [[RL-L11 - SAC, Decision Transformer & Diffuser]] as the motivating weakness that [[Soft Actor-Critic (SAC)|SAC]] fixes with a stochastic, entropy-regularized actor.
- **Related to SAC**: the reparameterized SAC policy update contains a "DDPG-like" term $\nabla_a Q \cdot \nabla_\phi f_\phi$ plus an entropy term — SAC is essentially a stochastic, max-entropy generalization of DDPG.
- **TD3 (Twin Delayed DDPG)**: addresses critic overestimation with two critics (uses the minimum), delayed actor updates, and target-policy smoothing.

```pseudo
Algorithm: DDPG (Deep Deterministic Policy Gradient)
────────────────────────────────────────────────────
Initialize critic Q_w(s,a) and actor μ_θ(s) with random weights w, θ
Initialize targets w⁻ ← w,  θ⁻ ← θ
Initialize empty replay buffer D

Loop for each episode:
  Initialize exploration noise process N
  Initialize state S
  Loop for each step:
    A ← μ_θ(S) + N            # deterministic action + exploration noise
    Execute A, observe R, S'
    Store (S, A, R, S') in D
    Sample minibatch {(s, a, r, s')} from D

    # Critic update (Q-learning with target nets)
    y ← r + γ Q_{w⁻}(s', μ_{θ⁻}(s'))
    w ← w − β ∇_w (1/N) Σ (Q_w(s,a) − y)²

    # Actor update (deterministic policy gradient)
    θ ← θ + α (1/N) Σ ∇_a Q_w(s,a)|_{a=μ_θ(s)} ∇_θ μ_θ(s)

    # Soft target updates
    w⁻ ← τ w + (1−τ) w⁻
    θ⁻ ← τ θ + (1−τ) θ⁻
    S ← S'
  until S is terminal
```

## Connections

- Scales: [[Deterministic Policy Gradient]] to deep networks
- Stabilization from: [[Deep Q-Network (DQN)]] ([[Experience Replay]] + [[Target Network]])
- Critic trained by: [[Q-Learning]] (TD target, no $\max$ needed)
- Architecture: [[Actor-Critic]] / [[Actor-Critic Methods]]
- Avoids: [[Importance Sampling]] (deterministic off-policy gradient); battles the [[Deadly Triad]]
- Generalized by: [[Soft Actor-Critic (SAC)]] (stochastic, [[Maximum Entropy RL|max-entropy]]) and [[TD3]] (twin critics)
- Contrast: stochastic [[Policy Gradient Methods]] and the [[Gaussian Policy]] used by [[REINFORCE]] / [[Actor-Critic]]

## Appears In

- [[Deterministic Policy Gradient]]
- [[Gaussian Policy]]
- [[RL-L11 - SAC, Decision Transformer & Diffuser]]
