---
type: concept
aliases: [A2C]
course: [RL]
tags: [policy-gradient, actor-critic, deep-rl, exam-topic]
status: complete
---

# Advantage Actor-Critic (A2C)

## Definition

> [!definition] Advantage Actor-Critic (A2C)
> **A2C** is an **on-policy** [[Actor-Critic]] algorithm that combines a parameterized policy (the **actor**, $\pi_\theta$) with a learned state-value function (the **critic**, $\hat{V}_w$). The actor is updated along the [[Policy Gradient Theorem|policy gradient]], weighting $\nabla_\theta \log \pi_\theta(a|s)$ by an **advantage estimate** $\hat{A}_t$ instead of the raw return; the critic provides the [[Baseline|value baseline]] used to form that advantage and is trained to predict returns. A2C is the **synchronous, single-thread** version of [[A3C]] (Asynchronous Advantage Actor-Critic): instead of asynchronous workers updating shared parameters, A2C collects rollouts from parallel environments, **synchronizes** them, and applies one batched gradient step.

## Intuition

[[REINFORCE]] scales the policy-gradient by the full Monte Carlo return $G_t$, which is unbiased but very high variance. A2C fixes this with two ideas working together:

- **Critic as baseline.** Subtracting the state value $\hat{V}(s_t)$ from the return centres the signal: actions are judged *relative to what was expected from that state*, not by absolute reward magnitude. This is the [[Advantage Function]] $A(s,a) = Q(s,a) - V(s)$. Subtracting a state-only baseline does **not** introduce bias (the expected score-function term is zero).
- **Bootstrapping for the target.** Rather than waiting for a full episode, the critic bootstraps with a [[Temporal Difference Learning|TD]] target $r_t + \gamma \hat{V}(s_{t+1})$, so A2C can learn online from short rollouts and continuing tasks.

The actor and critic form a feedback loop: the critic tells the actor "this action was $\hat{A}_t$ better than average here," the actor shifts probability mass toward positive-advantage actions, and the critic re-fits its value estimate to the new policy's returns.

## Mathematical Formulation

The advantage is estimated by the one-step TD error of the critic:

$$\hat{A}_t = \delta_t = r_t + \gamma \hat{V}_w(s_{t+1}) - \hat{V}_w(s_t)$$

**Actor (policy) update** — gradient ascent on expected return:

$$\theta \leftarrow \theta + \alpha_\theta \, \hat{A}_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

**Critic (value) update** — gradient descent on the squared TD error / value error:

$$w \leftarrow w - \alpha_w \, \nabla_w \tfrac{1}{2}\big(r_t + \gamma \hat{V}_w(s_{t+1}) - \hat{V}_w(s_t)\big)^2
\;=\; w + \alpha_w \, \delta_t \, \nabla_w \hat{V}_w(s_t)$$

In deep A2C the two are typically combined into a single loss (often with shared body) minimized by SGD/[[Adam]]:

$$\mathcal{L}(\theta, w) = \underbrace{-\,\hat{A}_t \log \pi_\theta(a_t\mid s_t)}_{\text{actor (policy) loss}} \;+\; c_v \underbrace{\big(R_t - \hat{V}_w(s_t)\big)^2}_{\text{critic (value) loss}} \;-\; c_e \underbrace{H\!\big(\pi_\theta(\cdot\mid s_t)\big)}_{\text{entropy bonus}}$$

where:
- $\pi_\theta(a\mid s)$ — actor: parameterized stochastic policy (softmax for discrete actions, Gaussian for continuous)
- $\hat{V}_w(s)$ — critic: parameterized estimate of the state-value function $V^\pi(s)$
- $\hat{A}_t = \delta_t$ — advantage estimate; the TD error treated as a **constant** when differentiating the actor loss (no gradient flows through it into $\theta$)
- $\gamma$ — discount factor
- $\alpha_\theta, \alpha_w$ — actor and critic step sizes (or a single learning rate for the joint loss)
- $R_t = \sum_{l=0}^{n-1}\gamma^l r_{t+l} + \gamma^n \hat{V}_w(s_{t+n})$ — $n$-step bootstrapped return target for the critic
- $H(\pi_\theta(\cdot\mid s)) = -\sum_a \pi_\theta(a\mid s)\log \pi_\theta(a\mid s)$ — policy [[Entropy]], added to encourage exploration and prevent premature collapse
- $c_v, c_e$ — value-loss and entropy-bonus coefficients

The single-step $\delta_t$ can be replaced by an $n$-step advantage $\sum_{l=0}^{n-1}\gamma^l r_{t+l} + \gamma^n \hat{V}(s_{t+n}) - \hat{V}(s_t)$ or by [[Generalized Advantage Estimation]] (GAE) to trade bias against variance via $\lambda$.

## Key Properties / Variants

- **On-policy.** Samples must come from the current $\pi_\theta$; rollouts are discarded after each update (no replay buffer). This makes A2C less sample-efficient than off-policy value methods but stable and simple.
- **Synchronous vs asynchronous.** A2C = synchronous [[A3C]]. It runs $N$ parallel environment copies, waits for all to produce a batch of transitions, then performs **one** averaged gradient update. This gives better GPU utilization and more reproducible gradients than A3C's lock-free asynchronous updates, usually matching or beating A3C's performance.
- **Bias–variance knob.** $n$-step returns or GAE-$\lambda$ interpolate between the low-variance/biased one-step TD advantage ($\lambda \to 0$) and the unbiased/high-variance Monte Carlo advantage ($\lambda \to 1$).
- **Entropy regularization.** The entropy bonus keeps the policy stochastic early on, improving exploration.
- **Shared network.** Actor and critic commonly share lower layers (e.g. a CNN/MLP trunk) with two heads: a policy head and a scalar value head.
- **Relation to others.** [[REINFORCE]] with a value baseline is the Monte Carlo special case; [[PPO]] adds a clipped surrogate objective and multiple epochs per batch on top of the same advantage-weighted gradient.

```pseudo
Algorithm: A2C (Synchronous Advantage Actor-Critic)
───────────────────────────────────────────────────
Initialize actor params θ and critic params w
Launch N parallel environment workers
Loop until converged:
  # 1. Collect a synchronous batch of rollouts
  For each worker i = 1..N (in parallel):
    Run policy π_θ for T steps, storing (s_t, a_t, r_t)
  Barrier: wait for all workers to finish T steps

  # 2. Compute bootstrapped targets and advantages
  For each worker, working backward from t = T-1 to 0:
    R_t ← r_t + γ R_{t+1}            # bootstrap R_T from V_w(s_T) if non-terminal
    Â_t ← R_t - V_w(s_t)             # (or n-step / GAE advantage)

  # 3. Single batched gradient update over all N·T samples
  θ ← θ + α_θ · mean[ Â_t ∇_θ log π_θ(a_t|s_t) + c_e ∇_θ H(π_θ(·|s_t)) ]
  w ← w - α_w · mean[ ∇_w (R_t - V_w(s_t))^2 ]
```

> [!warning] Advantage must be a constant in the actor gradient
> When forming the actor loss $-\hat{A}_t \log \pi_\theta(a_t|s_t)$, the advantage $\hat{A}_t$ must be **detached** (treated as a fixed scalar). If gradients are allowed to flow from $\hat{A}_t$ back into the critic's $V_w$ through the actor term, the update no longer follows the policy gradient and training destabilizes. The critic is optimized only through its own value loss.

> [!warning] On-policy data only
> A2C uses no replay buffer. Reusing stale transitions collected under an old policy biases the on-policy gradient. If you need experience reuse, move to importance-sampled / clipped objectives such as [[PPO]].

## Connections

- Specializes: [[Actor-Critic]] (uses the advantage as the critic signal)
- Synchronous version of: [[A3C]]
- Built on: [[Policy Gradient Theorem]], [[REINFORCE]]
- Uses: [[Advantage Function]], [[Baseline]] (the value critic), [[Temporal Difference Learning]] (bootstrapped target), [[Entropy]] (exploration bonus)
- Advantage estimation refined by: [[Generalized Advantage Estimation]]
- Extended by: [[PPO]] (clipped surrogate, multiple epochs), [[TRPO]]
- Optimized with: [[Adam]] / [[Stochastic Gradient Descent]]

## Appears In

- [[Advantage Function]]
- [[Baseline]]
- [[Generalized Advantage Estimation]]
- [[Policy Gradient Theorem]]
- [[REINFORCE]]
- [[RL-L09 - Policy Gradient Methods]]
- [[RL-L10 - Advanced Policy Search]]
