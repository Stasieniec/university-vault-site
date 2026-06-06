---
type: concept
aliases: [Asynchronous Advantage Actor-Critic]
course: [RL]
tags: [policy-gradient, actor-critic, deep-rl, exam-topic]
status: complete
---

# A3C

## Definition

> [!definition] A3C (Asynchronous Advantage Actor-Critic)
> **A3C** is an **on-policy**, **deep** [[Actor-Critic]] algorithm (Mnih et al., 2016) that runs **many actor-learners in parallel**, each on its own copy of the environment, asynchronously updating a **shared** set of global parameters. Each worker computes [[Policy Gradient]] updates using an [[Advantage Function|advantage]] estimate from short $n$-step rollouts, then pushes its gradients to the global network. The diversity of parallel, decorrelated experience **replaces the [[Experience Replay|replay buffer]]** as the mechanism for stabilizing deep RL.

## Intuition

The core problem in deep RL is that consecutive samples from a single agent are **highly correlated**, which destabilizes neural-network training. [[Deep Q-Network (DQN)]] solved this with a replay buffer that shuffles past transitions — but replay forces an **off-policy** method and is memory-hungry.

A3C's insight: instead of decorrelating *in time* with a buffer, decorrelate *in space* by running $K$ workers in parallel. At any instant the workers are in different states, exploring with different random seeds (and often different exploration rates), so the stream of gradients arriving at the global network is approximately independent. This:

- restores **on-policy** learning (no importance-sampling corrections needed),
- removes the replay memory entirely,
- and runs on **multi-core CPU** rather than requiring a GPU.

Each worker is an [[Actor-Critic]]: an **actor** $\pi(a\mid s;\theta)$ proposes actions and a **critic** $V(s;\theta_v)$ estimates state value, used as the [[Baseline]] to form the advantage that weights the policy-gradient step.

## Mathematical Formulation

Each worker collects an $n$-step rollout (up to $t_{max}$ steps or episode end), then bootstraps with the critic. The **$n$-step advantage** for a step at time $i$ in the rollout is

$$\hat{A}(s_i,a_i) = \left(\sum_{l=0}^{k-1} \gamma^{l}\, r_{i+l} \;+\; \gamma^{k}\, V(s_{i+k};\theta_v)\right) - V(s_i;\theta_v)$$

where:
- $k$ — number of steps remaining until the end of the rollout (capped at $t_{max}$); $k$ varies per step
- $\sum_{l=0}^{k-1}\gamma^l r_{i+l}$ — observed discounted reward over the $k$-step rollout
- $\gamma^{k} V(s_{i+k};\theta_v)$ — bootstrapped value of the last state ($0$ if terminal)
- $V(s_i;\theta_v)$ — critic baseline subtracted to reduce variance

The two heads are trained by **accumulating gradients** over the rollout. The **policy (actor)** ascends the advantage-weighted log-likelihood, with an entropy bonus for exploration:

$$d\theta \;\leftarrow\; d\theta \;+\; \nabla_{\theta} \log \pi(a_i\mid s_i;\theta)\,\hat{A}(s_i,a_i) \;+\; \beta\,\nabla_{\theta} H\big(\pi(\cdot\mid s_i;\theta)\big)$$

The **value (critic)** descends the squared advantage (a regression toward the $n$-step return):

$$d\theta_v \;\leftarrow\; d\theta_v \;+\; \nabla_{\theta_v}\,\tfrac{1}{2}\,\hat{A}(s_i,a_i)^2$$

where:
- $\hat{A}(s_i,a_i)$ in the actor term is treated as a **constant** (no gradient flows through the critic here)
- $H(\pi(\cdot\mid s;\theta)) = -\sum_a \pi(a\mid s;\theta)\log\pi(a\mid s;\theta)$ — policy [[Entropy]], pushing toward stochastic (exploratory) policies and away from premature collapse
- $\beta$ — entropy regularization coefficient (e.g. $0.01$)
- the actor and critic often **share** lower layers (a single body with two heads), so $\theta$ and $\theta_v$ overlap

## Key Properties / Variants

- **Asynchronous, lock-free updates (Hogwild!-style):** workers read the global parameters, compute gradients on a local copy, and apply them to the global network without locking. Stale gradients are tolerated rather than corrected.
- **No replay buffer:** parallelism provides decorrelation, keeping the method **on-policy**. This is the key structural difference from [[Deep Q-Network (DQN)]].
- **Runs on CPU:** the original results used a 16-core CPU, training faster (in wall-clock) than GPU DQN on Atari.
- **$n$-step returns** propagate reward to many preceding state-action pairs at once, speeding credit assignment versus 1-step methods.
- **Entropy regularization** is essential to maintain exploration and prevent the policy from collapsing to a near-deterministic distribution too early.
- **[[A2C]] (the synchronous variant):** removes asynchrony — a coordinator waits for all workers, averages their gradients, and applies a single batched update. Empirically A2C matches or beats A3C and is simpler/more GPU-efficient, suggesting the asynchrony itself was not the source of A3C's gains (the *parallelism* was).
- **General-advantage form:** the fixed $n$-step advantage can be replaced by [[Generalized Advantage Estimation|GAE]] for a smoother bias-variance trade-off.

```pseudo
Algorithm: A3C — per actor-learner thread
──────────────────────────────────────────────
Assume global shared params θ, θ_v and global counter T
Initialize thread step counter t ← 1

Repeat until T > T_max:
  Reset gradients:  dθ ← 0,  dθ_v ← 0
  Sync thread params:  θ' ← θ,  θ'_v ← θ_v
  t_start ← t
  Get state s_t
  Repeat:
    Sample a_t from policy π(·|s_t; θ')
    Take a_t, observe r_t and s_{t+1}
    t ← t + 1;  T ← T + 1
  until s_t terminal OR (t - t_start == t_max)

  # Bootstrap from the last state
  R ← 0                  if s_t terminal
  R ← V(s_t; θ'_v)       otherwise

  For i = t-1 down to t_start:        # accumulate over rollout
    R ← r_i + γ R
    A ← R - V(s_i; θ'_v)              # n-step advantage
    dθ   ← dθ + ∇_θ' log π(a_i|s_i; θ') · A + β ∇_θ' H(π(·|s_i; θ'))
    dθ_v ← dθ_v + ∇_θ'_v (1/2) A^2

  Asynchronously apply dθ to θ and dθ_v to θ_v   (shared optimizer, e.g. RMSProp)
```

## Connections

- Is a deep, parallel instance of: [[Actor-Critic]] / [[Policy Gradient Methods]]
- Synchronous counterpart: [[A2C]]
- Builds directly on: [[REINFORCE]] (with [[Baseline]]) and the [[Policy Gradient Theorem]]
- Uses: [[Advantage Function]], [[Entropy]] regularization, $n$-step [[Return]]s, [[RMSProp]]
- Contrast with: [[Deep Q-Network (DQN)]] (off-policy, [[Experience Replay]] vs. parallel workers)
- Advantage can be generalized via: [[Generalized Advantage Estimation]]
- Successors / related deep PG methods: [[TRPO]], [[PPO]]

## Appears In

- [[Advantage Function]]
- [[Generalized Advantage Estimation]]
- [[REINFORCE]]
- [[RL-L09 - Policy Gradient Methods]]
- [[RL-Book Ch13 - Policy Gradient Methods]]
