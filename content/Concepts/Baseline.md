---
type: concept
aliases: ["Baseline function", "Value baseline", "Advantage baseline"]
course: [RL]
tags: [variance-reduction, policy-gradient, reinforcement-learning]
status: complete
---

# Baseline

## Definition

A **baseline** is a reference value (typically depending on state only) that is subtracted from returns in policy gradient methods to reduce variance without introducing bias.

In policy gradient updates, we use:

$$\nabla_\theta J \approx \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot (G_t - b(s_t))]$$

where $b(s)$ is the baseline, commonly a learned [[Value Function]] estimate $V(s)$.

## Intuition

### The Problem

In vanilla [[REINFORCE]], all actions in a trajectory share credit/blame for the total return:

$$\nabla_\theta J \propto \nabla_\theta \log \pi_\theta(a|s) \cdot G$$

where $G$ is the return from the start of the episode. This means:
- Good early actions get blamed for bad later actions
- Bad early actions get credit for good later rewards
- **High variance**: Lots of noise in the gradient estimates

### The Solution

Subtract a baseline that represents "what was expected from this state":

$$G_t - V(s_t) = \text{Actual return} - \text{Expected return} = \text{Advantage}$$

The baseline:
- Reduces variance: Returns are centered around expected value
- Doesn't change expectation: $\mathbb{E}[b(s)] = 0$ w.r.t. actions sampled from $\pi(a|s)$
- Helps credit assignment: Actions are compared to state-dependent baseline

## Mathematical Formulation

### Why Baselines Don't Introduce Bias

The key insight:

$$\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi(a|s) \cdot b(s)] = b(s) \mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi(a|s)]$$

The gradient of log probabilities sums to zero (since probabilities sum to 1):

$$\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi(a|s)] = 0$$

Therefore: **Subtracting any baseline maintains unbiasedness.**

### Causality-Aware Baselines

In practice, we use causality: action $a_t$ only affects rewards from time $t$ onward:

$$\nabla_\theta J = \mathbb{E}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi(a_t|s_t) \left(G_t - b(s_t)\right)\right]$$

where $G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}$ is the return from step $t$ onward.

### Advantage Function

When $b(s_t) = V(s_t)$, the difference is the **advantage**:

$$A_t = G_t - V(s_t) = \text{how much better than expected}$$

This is a core concept in modern RL ([[Advantage function]]).

## Key Properties/Variants

### Choice of Baseline

1. **Constant baseline**: $b(s) = c$ (average return)
   - Simplest, provides some variance reduction
   - Not state-dependent

2. **Linear value function**: $V(s) = w^T \phi(s)$
   - Parametric, simple to learn
   - Good for linear relationships

3. **Neural network value**: $V(s) = \text{NN}_w(s)$
   - Highly expressive
   - Standard in modern deep RL

4. **Temporal difference targets**: $V(s) \approx r + \gamma V(s')$
   - One-step lookahead
   - Reduces variance further but introduces bias

### Learning the Baseline

Typically minimize MSE on observed returns:

$$\mathcal{L}_V = \mathbb{E}[(G_t - V(s_t))^2]$$

Update: $w \leftarrow w - \beta \nabla_w (G_t - V(s_t))^2$

Or TD-style:

$$\mathcal{L}_V = \mathbb{E}[(r_t + \gamma V(s_{t+1}) - V(s_t))^2]$$

### Variance Reduction Effectiveness

The amount of variance reduction depends on how well the baseline correlates with returns:

- **Bad baseline**: Little variance reduction
- **Good baseline** (close to actual $V(s)$): Significant variance reduction
- **Perfect baseline** (true $V(s)$): Minimal variance remains

In practice, a learned value function usually provides substantial variance reduction even if imperfect.

## Connections

- **Versus**: [[Return|Full trajectory return]] (high variance)
- **Related to**: [[Advantage function]] ($G_t - V(s)$)
- **Learned via**: [[Temporal Difference Learning]] or [[Monte Carlo Methods]]
- **Core in**: [[Actor-Critic]] (separate value baseline) and [[A2C]] algorithms
- **Implies**: [[Value Function]] is useful even in policy-gradient methods

## Appears In

- [[Policy Gradient Methods]] — Variance reduction technique
- [[REINFORCE]] — Common improvement (REINFORCE with baseline)
- [[Actor-Critic]] — Separates actor (policy) from critic (baseline/value)
- [[Advantage Actor-Critic (A2C)]] — Uses value baseline
- [[PPO]] — Reduces variance significantly
- [[Deep Reinforcement Learning]] — Essential for sample efficiency
