---
type: concept
aliases: ["REINFORCE algorithm", "Monte Carlo policy gradient", "Vanilla policy gradient"]
course: [RL]
tags: [policy-gradient, algorithm, monte-carlo, on-policy]
status: complete
---

# REINFORCE

## Definition

**REINFORCE** is a Monte Carlo [[Policy Gradient Methods|policy gradient]] algorithm that directly implements the [[Policy Gradient Theorem]]. It updates policy parameters by sampling complete episodes (trajectories) and using the discounted return as a gradient weight.

The update rule:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G(\tau)$$

where $G(\tau) = \sum_{t=0}^{T-1} \gamma^t r_t$ is the return for the entire episode.

## Historical Significance

REINFORCE (Williams, 1992) was the first practical implementation of policy gradients. It:
- Proved policy gradients were feasible
- Avoided value function learning
- Provided unbiased gradient estimates
- Formed the basis for modern policy gradient methods ([[Actor-Critic]], [[PPO]], etc.)

## Intuition

### Core Idea

Sample trajectories from the current policy and update it to make **high-return episodes more likely**:

1. Roll out an episode from current policy $\pi_\theta$
2. Compute total return $G = \sum r_t$
3. For each step, increase log-probability of that action weighted by the episode return
4. Repeat

### Why It Works

- **Gradient ascent**: Each update increases expected return in the direction of the policy gradient
- **Unbiased**: The gradient estimate has the correct expectation
- **Model-free**: Doesn't require knowing the environment dynamics
- **General**: Works with any differentiable policy parameterization

### Why It's Limited

- **High variance**: Uses full episode return (compounds over time)
- **Slow**: Needs many episodes for reliable estimates
- **Episodic only**: Requires complete episodes (continuing tasks need horizons)
- **Credit assignment**: All actions share credit/blame for entire trajectory

## Algorithm

### Pseudocode

```
Initialize policy parameters θ
Repeat:
  τ ← sample episode from π_θ
  G ← return of τ = Σ γ^t r_t
  θ ← θ + α · G · Σ ∇_θ log π_θ(a_t|s_t)
```

### Batch Version (Clearer for Implementation)

```
Initialize policy parameters θ
Repeat:
  D ← sample N episodes under π_θ
  For i = 1 to N:
    G_i ← return of episode i
    ∇_i ← Σ ∇_θ log π_θ(a_{i,t}|s_{i,t})
  θ ← θ + (α/N) Σ G_i · ∇_i
```

### Continuous Action Example

For a Gaussian policy $a \sim \mathcal{N}}(\mu_\theta(s), \sigma)$:

1. Sample action: $a_t \sim \mathcal{N}(\mu_\theta(s_t), \sigma)$
2. Gradient: $\nabla_\theta \log \pi_\theta(a_t|s_t) = \frac{1}{\sigma^2}(a_t - \mu_\theta(s_t)) \nabla_\theta \mu_\theta(s_t)$
3. Update: $\mu \leftarrow \mu + \alpha G \cdot \frac{1}{\sigma^2}(a - \mu) \nabla \mu$

### Discrete Action Example

For a softmax policy $\pi_\theta(a|s) = \frac{\exp f_\theta(s,a)}{\sum_a \exp f_\theta(s,a)}$:

1. Sample action from softmax
2. Gradient: $\nabla_\theta \log \pi_\theta(a|s) = \nabla_\theta f_\theta(s,a) - \mathbb{E}[\nabla_\theta f_\theta(s,a')]$
3. Update policy accordingly

## Variants and Improvements

### REINFORCE with Baseline

Reduces variance by subtracting a learned value function:

$$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G - V(s_t))$$

- **Unbiased**: Value function doesn't depend on actions
- **Lower variance**: Compares to expected return from state
- **Practical improvement**: Typically needed for good performance

### REINFORCE v2 (Causality)

Only uses forward returns (return from time $t$ onward):

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

where $G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}$

- **Intuition**: Action $a_t$ can't affect rewards before time $t$
- **Variance reduction**: Removes unnecessary noise from past
- **Still unbiased**: Doesn't change expectation

### With Advantages

Use advantage function $A(s,a) = Q(s,a) - V(s)$ from learned [[Actor-Critic]] setup:

$$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t)$$

- **Maximum flexibility**: Can approximate $Q$ and $V$ separately
- **Modern form**: Basis of [[A2C]], [[PPO]], etc.

## Properties

### Strengths

✓ **Unbiased**: Converges to stationary points of $J(\theta)$
✓ **Consistent**: Sample average converges to true gradient
✓ **General**: Works with any differentiable policy
✓ **Model-free**: Needs only policy samples, not dynamics
✓ **Simple**: Easy to implement and understand
✓ **Handles stochasticity**: Works with stochastic optimal policies

### Weaknesses

✗ **High variance**: Full return has high variance
✗ **Sample inefficiency**: Needs many episodes
✗ **Slow convergence**: Can be slower than value-based methods
✗ **Episodic**: Requires complete episodes
✗ **Credit assignment**: Actions credited equally for entire trajectory
✗ **Deterministic policies**: Can't learn truly deterministic optimal policies

## Practical Considerations

### Gradient Magnitude Issues

Returns $G$ can vary wildly (positive/negative, large magnitudes):
- **Solution 1**: Normalize returns: $(G - \bar{G}) / \sigma_G$
- **Solution 2**: Use advantage: $G - V(s)$ (learned baseline)
- **Solution 3**: Reduce baseline per-state: $G - V(s_0)$ (constant)

### Step Size Tuning

- Learning rate $\alpha$ is critical (0.01 to 0.001 typical)
- Too high: Divergence
- Too low: Very slow learning
- Often use adaptive learning rates (Adam, RMSprop)

### Variance Reduction in Practice

Order of importance:
1. **Baseline** (most important): Reduces variance dramatically
2. **Causality** (moderate): Cuts one source of variance
3. **Normalization** (helpful): Stabilizes learning
4. **Entropy regularization** (optional): Encourages exploration

## Connections

- **Implements**: [[Policy Gradient Theorem]]
- **Foundation for**: [[Actor-Critic]], [[A2C]], [[A3C]]
- **Related to**: [[Monte Carlo Methods]], [[Gradient Ascent]]
- **Uses**: [[Policy]] parameterization (softmax, Gaussian, etc.)
- **Requires**: [[Differentiable policy]]

## Modern Context

REINFORCE is largely superseded by more advanced algorithms ([[PPO]], [[A3C]]), but:
- Still the simplest policy gradient algorithm
- Educational value: teaches core principles
- Effective with proper baselines and variance reduction
- Used in some simple domains

## Appears In

- [[Policy Gradient Methods]] — Foundational algorithm
- [[Actor-Critic]] — Extensions with learned value
- [[Advantage Actor-Critic (A2C)]] — Direct successor
- [[Policy Gradient Methods]] — Core RL course topic
- [[Deep Reinforcement Learning]] — When optimizing neural network policies
