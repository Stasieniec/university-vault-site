---
type: concept
aliases: ["Policy gradient", "Gradient ascent for policies", "Log-derivative trick"]
course: [RL]
tags: [policy-gradient, theoretical-foundation, gradient-ascent]
status: complete
---

# Policy Gradient Theorem

## Definition

The **Policy Gradient Theorem** is a fundamental result in reinforcement learning that expresses the gradient of expected return with respect to policy parameters:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[G(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

This equation is the foundation for all policy gradient methods. It says: **to increase expected return, increase the log-probability of actions in high-return trajectories.**

## Intuition

### Why This Makes Sense

Imagine sampling episodes (trajectories) from your current policy:
- Episodes with high total reward should be reinforced
- Episodes with low total reward should be deprioritized
- The log-probability gradient acts as a "handle" to adjust the policy

The algorithm works by:
1. Sample an episode and measure its return $G$
2. Compute $\nabla_\theta \log \pi_\theta(a|s)$ for each action
3. Update: $\theta \leftarrow \theta + \alpha \cdot G \cdot \nabla_\theta \log \pi_\theta(a|s)$

**Result**: Good actions become more likely, bad actions less likely.

### The Log-Derivative Trick

The key technical insight is the **log-derivative trick**:

$$\nabla_x f(x) = f(x) \nabla_x \log f(x)$$

This allows us to move the gradient inside an expectation with respect to a distribution that depends on the parameters:

$$\nabla_\theta \mathbb{E}_{x \sim p_\theta(x)}[f(x)] = \mathbb{E}_{x \sim p_\theta(x)}[f(x) \nabla_\theta \log p_\theta(x)]$$

This is why we work with log-probabilities: they make the gradient tractable.

## Mathematical Derivation

### Starting Point

For an episodic task with trajectories $\tau = (s_0, a_0, r_0, \ldots, s_T)$:

$$J(\theta) = \mathbb{E}_\tau[G(\tau)] = \int p_\theta(\tau) G(\tau) d\tau$$

### Applying the Log-Derivative Trick

$$\nabla_\theta J = \nabla_\theta \int p_\theta(\tau) G(\tau) d\tau = \int \nabla_\theta p_\theta(\tau) G(\tau) d\tau$$

Using the log-derivative trick:

$$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$$

Therefore:

$$\nabla_\theta J = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) G(\tau) d\tau = \mathbb{E}_\tau[\nabla_\theta \log p_\theta(\tau) \cdot G(\tau)]$$

### Factoring the Trajectory Probability

The trajectory probability factors as:

$$p_\theta(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) \cdot p(s_{t+1}|a_t, s_t)$$

The log:

$$\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T-1} [\log \pi_\theta(a_t|s_t) + \log p(s_{t+1}|a_t, s_t)]$$

Gradient w.r.t. $\theta$ (only the policy term depends on $\theta$):

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

### Final Result

Substituting back:

$$\nabla_\theta J = \mathbb{E}_\tau\left[G(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

## Key Properties

### Unbiasedness

The estimator is **unbiased**: sampling one trajectory gives an unbiased estimate of the true gradient.

$$\mathbb{E}[\hat{\nabla}_\theta J] = \nabla_\theta J$$

### Consistency

With enough samples, the empirical average converges to the true gradient (by law of large numbers).

### On-Policy

The theorem requires samples from the current policy $\pi_\theta$. Using samples from a different policy (off-policy) requires importance sampling correction.

### Direct Dependency on Dynamics is Not Needed

Crucially, the dynamics $p(s_{t+1}|a_t, s_t)$ **cancel out** in the gradient. We don't need to know or learn the environment dynamics!

## Variations and Extensions

### Causality-Aware Version

We can improve variance by noting that action $a_t$ only affects rewards at time $t$ onward:

$$\nabla_\theta J = \mathbb{E}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

where $G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}$.

### With Baseline

Subtracting any baseline $b(s)$ preserves unbiasedness:

$$\nabla_\theta J = \mathbb{E}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]$$

### Continuous Time / Continuing Tasks

The theorem extends to continuing MDPs with discounted returns:

$$J(\theta) = \mathbb{E}[\sum_{t=0}^\infty \gamma^t r_t]$$

### State Value Version

Can also express as:

$$\nabla_\theta J(\theta) \propto \mathbb{E}_{s \sim \rho(s)}[\mathbb{E}_{a \sim \pi_\theta(a|s)}[\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a)]]$$

where $\rho(s)$ is state visitation distribution and $Q^\pi$ is the action-value function.

## Practical Implications

### Algorithm Design

The theorem motivates:

1. **[[REINFORCE]]**: Use Monte Carlo sample of return $G$ directly
2. **[[Actor-Critic]]**: Use learned $Q(s,a)$ or $V(s)$ estimate instead of full $G$
3. **[[PPO]]**: Efficient trust-region variant of policy gradients
4. **[[A2C]]**: Parallel actor-critic with baseline

### Gradient Variance

The theorem shows variance comes from:
- **Return sampling**: Monte Carlo returns have high variance
- **Policy stochasticity**: Exploration adds noise

Variance reduction techniques:
- **Baselines**: Subtract expected return $V(s)$
- **Advantage estimates**: Use $Q(s,a) - V(s)$ instead of raw returns
- **Function approximation**: Smooth out noisy returns

## Connections

- **Foundation of**: [[Policy Gradient Methods]], [[Actor-Critic]], [[PPO]]
- **Related to**: [[Log derivative trick]], [[Gradient ascent]]
- **Assumes**: [[Policy]] is differentiable w.r.t. parameters
- **Versus**: [[Bellman equation]] (basis of value-based methods)
- **Enables**: Model-free learning (no dynamics needed)

## Appears In

- [[Policy Gradient Methods]] — Core theoretical foundation
- [[REINFORCE]] — Direct application
- [[Actor-Critic]] — Extends with learned value
- [[Advantage Actor-Critic (A2C)]] — With baselines
- [[PPO]] — Trust-region variant
- [[Deep Reinforcement Learning]] — When using neural network policies
