---
type: concept
aliases: ["Gaussian stochastic policy", "Normal distribution policy"]
course: [RL]
tags: [policy-gradient, continuous-actions, stochastic]
status: complete
---

# Gaussian Policy

## Definition

A **Gaussian policy** is a stochastic policy for continuous action spaces that models the action distribution as a multivariate Gaussian (normal) distribution:

$$a \sim \mathcal{N}(\mu_\theta(s), \Sigma_\theta(s))$$

where:
- $\mu_\theta(s) \in \mathbb{R}^{|A|}$ is the **mean** (mean action), parameterized by $\theta$
- $\Sigma_\theta(s)$ is the **covariance matrix** (controls exploration magnitude)
- $a \in \mathbb{R}^{|A|}$ is the continuous action

## Intuition

For continuous control (e.g., robot joint angles, continuous force), we need a policy that:
- Learns a preferred action (the mean)
- Maintains uncertainty/exploration around that mean
- Adjusts both mean and variance based on state

Gaussian policy naturally provides this: it's differentiable, supported on $\mathbb{R}^{|A|}$, and captures both exploitation (mean) and exploration (variance).

## Mathematical Formulation

### Probability Density

For a diagonal Gaussian (common simplification):

$$\pi_\theta(a|s) = \prod_{i=1}^{|A|} \frac{1}{\sqrt{2\pi \sigma_i^2}} \exp\left(-\frac{(a_i - \mu_i(s))^2}{2\sigma_i^2}\right)$$

### Log-Policy (for gradient computation)

$$\log \pi_\theta(a|s) = -\sum_{i=1}^{|A|} \left[\frac{(a_i - \mu_i(s))^2}{2\sigma_i^2} + \log \sigma_i\right] + \text{const}$$

### Gradient w.r.t. Mean

$$\nabla_\theta \log \pi_\theta(a|s) = \frac{1}{\sigma^2}(a - \mu_\theta(s)) \nabla_\theta \mu_\theta(s)$$

**Interpretation**: Update mean in direction of the action error, scaled by inverse variance.

### Gradient w.r.t. Variance

$$\frac{\partial}{\partial \sigma} \log \pi_\theta(a|s) = \frac{(a - \mu(s))^2}{\sigma^3} - \frac{1}{\sigma}$$

This shows variance should increase when actions are far from mean, decrease when close.

## Key Properties/Variants

### Mean Parameterization

Common choices:

1. **Linear**: $\mu_\theta(s) = \theta^T \phi(s)$
   - Simple, interpretable
   - Good for linear relationships

2. **Neural network**: $\mu_\theta(s) = \text{NN}_\theta(s)$
   - Highly expressive
   - Standard for deep RL

### Variance Parameterization

1. **Fixed variance**: $\sigma$ is a hyperparameter, not learned
   - Simpler, faster
   - May require careful tuning

2. **Learned scalar variance**: One $\sigma$ per dimension
   - Adapts exploration per action dimension
   - Common in practice

3. **State-dependent variance**: $\sigma_\theta(s)$ also learned
   - Maximum flexibility
   - Needs careful initialization

4. **Log-variance**: Often parameterize $\log \sigma$ to ensure positivity

### Diagonal vs Full Covariance

- **Diagonal** (most common): $\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$
  - Simpler gradient computation
  - Assumes action dimensions are independent
  
- **Full covariance**: Allows correlation between actions
  - More expressive, more expensive
  - Rarely needed

## Connections

- **Related to**: [[Normal distribution]], [[Continuous control]]
- **Basis for**: [[Policy Gradient Methods]] for continuous actions
- **Alternative to**: [[Softmax Policy]] (which is for discrete actions)
- **Enables**: Smooth, differentiable action sampling

## Appears In

- [[Policy Gradient Methods]] — Standard for continuous action spaces
- [[REINFORCE]] — Continuous control variant
- [[Actor-Critic]] — Continuous action actor
- [[PPO]] — Continuous benchmark tasks
- [[Deep Deterministic Policy Gradient]] — Alternative to Gaussian (deterministic policy)
- [[Soft Actor-Critic (SAC)]] — Uses Gaussian policies with entropy regularization
