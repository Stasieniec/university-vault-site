---
type: concept
aliases: ["Soft-max policy", "Boltzmann policy"]
course: [RL]
tags: [policy-gradient, discrete-actions, stochastic, exploration]
status: complete
---

# Softmax Policy

## Definition

A **softmax policy** is a stochastic policy for discrete action spaces that converts action preference values (scores) into a probability distribution using the softmax function:

$$\pi_\theta(a|s) = \frac{\exp f_\theta(s,a)}{\sum_{a' \in A} \exp f_\theta(s,a')}$$

where:
- $f_\theta(s,a)$ is the **preference function** (can be linear, neural network, etc.)
- The softmax function normalizes preferences into valid probabilities
- Preferences with higher values get higher probability

## Intuition

Think of softmax as a "soft" version of argmax:
- **Argmax**: Picks action with highest preference with probability 1
- **Softmax**: Weights actions by their preference values, maintaining some probability for all actions

This built-in exploration is automatic: even low-preference actions retain probability mass. The temperature of exploration is controlled by the preference magnitudes.

## Mathematical Formulation

### Log-Policy Gradient

For implementation, the log-probability is:

$$\log \pi_\theta(a|s) = f_\theta(s,a) - \log \sum_{a'} \exp f_\theta(s,a')$$

The gradient w.r.t. $\theta$:

$$\nabla_\theta \log \pi_\theta(a|s) = \nabla_\theta f_\theta(s,a) - \mathbb{E}_{a' \sim \pi_\theta}[\nabla_\theta f_\theta(s,a')]$$

This shows two components:
- **Positive term**: Increases the preference for the taken action $a$
- **Negative term**: Decreases the expected preference (regularization toward balanced exploration)

### Properties

- **Normalized**: Always sums to 1 over actions
- **Differentiable**: Smooth w.r.t. $\theta$, enabling gradient-based optimization
- **Exploration**: All actions have positive probability (never zero unless explicit constraint)
- **Entropy**: Has natural entropy from probability distribution

## Key Properties/Variants

### Preference Function Choice

The preference function can be:
1. **Linear**: $f_\theta(s,a) = \theta^T \phi(s,a)$ (linear in features)
2. **Neural network**: $f_\theta(s,a) = \text{NN}_\theta(s,a)$ (nonlinear)
3. **Single output**: $f_\theta(s) \in \mathbb{R}^{|A|}$ (network outputs all action preferences at once)

### Temperature Scaling

Often seen with explicit temperature parameter:

$$\pi_\theta(a|s) = \frac{\exp(f_\theta(s,a) / \tau)}{\sum_{a'} \exp(f_\theta(s,a') / \tau)}$$

- **Low temperature** ($\tau \to 0$): More greedy, sharper distribution
- **High temperature** ($\tau \to \infty$): More exploration, uniform distribution

## Connections

- **Related to**: [[Boltzmann distribution]] in statistical mechanics
- **Basis for**: [[REINFORCE]] algorithm for discrete actions
- **Explores via**: [[Entropy]] of the policy distribution
- **Contrasts with**: [[Epsilon-Greedy Policy]] (less smooth, harder to optimize)

## Appears In

- [[Policy Gradient Methods]] — Most natural choice for discrete actions
- [[Actor-Critic]] — Policy component
- [[PPO]] — Soft policy parameterization
- [[Deep Reinforcement Learning]] — When using neural networks for preferences
