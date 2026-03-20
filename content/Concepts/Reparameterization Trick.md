---
type: concept
aliases: [reparameterization trick, reparametrization trick, pathwise derivative]
course: [RL]
tags: [deep-rl, optimization]
status: complete
---

# Reparameterization Trick

> [!definition] Reparameterization Trick
> A technique for computing gradients through stochastic sampling operations. Instead of sampling $a \sim \pi_\theta(\cdot|s)$ (which blocks gradient flow), express the sample as a **deterministic, differentiable function** of the parameters and independent noise: $a = f_\theta(\epsilon; s)$ where $\epsilon \sim \mathcal{N}(0, I)$.

## Intuition

> [!intuition] Making Randomness Differentiable
> Normally, you can't backpropagate through a random sampling step. The reparameterization trick moves the randomness into an input variable $\epsilon$ that doesn't depend on the parameters $\theta$. The actual sample becomes a deterministic transformation of $\epsilon$, so gradients can flow through $f_\theta$.

## For Gaussian Policies

If $\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$, then:

$$a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Now $\nabla_\theta a$ is well-defined:
$$\nabla_\theta a = \nabla_\theta \mu_\theta(s) + \nabla_\theta \sigma_\theta(s) \odot \epsilon$$

## Application in SAC

In [[Soft Actor-Critic (SAC)]], the reparameterization trick enables computing the policy gradient as:

$$\nabla_\theta J(\pi) = \nabla_\theta \mathbb{E}_\epsilon \left[ Q(s, f_\theta(\epsilon; s)) - \alpha \log \pi_\theta(f_\theta(\epsilon; s)|s) \right]$$

The expectation over $\epsilon$ doesn't depend on $\theta$, so the gradient moves inside the expectation.

## Key Properties

- Enables **lower-variance** gradient estimates compared to the log-derivative trick (REINFORCE)
- Works for continuous distributions that can be expressed as transformations of base distributions
- Standard technique in variational autoencoders (VAEs) and modern deep RL
- Requires the sampling distribution to be reparameterizable (Gaussian, etc.)

## Connections

- Used in [[Soft Actor-Critic (SAC)]] for policy optimization
- Related to [[Policy Gradient Methods]] — an alternative way to compute policy gradients
- Also used in Variational Autoencoders (VAEs)
- Provides lower variance than the [[REINFORCE]] log-derivative estimator

## Appears In

- [[RL-L11 - SAC, Decision Transformer & Diffuser]]
