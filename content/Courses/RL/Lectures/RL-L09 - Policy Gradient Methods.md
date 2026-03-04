---
type: lecture
course: RL
week: 5
lecture: 9
book_sections: ["Ch 13.1", "Ch 13.2", "Ch 13.3", "Ch 13.4", "Ch 13.7"]
topics:
  - "[[Policy Gradient Methods]]"
  - "[[REINFORCE]]"
  - "[[Policy Gradient Theorem]]"
  - "[[Baseline]]"
  - "[[Softmax Policy]]"
  - "[[Gaussian Policy]]"
  - "[[Actor-Critic]]"
status: complete
---

# RL-L09: Policy Gradient Methods

## Overview

This lecture introduces **policy-based methods**, which directly optimize the parameters of a policy function rather than learning a value function. While value-based methods learn $V(s)$ or $Q(s,a)$ and derive a deterministic policy from them, policy-based methods explicitly parameterize $\pi_\theta(a|s)$ and optimize it using gradient ascent on the expected return.

Policy-based methods address key limitations of action-value methods:
- Handle continuous action spaces naturally (no argmax required)
- Learn stochastic policies (useful for partial observability and exploration)
- Provide policy smoothness guarantees through step size control
- Allow incorporation of prior knowledge via policy structure

The core insight is the **policy gradient theorem**: we can compute an unbiased gradient of expected return w.r.t. policy parameters using only samples of trajectories.

---

## Why Policy-Based Methods?

### Limitations of Action-Value Methods

> [!warning] **Key Problems with Value-Based Approaches**
> 1. **Continuous actions**: Can't efficiently compute $\max_a Q(s,a)$ in continuous action spaces
> 2. **Policy instability**: Small changes in $Q$-values can cause large changes in the greedy policy
> 3. **Stochastic policies**: Impossible to learn stochastic optimal policies (e.g., mixed strategies, handling aliased states)
> 4. **Exploration**: $\epsilon$-greedy exploration is crude; can't learn optimal exploration strategy

**Example of aliased states**: If two different states look identical to the agent due to function approximation, the greedy policy might pick the same action for both. A stochastic policy choosing each action with 50% probability could be optimal.

---

## Policy Representation

### Stochastic Policies

Instead of learning a value function, we directly parameterize the policy as:

$$\pi_\theta(a|s) : \text{probability of action } a \text{ in state } s$$

Requirements:
- **Differentiability**: $\pi_\theta$ must be differentiable w.r.t. $\theta$ (to compute gradients)
- **Stochasticity**: Outputs a valid probability distribution over actions

### Softmax Policy (Discrete Actions)

For discrete action spaces, use softmax over action preferences:

$$\pi_\theta(a|s) = \frac{\exp f_\theta(s,a)}{\sum_{a' \in A} \exp f_\theta(s,a')}$$

where $f_\theta(s,a)$ can be linear, neural network, or any differentiable function.

> [!intuition]
> The softmax policy acts like a "soft" argmax: preferences with higher values get higher probability, but all actions retain some probability. The temperature-like behavior makes exploration automatic.

### Linear Gaussian Policy (Continuous Actions)

For continuous action spaces, parameterize a Gaussian distribution:

$$a \sim \mathcal{N}(\theta^T \phi(s), \sigma)$$

where:
- Mean: linear in state features $\phi(s)$ with weight $\theta$
- Variance: $\sigma$ (can be fixed or learned)

### Neural Network Policies (Continuous Actions)

With neural networks, output both mean and variance:

$$a \sim \mathcal{N}(\text{NN}_{\theta_\mu}(s), \text{NN}_{\theta_\sigma}(s))$$

This gives highly flexible, nonlinear action selection.

---

## The Policy Gradient Theorem

### Objective Function

Every policy $\pi_\theta$ has an expected return:

$$J(\theta) = \mathbb{E}_\tau[G(\tau)] = \mathbb{E}_{s_0, a_0, r_0, \ldots}[\sum_{t=0}^{T-1} \gamma^t r_t]$$

We want to find: $\theta^* = \arg\max_\theta J(\theta)$

Using **gradient ascent**: $\theta_{t+1} = \theta_t + \alpha \nabla J(\theta_t)$

### Deriving the Gradient

Starting from the definition of $J(\theta)$ for episodic tasks:

$$\nabla_\theta J = \nabla_\theta \mathbb{E}_\tau[G(\tau)] = \nabla_\theta \int p_\theta(\tau) G(\tau) d\tau$$

Using the **log-derivative trick** ($\nabla_x \log f(x) = \frac{\nabla_x f(x)}{f(x)}$):

$$\nabla_\theta J = \int \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} p_\theta(\tau) G(\tau) d\tau = \mathbb{E}_\tau[\nabla_\theta \log p_\theta(\tau) \cdot G(\tau)]$$

### Factoring the Trajectory Probability

The trajectory probability factors as:

$$p_\theta(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) \cdot p(s_{t+1}|a_t, s_t)$$

Taking the log and gradient, only the policy terms depend on $\theta$:

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

(The dynamics and initial state gradients are zero)

### Final Result: The Policy Gradient Theorem

> [!formula] **Policy Gradient Theorem (Episodic)**
> $$\nabla_\theta J(\theta) = \mathbb{E}_\tau\left[G(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$
> 
> **Interpretation**: To increase expected return, increase the log-probability of actions in high-return trajectories.

---

## REINFORCE: The Original Policy Gradient Algorithm

### Algorithm

The simplest practical implementation: sample trajectories and average the gradient estimate.

> [!definition] **REINFORCE Algorithm**
> 
> **Hyperparameters**: Step size $\alpha$, episode length $T$
> 
> **Repeat**:
> 1. Sample an episode (trajectory): $\tau = (s_0, a_0, r_0, \ldots, s_T)$
> 2. Compute return: $G = \sum_{t=0}^{T-1} \gamma^t r_t$
> 3. Update policy:
> $$\theta \leftarrow \theta + \alpha \cdot G \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$
> 
> Or with $N$ sampled trajectories (batch update):
> $$\hat{\nabla} J = \frac{1}{N} \sum_{i=1}^{N} G(\tau_i) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})$$

### Example: Bernoulli Policy

For a Bernoulli policy with two actions:
$$\pi_\theta(a_1|s) = \theta, \quad \pi_\theta(a_0|s) = 1-\theta$$

Gradient computation:
$$\nabla_\theta \log \pi_\theta(a_1|s) = \frac{1}{\theta}$$
$$\nabla_\theta \log \pi_\theta(a_0|s) = -\frac{1}{1-\theta}$$

Update rule: if action $a_1$ was taken and return is $G$:
$$\theta \leftarrow \theta + \alpha G \cdot \frac{1}{\theta}$$

### Properties

> [!tip]
> - **Unbiased**: $\mathbb{E}[\hat{\nabla} J] = \nabla J$ — our estimate has correct expectation
> - **Consistent**: Converges as sample size increases
> - **Easy**: Just requires computing log-policy gradients, no need to know dynamics
> - **On-policy**: Must sample from current policy $\pi_\theta$

### Limitations

- **High variance**: Uses full trajectory return, which compounds over time
- **Episodic only**: Requires episodes of defined length
- **Slow learning**: May need many episodes to estimate gradient accurately

---

## REINFORCE with Baseline

### Motivation

A fundamental issue: **all actions in a trajectory share credit/blame for the final return**.

> [!intuition]
> If an episode has:
> - Time $t=0$: good action → good reward
> - Time $t=1$: bad action → bad reward
> - ...
> - Time $t=T-1$: mediocre action
> 
> The REINFORCE update uses the same total return $G$ for all actions. The good action gets blamed for later bad actions, and the bad action gets credit for early good rewards.

### The Fix: Causality-Aware Gradient

**Key insight**: Action $a_t$ can only affect rewards at time $t$ and later, not before!

$$\nabla_\theta J = \mathbb{E}_\tau\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \sum_{t'=t+1}^{T-1} \gamma^{t'-t} r_{t'}\right]$$

This only uses the **return from time $t$ onward**:

$$G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}$$

> [!formula] **REINFORCE v2 (with causality)**
> $$\hat{\nabla} J = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \cdot G_{i,t}$$
> where $G_{i,t} = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{i,t'}$

### Adding a Baseline

Further variance reduction: subtract any baseline $b(s_t)$ (typically learned value function):

$$\nabla_\theta J \approx \mathbb{E}_\tau\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \left(G_t - b(s_t)\right)\right]$$

**Why valid?** The baseline doesn't depend on $a_t$, so:
$$\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = b(s) \mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s)] = 0$$

> [!definition] **Baseline**
> A baseline is any function $b(s)$ that estimates the expected return from a state. Commonly, use a learned value function $V(s)$ or simple running average. Baselines reduce variance without introducing bias.

### Learning the Baseline

Often learn $V(s)$ alongside the policy using TD or MC updates:

$$V(s) \approx \mathbb{E}[G_t | s_t = s]$$

This is straightforward: for each state visited with return $G$, update:
$$V(s) \leftarrow V(s) + \beta(G - V(s))$$

---

## Alternative Parametrizations

### Softmax Policy Details

For action preferences $f_\theta(s,a)$:

$$\log \pi_\theta(a|s) = f_\theta(s,a) - \log \sum_{a'} \exp f_\theta(s,a')$$

Gradient:
$$\nabla_\theta \log \pi_\theta(a|s) = \nabla_\theta f_\theta(s,a) - \mathbb{E}_{a' \sim \pi_\theta}[\nabla_\theta f_\theta(s,a')]$$

This naturally includes an exploration bonus (the second term subtracts the expected preference gradient).

### Gaussian Policy (Continuous)

For $a \sim \mathcal{N}(\mu_\theta(s), \sigma)$:

$$\log \pi_\theta(a|s) = -\frac{1}{2\sigma^2}(a - \mu_\theta(s))^2 + \text{const}$$

Gradient w.r.t. $\theta$:
$$\nabla_\theta \log \pi_\theta(a|s) = \frac{1}{\sigma^2}(a - \mu_\theta(s)) \nabla_\theta \mu_\theta(s)$$

This shows: **increase the mean in the direction of good actions**, weighted by how far they were from the current mean.

---

## Comparison: Policy Gradient Methods

### REINFORCE vs Finite Differences

| Aspect | Finite Differences | REINFORCE |
|--------|-------------------|-----------|
| **Gradient type** | Black-box (0-order) | White-box (1st-order) |
| **Variance** | Very high (noisy evaluation) | Lower (one sample per step) |
| **Efficiency** | Low (needs many rollouts) | Higher |
| **Requires differentiability** | No | Yes |

### REINFORCE v2 vs Original REINFORCE

| Aspect | Original | v2 (causality) | v2 + Baseline |
|--------|----------|----------------|---------------|
| **Unbiased** | ✓ | ✓ | ✓ |
| **Variance** | High | Lower | Much lower |
| **Implementation** | Simple | Simple | Requires value learning |
| **Practical performance** | Poor | Good | Best |

---

## Strengths and Weaknesses

### Advantages of Policy-Based Methods

> [!tip]
> 1. **Continuous actions**: Natural handling without discretization
> 2. **Stochastic policies**: Can learn optimal exploration/randomness
> 3. **Convergence guarantees**: To local optimum under mild conditions
> 4. **Prior knowledge**: Easy to initialize with expert policies
> 5. **Smooth updates**: Step size control → smooth policy changes

### Weaknesses

> [!warning]
> 1. **High variance**: Monte Carlo returns have high variance, especially for long episodes
> 2. **Episodic setting**: Current algorithms require complete episodes
> 3. **Deterministic policies**: Can't learn truly deterministic optimal policies (though near-deterministic is possible)
> 4. **Computational cost**: Need many trajectory samples to estimate gradients reliably
> 5. **Slow convergence**: Can be slower than value-based methods

---

## Key Concepts Introduced

### New Concepts (Concept Notes Created)

The following new concepts are introduced in this lecture and deserve separate study:

1. **[[Softmax Policy]]** - Stochastic policy using softmax over action preferences
2. **[[Gaussian Policy]]** - Stochastic policy for continuous actions as Gaussian distribution
3. **[[Baseline]]** - Value function subtracted from returns to reduce variance in policy gradients
4. **[[Policy Gradient Theorem]]** - Fundamental result: gradient of expected return w.r.t. policy parameters
5. **[[REINFORCE]]** - Monte Carlo policy gradient algorithm

### Existing Concepts Referenced

- [[Policy Gradient Methods]] - Central topic
- [[Reinforcement Learning]] - Field
- [[Policy]] - Parameterized as $\pi_\theta(a|s)$
- [[Return]] - Discounted sum of rewards $G$
- [[Discount Factor]] - $\gamma$
- [[Stochastic Gradient Descent]] - Optimization method
- [[Function Approximation]] - Using neural networks for $\pi_\theta$
- [[Neural Networks]] - For policy representation
- [[Gradient Descent]] - Core update rule $\theta \leftarrow \theta + \alpha \nabla J$
- [[Value Function]] - $V(s)$ as baseline
- [[Markov Decision Process]] - Underlying environment model
- [[Monte Carlo Methods]] - REINFORCE uses MC sampling
- [[Exploration vs Exploitation]] - Handled via policy stochasticity
- [[On-Policy Learning]] - Must sample from $\pi_\theta$
- [[Temporal Difference Learning]] - Value learning alternative to MC
- [[Deep Reinforcement Learning]] - When using neural networks

---

## Summary and Takeaways

> [!example] **Big Picture**
> 
> Policy gradient methods directly optimize policy parameters $\theta$ using gradient ascent. The **policy gradient theorem** gives us:
> 
> $$\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t]$$
> 
> This says: *increase log-probability of actions with high return*.
> 
> **REINFORCE** implements this via Monte Carlo sampling. Improvements:
> - **Causality**: Use only forward returns $G_t$ (not full trajectory return)
> - **Baseline**: Subtract value function $V(s)$ to reduce variance
> 
> These methods naturally handle:
> - Continuous action spaces
> - Stochastic optimal policies
> - Exploration via policy entropy
> 
> But they struggle with:
> - Variance from long episodes
> - Sample efficiency
> - Episodic-only settings (so far)

### Exam-Ready Facts

- Policy gradients = directly optimize $\pi_\theta$, not value function
- Core equation: $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) G_t]$
- REINFORCE: unbiased but high variance
- Baselines reduce variance without introducing bias
- Softmax for discrete, Gaussian for continuous actions
- On-policy: must sample from current policy
- Advantages: continuous actions, stochastic policies, smooth updates
- Disadvantages: high variance, slow convergence, episodic only (in basic form)
