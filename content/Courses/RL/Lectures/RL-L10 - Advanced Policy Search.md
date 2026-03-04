---
type: lecture
course: RL
week: 5
lecture: 10
book_sections: ["Ch 13.5"]
topics:
  - "[[Policy Gradient Methods]]"
  - "[[Policy Gradient Theorem]]"
  - "[[Actor-Critic]]"
  - "[[Deterministic Policy Gradient]]"
  - "[[Natural Policy Gradient]]"
status: complete
---

# RL Lecture 10 - Advanced Policy Search Methods

## Overview & Motivation

This lecture builds on foundational policy gradient methods, moving from simple stochastic policies to more sophisticated approaches. We tackle two main challenges: **high variance** in policy gradient estimates and **the need for deterministic policies** in continuous action spaces. The lecture introduces the Policy Gradient Theorem (PGT), actor-critic methods that use critics to reduce variance, and deterministic policy gradients for learning greedy policies off-policy.

The core tension in policy-based RL is the bias-variance tradeoff: Monte Carlo returns are unbiased but high-variance, while TD estimates are lower-variance but potentially biased.

---

## REINFORCE v2: Revisited

Before moving to advanced methods, we recap the monte-carlo policy gradient approach:

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \sum_{t'=t+1}^T \gamma^{t'-t} r_{t'} \right]$$

This can be estimated as:
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) G_t^{(i)}$$

> [!tip]
> Only future rewards ($t'  > t$) contribute to the gradient at time $t$, since only future actions affect future rewards (causality).

---

## Policy Gradient Theorem

### Formal Statement

The **Policy Gradient Theorem** elegantly replaces hard-to-estimate expected returns with the action-value function:

$$\nabla J(\pi_\theta) = \mathbb{E}_{\tau \sim \mu^\pi, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) Q^\pi(s, a) \right]$$

Equivalently (restating the expectation):
$$\nabla J(\pi_\theta) \propto \mathbb{E}_{\mu(s) \pi(a|s)} [\nabla_\theta \log \pi(a|s) q^\pi(s, a)]$$

where $\mu(s)$ is the on-policy distribution of states.

### Key Insight

- State-action pairs at each timestep **contribute equally** to the total gradient
- This is proportional to an expectation over the on-policy distribution over state-action pairs
- We can replace expected returns $G_t$ with learned $Q$ estimates

### Reference
See *Sutton et al., Policy Gradient Methods for Reinforcement Learning with Function Approximation* for the formal proof (including the continuing case).

---

## Actor-Critic Methods

### Motivation: Addressing Variance

> [!intuition]
> Instead of waiting for the full episode return (Monte Carlo), we can use a learned critic to estimate the value, reducing variance at the cost of potential bias.

### The Actor-Critic Update Rule

$$\theta_{t+1} = \theta_t + \alpha (R_{t+1} + \hat{v}(s_{t+1}, w)) \nabla_\theta \log \pi(a_t|s_t, \theta_t)$$

where:
- **Actor**: the policy $\pi(a|s, \theta)$ updated using policy gradients
- **Critic**: value function $\hat{v}(s, w)$ (or action-value $\hat{Q}(s,a,w)$) estimating the discounted future return

> [!definition]
> **Actor-Critic**: A method that uses both a parametrized policy (actor) and a parametrized value function (critic), with parameters $\theta$ and $w$ respectively.

### Actor-Critic with Baseline

We can further improve by subtracting a baseline $v̂(s_t, w)$:

$$\theta_{t+1} = \theta_t + \alpha (R_{t+1} + \hat{v}(s_{t+1}, w) - \hat{v}(s_t, w)) \nabla_\theta \log \pi(a_t|s_t, \theta_t)$$

The quantity in parentheses is the **temporal difference (TD) error**, or **advantage**:
$$\delta = R_{t+1} + \gamma \hat{v}(s_{t+1}, w) - \hat{v}(s_t, w)$$

> [!formula]
> **TD Advantage Estimate**:
> - **Advantage Function**: $A(s,a) = Q(s,a) - V(s)$
> - In practice: $\hat{A}_t = R_{t+1} + \gamma \hat{v}(s_{t+1}) - \hat{v}(s_t)$
> - Removes the value baseline, reducing variance while remaining unbiased if critic is perfect

### Advantages & Disadvantages

**Advantages**:
- Reduces variance compared to REINFORCE (Monte Carlo)
- Can be used in both episodic and continuing settings
- More sample-efficient than pure actor-only methods

**Disadvantages**:
- Introduces bias (critic may be inaccurate)
- "Fiddly": requires managing two function approximators (actor + critic)
- Requires stochastic policies (what if a deterministic policy is optimal?)
- Many hyperparameters and moving parts

---

## Deterministic Policy Gradients (DPG)

### Motivation: From Stochastic to Deterministic

> [!intuition]
> All policy gradients so far learned a **stochastic policy**. But deterministic policies can be more sample-efficient and optimal for many tasks (e.g., continuous control where the optimal action is to apply maximum force).

**Problem with direct application**: With deterministic policies, the action distribution has zero variance, so $\nabla_\theta \log \pi_\theta(a|s) = 0$ and the standard policy gradient is zero.

### The DPG Idea: Off-Policy Learning

To address this, we use **two policies**:
- **Behavior policy** $\beta(a|s)$: typically $\pi(a|s) + \text{noise}$ (e.g., Gaussian noise)
- **Target/actor policy** $\pi_\theta(a|s)$: deterministic, what we want to learn

We collect data under $\beta$ but optimize $\pi_\theta$.

### Changing the Objective

Instead of maximizing returns under the policy we're learning, we optimize:

$$J(\pi_\theta) = \int_S \rho^\beta(s) V^\pi(s) \, ds = \int_S \rho^\beta(s) Q^\pi(s, \pi_\theta(s)) \, ds$$

where $\rho^\beta(s)$ is the **state distribution under behavior policy $\beta$**.

> [!key point]
> **Crucial**: This objective depends only on **states**, not sampled actions! We integrate over state space only, removing the need for importance weights on actions.

### The Deterministic Policy Gradient

Taking the gradient of the objective:

$$\nabla_\theta J(\pi_\theta) = \int_S \rho^\beta(s) \nabla_\theta Q^\pi(s, \pi_\theta(s)) \, ds$$

Using the chain rule:
$$\nabla_\theta Q^\pi(s, a) = \nabla_a Q^\pi(s,a) \nabla_\theta \pi_\theta(s)$$

At $a = \pi_\theta(s)$:

$$\boxed{\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s \sim \rho^\beta} [\nabla_\theta \pi_\theta(s) \nabla_a Q^\pi(s, a)|_{a=\pi_\theta(s)}]}$$

This is the **off-policy deterministic policy gradient**.

> [!formula]
> **DPG**: 
> - Deterministic actor: $\mu_\theta(s)$ (output an action, not a distribution)
> - Critic: $Q^\pi(s,a)$ estimates action-value
> - Gradient: $\nabla_\theta J(\mu_\theta) = \mathbb{E}_s [\nabla_a Q(s,a)|_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s)]$
> - No importance sampling needed!

### Advantages of DPG

- **Off-policy learning**: Can learn from data collected by any behavior policy
- **Deterministic policy**: Can learn exact greedy policies
- **No importance sampling**: Avoids variance explosion from importance weights
- **More sample efficient**: Better for continuous control

### Connection to Q-Learning

Like Q-learning, DPG finds the greedy policy that maximizes $Q$ (but tractably for continuous actions):
$$\pi(s) = \arg\max_a Q^\pi(s,a)$$

---

## Natural Policy Gradient (NPG)

> [!definition]
> **Natural Gradient**: Instead of moving in the direction of steepest ascent in parameter space, move in the direction of steepest ascent in **policy space** (as measured by KL divergence).

The standard (vanilla) gradient is:
$$\nabla J = \text{gradient in parameter space}$$

The natural gradient is:
$$\tilde{\nabla} J = F^{-1} \nabla J$$

where $F$ is the **Fisher Information Matrix**:
$$F = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T]$$

> [!intuition]
> The Fisher Information Matrix rescales gradients to account for the geometry of the policy distribution. Directions that change the policy more receive smaller steps.

### Advantages

- Reduces variance and stabilizes learning
- Takes into account the curvature of the policy space
- More principled step sizes
- **Trade-off**: Computational cost (inverting a matrix)

---

## Advantage Function & Generalized Advantage Estimation (GAE)

### The Advantage Function

$$A(s,a) = Q(s,a) - V(s)$$

The advantage measures how much better an action is compared to the baseline value.

> [!formula]
> **TD Advantage**:
> $$\hat{A}_t = r_t + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t)$$
> Also called the **temporal difference error** or **1-step advantage**.

### Generalized Advantage Estimation (GAE)

A single TD step is biased. We can take multiple steps:

$$\hat{A}_t^{(1)} = \delta_t = r_t + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t)$$

$$\hat{A}_t^{(2)} = \delta_t + \gamma \delta_{t+1}$$

$$\hat{A}_t^{(\infty)} = \sum_{l=0}^\infty \gamma^l \delta_{t+l} = r_t + \gamma r_{t+1} + \cdots - \hat{V}(s_t)$$
(This is the Monte Carlo return minus the baseline.)

**GAE** smooths between these:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1-\lambda) \sum_{l=1}^\infty \lambda^{l-1} \hat{A}_t^{(l)}$$

> [!key point]
> $\lambda$ controls the bias-variance tradeoff:
> - $\lambda = 0$: Single TD step (low variance, high bias)
> - $\lambda = 1$: Monte Carlo (high variance, unbiased)
> - $\lambda \in (0,1)$: Interpolation

---

## Trust Region Methods (Overview)

While not detailed here, **Trust Region Policy Optimization (TRPO)** and **Proximal Policy Optimization (PPO)** are important advanced methods built on PGT and actor-critic.

> [!tip]
> **Key idea**: Constrain policy updates to a "trust region" where the approximation of $J(\theta)$ is accurate, preventing large destructive policy changes.

---

## The Policy Search Landscape

```
                    RL Methods
                        |
        ________________|________________
       |                                 |
   Value-Based                    Policy-Based
       |                                 |
   Q-learning          Policy Gradient Methods
   SARSA              /      |          |      \
   MC                /       |          |       \
                 REINFORCE  PGT    Actor-Critic  DPG
                     |       |          |         |
                   (v1)     (v2)    (w/baseline) (deterministic)
                                      /  |  \
                                  1-step GAE ∞-step
```

---

## When to Use Policy-Based Methods

Policy search methods are typically preferred when:

1. **Continuous action spaces**: Easy to parameterize continuous policies; hard to handle with Q-learning
2. **Stochastic policies needed**: Exploration naturally built in
3. **Prior knowledge about policy structure**: Can encode domain knowledge in policy architecture
4. **Small policy updates required**: For physical systems (robots) that can't handle sudden policy changes
5. **Deterministic optimal policies**: Use DPG for true greedy policies

---

## Summary & Key Takeaways

> [!summary]
> **Core Contributions of This Lecture**:
> 
> 1. **Policy Gradient Theorem**: Replaces hard-to-estimate returns with $Q$-function expectations, enabling actor-critic methods
> 
> 2. **Actor-Critic**: Combines policy updates (actor) with value function learning (critic) to reduce variance while maintaining theoretical grounding
> 
> 3. **Deterministic Policy Gradients**: Enables off-policy learning of deterministic policies without importance sampling weights
> 
> 4. **Advantage Functions & GAE**: Provides a principled way to interpolate between biased TD and unbiased MC advantage estimates
> 
> 5. **Natural Policy Gradient**: Incorporates policy space geometry for more stable, principled updates

---

## New Concepts to Explore

The following concepts are introduced but require deeper study:

- [[Deterministic Policy Gradient]] - Off-policy learning of deterministic policies
- [[Natural Policy Gradient]] - Fisher Information Matrix and policy space geometry
- [[Advantage Function]] - Baseline-corrected policy updates
- [[Generalized Advantage Estimation]] - Bias-variance tradeoff in TD advantage estimates
- [[Trust Region Policy Optimization]] (TRPO) - Constrained policy updates
- [[Proximal Policy Optimization]] (PPO) - Practical approximation to TRPO
- [[Compatible Function Approximation]] - Conditions for unbiased critic in actor-critic
- [[Soft Actor-Critic]] (SAC) - Maximum entropy RL with deterministic policies

---

## References

- Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. NIPS.
- Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmüller, M. (2014). *Deterministic Policy Gradient Algorithms*. ICML.
- Peters, J., & Schaal, S. (2008). *Reinforcement Learning of Motor Skills with Policy Search*. Handbook of Robotics.
- Schulman, G., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR.
