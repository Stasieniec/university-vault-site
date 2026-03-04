---
type: concept
aliases: [DPG, Off-policy Deterministic Policy Gradient]
course: [RL]
tags: [policy-gradient, off-policy, continuous-control, actor-critic]
status: complete
---

# Deterministic Policy Gradient (DPG)

## Definition

**Deterministic Policy Gradient** is a policy gradient method that learns a **deterministic policy** (outputting a single action per state) using **off-policy data**. Unlike standard stochastic policy gradients, DPG does not require importance sampling weights, making it more sample-efficient.

---

## Intuition

**Problem**: Standard policy gradients learn stochastic policies, but:
- Stochastic policies have inherent exploration but are sample-inefficient
- For many continuous control tasks, the optimal policy is deterministic (e.g., apply maximum torque)

**Solution**: Use a deterministic actor policy while learning from data collected by a stochastic behavior policy. The key insight is that the objective function depends only on states (not sampled actions), so importance weights are unnecessary.

---

## Mathematical Formulation

### Off-Policy Objective

Instead of maximizing return under the policy being learned, optimize:

$$J(\pi_\theta) = \int_{\mathcal{S}} \rho^\beta(s) Q^\pi(s, \pi_\theta(s)) \, ds$$

where:
- $\rho^\beta(s)$ = state visitation distribution under behavior policy $\beta$
- $Q^\pi(s,a)$ = learned action-value function
- $\pi_\theta(s)$ = **deterministic** actor policy (outputs single action, not distribution)

**Crucial**: The integral is only over states, not actions. This avoids the need for importance weights on actions.

### The DPG Formula

Taking the gradient:
$$\nabla_\theta Q^\pi(s,a) = \nabla_a Q^\pi(s,a) \nabla_\theta \pi_\theta(s) \quad \text{(chain rule)}$$

At $a = \pi_\theta(s)$:

$$\boxed{\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s \sim \rho^\beta} [\nabla_a Q^\pi(s,a)|_{a=\pi_\theta(s)} \cdot \nabla_\theta \pi_\theta(s)]}$$

### Practical Update Rule

1. **Actor update** (DPG):
   $$\theta_{t+1} = \theta_t + \alpha \nabla_a Q(s_t, a)|_{a=\mu_\theta(s_t)} \nabla_\theta \mu_\theta(s_t)$$

2. **Critic update** (Q-learning):
   $$w_{t+1} = w_t + \beta (r_t + \gamma \max_{a'} Q(s_{t+1}, a', w) - Q(s_t, a_t, w)) \nabla_w Q(s_t, a_t, w)$$

---

## Key Properties

### 1. Off-Policy Learning
- Learns from data collected by **any behavior policy** $\beta$
- Behavior policy typically: $\beta(a|s) = \pi_\theta(a|s) + \mathcal{N}(0, \sigma)$ (deterministic policy + noise)
- Much more sample-efficient than on-policy methods

### 2. No Importance Sampling Weights
- Standard off-policy methods need: $\frac{p(a|s)}{q(a|s)}$ (importance weights)
- This can have very high variance (variance explosion with continuous actions)
- DPG avoids this entirely

### 3. Deterministic Policy
- Can learn policies that always take the same action in a given state
- Matches the greedy solution: $\pi(s) = \arg\max_a Q(s,a)$
- Often optimal for well-shaped reward functions

---

## Variants & Extensions

### Deep Deterministic Policy Gradient (DDPG)
Applied DPG with deep neural networks:
- Actor network: $\mu_\theta(s) \to a$
- Critic network: $Q_w(s,a) \to \mathbb{R}$
- Target networks for stability
- Experience replay buffer

### Soft Actor-Critic (SAC)
- Extends DPG to maximum entropy RL
- Maintains stochastic policy (for exploration) but optimizes deterministic actor
- Entropy regularization: $J(\pi) = \mathbb{E}[Q(s,a)] + \alpha H(\pi)$

### TD3 (Twin Delayed DDPG)
- Addresses overestimation bias in Q-learning
- Uses two critic networks (twin Q-networks)
- Delayed policy updates

---

## Comparison: Stochastic vs. Deterministic

| Aspect | Stochastic PG | DPG |
|--------|---------------|-----|
| **Policy** | $\pi(a\|s)$ distribution | $\mu(s)$ deterministic |
| **Exploration** | Built-in (entropy) | Behavior policy adds noise |
| **Off-policy** | Requires importance weights | No weights needed |
| **Gradient** | $\mathbb{E}[\nabla \log \pi Q]$ | $\mathbb{E}[\nabla_a Q \nabla \mu]$ |
| **Sample efficiency** | Lower | Higher |
| **Optimal policy** | Can be stochastic | Deterministic |

---

## Advantages

✓ **High sample efficiency** - Off-policy learning with continuous actions  
✓ **No importance weights** - Avoids variance explosion  
✓ **Deterministic optimal policies** - Matches greedy solution  
✓ **Continuous action spaces** - Natural for continuous control  

---

## Disadvantages

✗ Requires learning both actor and critic (two networks)  
✗ Can suffer from Q-function overestimation (addressed by TD3)  
✗ Needs careful tuning of target networks and learning rates  
✗ Less exploration than stochastic policies (relies on behavior policy)  

---

## Connections

- **Related to**: [[Q-Learning]], [[Actor-Critic]], [[Policy Gradient Methods]]
- **Extends**: [[Policy Gradient Theorem]] from stochastic to deterministic
- **Foundation for**: [[Soft Actor-Critic (SAC)]], [[DDPG]], [[TD3]]
- **Appears in**: [[Deep Reinforcement Learning]]

---

## Key References

1. Silver, D., et al. (2014). *Deterministic Policy Gradient Algorithms*. ICML.
2. Lillicrap, T., et al. (2015). *Continuous Control with Deep Reinforcement Learning* (DDPG). ICLR.
3. Fujimoto, S., et al. (2018). *Addressing Function Approximation Error in Actor-Critic Methods* (TD3). ICML.

