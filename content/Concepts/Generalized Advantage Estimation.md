---
type: concept
aliases: [GAE, Generalized Advantage, λ-advantage]
course: [RL]
tags: [advantage-function, temporal-difference, policy-gradient, bias-variance]
status: complete
---

# Generalized Advantage Estimation (GAE)

## Definition

**Generalized Advantage Estimation** is a method for estimating the advantage function that **interpolates between single-step temporal difference** (low bias, low variance) **and Monte Carlo** (unbiased, high variance) advantage estimates using a single hyperparameter $\lambda$.

---

## Problem It Solves

### The Bias-Variance Tradeoff

Different advantage estimators have different properties:

1. **1-step TD advantage**: $\hat{A}_t^{(1)} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
   - ✓ Low variance
   - ✗ High bias (only one step)

2. **n-step advantage**: $\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}$
   - Medium bias and variance

3. **Monte Carlo advantage**: $\hat{A}_t^{(\infty)} = G_t - V(s_t)$
   - ✓ Unbiased
   - ✗ High variance (depends on entire trajectory)

**GAE's solution**: Use a **weighted combination** of all these, controlled by a single $\lambda$ parameter.

---

## Mathematical Formulation

### GAE Definition

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1-\lambda) \sum_{l=1}^\infty \lambda^{l-1} \hat{A}_t^{(l)}$$

where $\hat{A}_t^{(l)}$ is the $l$-step advantage.

### Equivalent Form (TD Residual)

GAE can be expressed as an exponential sum of temporal difference errors:

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the **TD residual** (1-step advantage).

### Recursive Computation

For efficient computation, compute advantages **backwards** through the trajectory:

$$A_t = \delta_t + (\gamma \lambda) A_{t+1}$$

with $A_T = 0$ at the terminal state.

**Pseudocode**:
```
gae = 0
for t in reversed(range(T)):
    delta_t = rewards[t] + gamma * V(s_{t+1}) - V(s_t)
    gae = delta_t + gamma * lambda * gae
    advantages[t] = gae
```

---

## Hyperparameter $\lambda$

The parameter $\lambda \in [0,1]$ controls the bias-variance tradeoff:

### $\lambda = 0$ (TD)
$$\hat{A}_t^{\text{GAE}(0)} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
- Only uses immediate reward and next state value
- **Bias**: High (only 1-step lookahead)
- **Variance**: Low

### $\lambda = 1$ (Monte Carlo)
$$\hat{A}_t^{\text{GAE}(1)} = G_t - V(s_t)$$
- Uses entire trajectory (full return $G_t$)
- **Bias**: Zero (unbiased)
- **Variance**: High (depends on full episode)

### $\lambda \in (0,1)$ (Interpolation)
Smooth tradeoff between bias and variance.

**Common choices**:
- $\lambda = 0.95$ (slightly favor MC, good for most domains)
- $\lambda = 0.99$ (more MC-like, less bias)
- $\lambda = 0.97$ (balanced)

---

## Intuition: Why the Weighted Combination?

The weights $(1-\lambda) \lambda^{l-1}$ give exponentially decaying importance to longer TD chains:

- 1-step: weight $(1-\lambda)$
- 2-step: weight $(1-\lambda) \lambda$
- 3-step: weight $(1-\lambda) \lambda^2$
- ...

**Lower $\lambda$**: Weight concentrated on short steps (low variance)  
**Higher $\lambda$**: Weight spread across longer steps (less bias)

---

## Properties

### 1. Exponential Weighting
The decay factor $(\gamma \lambda)$ ensures:
- Distant future terms contribute exponentially less
- Numerical stability (sum converges)

### 2. Consistency
- At $\lambda=0$: Consistent with 1-step TD
- At $\lambda=1$: Consistent with MC return
- At $\lambda \in (0,1)$: Smooth interpolation

### 3. Causality
Each $\delta_t$ depends only on the current transition $(s_t, a_t, r_t, s_{t+1})$:
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

This respects causality: action at $t$ only affects future rewards.

### 4. Off-Policy Extension
GAE can be extended for off-policy learning using importance sampling:
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$\rho_t = \min\left(1, \frac{\pi(a_t|s_t)}{\beta(a_t|s_t)}\right) \quad \text{(clipped importance ratio)}$$
$$\text{(off-policy } A_t) = \rho_t \delta_t + (\gamma \lambda \rho_{t+1}) A_{t+1}$$

---

## Comparison with Alternatives

| Estimator | Formula | $\lambda$ | Bias | Variance |
|-----------|---------|----------|------|----------|
| **TD(0)** | $\delta_t$ | 0 | High | Low |
| **TD($\lambda$)** | Trace-based | $\lambda$ | Medium | Medium |
| **GAE($\lambda$)** | $\sum (\gamma\lambda)^l \delta_{t+l}$ | $\lambda$ | Tunable | Tunable |
| **MC** | $G_t - V(s_t)$ | 1 | None | High |

---

## Algorithm: A3C with GAE

```python
def compute_gae_advantages(trajectory, values, gamma=0.99, lambda=0.95):
    """Compute GAE advantages for a trajectory."""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(trajectory))):
        state, action, reward, next_state, done = trajectory[t]
        value = values[state]
        next_value = 0 if done else values[next_state]
        
        # TD residual
        delta = reward + gamma * next_value - value
        
        # Exponential sum: A_t = delta_t + (gamma * lambda) * A_{t+1}
        gae = delta + (gamma * lambda) * gae
        advantages.insert(0, gae)
    
    return advantages

def policy_update(advantages, log_probs):
    """Update policy using GAE advantages."""
    policy_loss = -(torch.stack(log_probs) * advantages).mean()
    return policy_loss
```

---

## Empirical Performance

GAE is widely used because:

✓ **Practical**: Single hyperparameter controls tradeoff  
✓ **Efficient**: Backward pass is $O(T)$ complexity  
✓ **Flexible**: Works with different value function approximators  
✓ **Empirically strong**: Consistently outperforms pure TD or MC  

**Typical results**: With $\lambda \approx 0.95$, GAE achieves:
- Lower sample complexity than MC
- Lower bias than TD
- Faster convergence than either alone

---

## Related Hyperparameters

When using GAE, also tune:

### $\gamma$ (Discount Factor)
- Affects TD residual magnitude
- Typically: $\gamma = 0.99$ for long horizons

### Value Function Learning Rate
- GAE quality depends on accurate $V(s)$ estimates
- Needs sufficient critic updates per policy update

### Entropy Coefficient (for entropy regularization)
- Can be paired with GAE in policy methods
- Encourages exploration

---

## Connections

- **Extends**: [[Advantage Function]], [[Temporal Difference Learning]]
- **Used in**: [[A3C]], [[A2C]], [[PPO]], [[TRPO]], [[SAC]]
- **Related to**: [[Bias-Variance Trade-off]], [[Value Function]]
- **Appears in**: [[Actor-Critic]], [[Policy Gradient Methods]]

---

## Key References

1. **Schulman, G., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P.** (2015). 
   *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR.
   - **Original paper introducing GAE**

2. **Mnih, V., et al.** (2016). 
   *Asynchronous Methods for Deep Reinforcement Learning* (A3C). ICML.
   - Uses GAE in asynchronous policy gradient method

3. **Schulman, G., et al.** (2017). 
   *Proximal Policy Optimization Algorithms* (PPO). Arxiv.
   - Standard method using GAE for advantage estimation

