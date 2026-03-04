---
type: concept
aliases: [NPG, Fisher Information Matrix, Natural Gradient]
course: [RL]
tags: [policy-gradient, optimization, fisher-information, geometry]
status: complete
---

# Natural Policy Gradient (NPG)

## Definition

**Natural Policy Gradient** is an improvement over vanilla policy gradient that accounts for the **geometry of the policy distribution space**. Instead of moving in the direction of steepest parameter change, it moves in the direction of steepest policy change (as measured by KL divergence).

---

## Motivation: The Problem with Vanilla Gradients

### Vanilla (Steepest Ascent in Parameter Space)

The standard policy gradient is:
$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta)$$

**Problem**: The step size depends on how much a parameter change affects the policy itself.

**Example**: For different policy parameterizations (e.g., $\log \sigma$ vs $\sigma^2$), the same parameter change can cause vastly different policy changes.

**Result**: Training is unstable, hyperparameter-sensitive, and slow to converge.

### Natural Gradient (Steepest Ascent in Policy Space)

Instead, move in the direction that achieves the steepest **policy improvement**:

$$\tilde{\nabla}_\theta J = F^{-1}(\theta) \nabla_\theta J(\theta)$$

where $F(\theta)$ is the **Fisher Information Matrix**:

$$F(\theta) = \mathbb{E}_{a \sim \pi_\theta(a|s)} [\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T]$$

---

## The Fisher Information Matrix

### Definition

The Fisher Information Matrix (or Fisher Information in the policy gradient context) is:

$$F(\theta) = \mathbb{E}_{s, a}[\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T]$$

**Dimensions**: $d \times d$ matrix where $d$ = number of policy parameters.

### Interpretation

- **Positive semi-definite**: Measures the **curvature** of the log-likelihood
- **Large eigenvalues**: Directions where small parameter changes cause large policy changes
- **Small eigenvalues**: Directions where large parameter changes cause small policy changes

### Connection to KL Divergence

The Fisher Information Matrix is related to the **Hessian of KL divergence**:

$$H_{\text{KL}}(\pi_\theta \| \pi_{\theta'}) \approx \frac{1}{2} (\theta - \theta')^T F(\theta) (\theta - \theta')$$

This is the local quadratic approximation to KL divergence near $\theta$.

---

## Natural Gradient Update Rule

### Standard Form

$$\theta_{t+1} = \theta_t + \alpha F^{-1}(\theta_t) \nabla_\theta J(\theta_t)$$

### Interpretation

- $\nabla_\theta J(\theta_t)$: Direction of steepest ascent in parameter space
- $F^{-1}(\theta_t)$: Rescaling matrix that converts parameter changes to policy changes
- **Effect**: Large steps in directions that barely change the policy, small steps in directions that dramatically change the policy

---

## Practical Advantages

### 1. Parameterization Invariance

The natural gradient is **invariant** to how you parameterize the policy. Whether you use:
- $\log \sigma$ vs $\sigma$ vs $\sigma^2$
- Different neural network architectures

The natural gradient produces the same **policy update** (though different parameter updates).

**Vanilla gradient**: Different for each parameterization  
**Natural gradient**: Same policy update regardless of parameterization

### 2. Better Convergence

- Converges faster (fewer iterations)
- More stable (less sensitive to hyperparameters)
- Larger effective step sizes

### 3. Automatic Step Size Scaling

The Fisher matrix automatically scales step sizes based on:
- How sensitive the policy is to parameter changes
- The variance in the gradient estimates

---

## Computational Considerations

### Computing the Fisher Matrix

**Full approach** (expensive):
1. Compute $\nabla_\theta \log \pi(a|s)$ for each sample
2. Compute outer product: $g_i g_i^T$ where $g_i = \nabla_\theta \log \pi_\theta(a_i|s_i)$
3. Average: $F = \mathbb{E}[g g^T]$
4. Invert: $F^{-1}$ (requires $d^3$ operations for $d$-dimensional parameters)

**Computational cost**: $O(d^3)$ which is prohibitive for high-dimensional policies (neural networks).

### Efficient Approximations

#### 1. Conjugate Gradient Method
Instead of explicitly computing $F^{-1}$, solve the linear system:
$$F x = \nabla_\theta J(\theta)$$

using **Conjugate Gradient** (20-50 iterations typically suffice).

**Cost**: $O(d^2)$ per update (much better than $O(d^3)$)

#### 2. Diagonal Fisher
Approximate $F$ as diagonal:
$$F \approx \text{diag}(\mathbb{E}[g \odot g])$$

where $\odot$ is element-wise multiplication.

**Cost**: $O(d)$ (trivial to invert)  
**Trade-off**: Less accurate but much faster

#### 3. K-FAC (Kronecker-Factored Approximation)
For neural networks, factor the Fisher as a Kronecker product of smaller matrices.

**Cost**: Between $O(d)$ and $O(d^3)$  
**Accuracy**: Good approximation for neural networks

---

## Natural Policy Gradient in Practice

### Common Implementation: Trust Region Methods

**TRPO** (Trust Region Policy Optimization) uses the natural gradient with a **trust region constraint**:

$$\max_\theta \mathbb{E}[\nabla_\theta \log \pi(a|s) Q(s,a)]$$
$$\text{subject to: } \mathbb{E}_s[D_{\text{KL}}(\pi_{\text{old}} || \pi_\theta)] \leq \delta$$

The trust region constraint ensures the policy doesn't change too much (limits KL divergence).

**Solution**: Use natural gradient with step size chosen to maintain the KL constraint.

### Pseudocode

```python
def natural_policy_gradient_update(trajectories, policy, value_fn, damping=0.01):
    """Compute natural policy gradient using conjugate gradient."""
    
    # Compute vanilla gradient
    grad = compute_policy_gradient(trajectories, policy, value_fn)
    
    # Compute Fisher-vector products via Hessian-vector products
    # (without explicitly forming the Fisher matrix)
    def fisher_vector_product(v):
        # Use autograd twice to compute H*v where H is Hessian of KL
        return compute_hessian_vector_product(policy, v) + damping * v
    
    # Solve: F x = grad using conjugate gradient
    natural_grad = conjugate_gradient(fisher_vector_product, grad, iterations=20)
    
    # Update
    theta_new = theta + alpha * natural_grad
    return theta_new
```

---

## Comparison: Vanilla vs Natural Gradient

| Aspect | Vanilla Gradient | Natural Gradient |
|--------|------------------|------------------|
| **Update** | $\nabla J(\theta)$ | $F^{-1} \nabla J(\theta)$ |
| **Convergence** | Slower | Faster |
| **Stability** | Parameter-dependent | Parameterization-invariant |
| **Computation** | $O(d)$ | $O(d)$ to $O(d^3)$ (depends on approximation) |
| **Hyperparameter sensitivity** | High | Lower |
| **Implementation** | Simple | More complex |

---

## Example: Gaussian Policy

### Vanilla Gradient

For a diagonal Gaussian policy with mean $\mu$ and std $\sigma$:

$$\nabla_\mu J = \mathbb{E}[\nabla_\mu \log \pi(a|s) Q(s,a)]$$
$$\nabla_\sigma J = \mathbb{E}[\nabla_\sigma \log \pi(a|s) Q(s,a)]$$

Update each parameter separately with same learning rate.

### Natural Gradient

The Fisher matrix (for Gaussian) has a natural block structure. The natural gradient:
- Scales $\mu$ updates by $1/\sigma^2$ (covariance is important!)
- Scales $\sigma$ updates by special factors

**Effect**: Automatically adjusts step sizes based on policy uncertainty.

---

## Variants & Extensions

### 1. Trust Region Policy Optimization (TRPO)
- Constrains KL divergence between old and new policy
- Uses natural gradient to respect the constraint
- Sample-efficient, but slower to compute

### 2. PPO (Proximal Policy Optimization)
- Simpler approximation to TRPO
- Uses clipped objective instead of explicit trust region
- Much easier to implement, nearly as effective

### 3. A-Opt (Fisher-Damped)
- Adds damping term: $F + \lambda I$ (regularization)
- Improves stability and condition number
- Commonly used in practice

---

## When to Use Natural Gradient

### ✓ Use when:
- Sample efficiency is critical
- Policy changes are large and destabilizing
- Working with trust region methods (TRPO)
- Parameter initialization is poor

### ✗ Don't use when:
- Computational cost is prohibitive (without approximations)
- Standard gradient descent works well (simpler is better)
- Data is very noisy (first-order methods already struggle)

---

## Connections

- **Foundation for**: [[Trust Region Policy Optimization (TRPO)]], [[PPO]]
- **Related to**: [[Policy Gradient Methods]], [[Fisher Information]]
- **Used in**: [[Actor-Critic]], [[Policy Gradient Theorem]]
- **Appears in**: [[Deep Reinforcement Learning]], [[Optimization]]

---

## Key References

1. **Kakade, S.** (2001). 
   *Natural Policy Gradients*. NIPS.
   - **Original paper on natural policy gradients**

2. **Schulman, G., Levine, S., Moritz, P., Jordan, M., & Abbeel, P.** (2015). 
   *Trust Region Policy Optimization*. ICML.
   - **TRPO: Uses natural gradients with trust regions**

3. **Schulman, G., et al.** (2017). 
   *Proximal Policy Optimization Algorithms*. Arxiv.
   - **PPO: Simpler approximation to natural gradient methods**

4. **Martens, J., & Grosse, R.** (2015). 
   *Optimizing Neural Networks with Kronecker-Factored Approximate Curvature*. ICML.
   - **K-FAC: Efficient Fisher approximation for neural networks**

