---
type: book-chapter
course: RL
book: "Reinforcement Learning: An Introduction (2nd ed.)"
chapter: 13
sections: ["13.1", "13.2", "13.3", "13.4", "13.5", "13.6", "13.7", "13.8"]
topics:
  - "[[Policy Gradient Methods]]"
  - "[[Policy Gradient Theorem]]"
  - "[[REINFORCE]]"
  - "[[Actor-Critic]]"
status: complete
---

# RL-Book Ch13 - Policy Gradient Methods

## Overview
In this chapter, we transition from [[Action-Value Methods]] (which learn values and then derive a policy) to **Policy Gradient Methods**, which learn a parameterized [[Policy]] $\pi(a|s, \theta)$ that can select actions without consulting a value function. While a value function may still be used to learn the policy parameters, it is not required for action selection.

> [!definition] Policy Gradient Methods
> Methods that learn a parameterized policy $\pi(a|s, \theta)$ and update the parameters $\theta \in \mathbb{R}^d$ by approximating gradient ascent in a scalar performance measure $J(\theta)$:
> $$\theta_{t+1} = \theta_{t} + \alpha \widehat{\nabla J(\theta_t)}$$

## 13.1 Policy Approximation and its Advantages

A common parameterization for discrete action spaces is the **soft-max in action preferences**:
$$\pi(a|s, \theta) = \frac{e^{h(s, a, \theta)}}{\sum_b e^{h(s, b, \theta)}}$$
where $h(s, a, \theta)$ are numerical preferences (e.g., linear in features: $h(s, a, \theta) = \theta^\top x(s, a)$).

### Advantages over Action-Value Methods
1. **Convergence to Deterministic Policies**: Action-value methods with $\epsilon$-greedy always explore. Policy gradients can drive preferences of optimal actions infinitely higher, approaching a deterministic policy.
2. **Stochastic Optimal Policies**: In many problems (e.g., imperfect information games like Poker, or the [[Short Corridor with Switched Actions]]), the optimal policy is stochastic. Policy gradient methods can learn these specific probabilities naturally.
3. **Simpler Approximation**: The policy may be a simpler function to approximate than the value function.
4. **Injection of Prior Knowledge**: Parameterization allows specific domain knowledge about the policy's form to be encoded.
5. **Continuous Action Spaces**: Naturally handles infinite action sets by learning distribution statistics (e.g., mean and variance).

---

## 13.2 The [[Policy Gradient Theorem]]

The challenge in policy gradients is that performance $J(\theta)$ depends on both action selections and the state distribution, the latter of which is often unknown and affected by the policy in complex ways.

> [!formula] The Policy Gradient Theorem (Episodic)
> $$\nabla J(\theta) \propto \sum_s \mu(s) \sum_a q_\pi(s, a) \nabla \pi(a|s, \theta)$$
> where $\mu(s)$ is the on-policy distribution. Crucially, the gradient does **not** involve the derivative of the state distribution.

### Proof Sketch (Episodic Case)
1. Start with the gradient of the state-value function: $\nabla v_\pi(s) = \nabla \sum_a \pi(a|s) q_\pi(s, a)$.
2. Apply the product rule: $\nabla v_\pi(s) = \sum_a [\nabla \pi(a|s) q_\pi(s, a) + \pi(a|s) \nabla q_\pi(s, a)]$.
3. Expand $\nabla q_\pi(s, a)$ using the Bellman equation: $\nabla q_\pi(s, a) = \sum_{s'} p(s'|s, a) \nabla v_\pi(s')$.
4. Unroll the recurrence: $\nabla v_\pi(s) = \sum_a \nabla \pi(a|s) q_\pi(s, a) + \sum_a \pi(a|s) \sum_{s'} p(s'|s, a) \nabla v_\pi(s')$.
5. After repeated unrolling, we see the gradient is the sum over all states reachable from the start state, weighted by the probability of being in that state at any time step.

---

## 13.3 [[REINFORCE]]: Monte Carlo Policy Gradient

REINFORCE approximates the gradient using a single sample $A_t$ at time $t$. By noting that $\nabla \pi = \pi \frac{\nabla \pi}{\pi} = \pi \nabla \ln \pi$, we can express the gradient as an expectation.

### The Algorithm
The update rule is:
$$\theta_{t+1} = \theta_t + \alpha G_t \nabla \ln \pi(A_t | S_t, \theta_t)$$

> [!intuition] Eligibility Vector
> The vector $\nabla \ln \pi(A_t|S_t, \theta)$ is the direction in parameter space that most increases the probability of repeating action $A_t$. The update scales this by the return $G_t$.

**Pseudocode (REINFORCE):**
```python
Initialize policy parameter theta
Loop forever (for each episode):
    Generate an episode S0, A0, R1, ..., RT-1, AT-1, RT following pi(.|., theta)
    Loop for each step t = 0, 1, ..., T-1:
        G = sum_{k=t+1}^{T} R_k
        theta = theta + alpha * G * grad_ln_pi(At | St, theta)
```

---

## 13.4 REINFORCE with Baseline

To reduce the high variance of [[Monte Carlo Methods]], we subtract a baseline $b(s)$ from the return. $b(s)$ can be any function as long as it does not depend on action $a$.

> [!tip] 
> The most natural baseline is an estimate of the state value $\hat{v}(S_t, w)$.

**Update Rule:**
$$\theta_{t+1} = \theta_t + \alpha (G_t - b(S_t)) \nabla \ln \pi(A_t | S_t, \theta_t)$$

---

## 13.5 [[Actor-Critic]] Methods

While REINFORCE with baseline uses $\hat{v}(S_t)$ to reduce variance, it still uses the full return $G_t$, requiring the end of the episode. **Actor-Critic** methods use [[Bootstrapping]] via the **TD Error** $\delta_t$.

> [!definition] Actor and Critic
> - **Actor**: The learned policy $\pi(a|s, \theta)$.
> - **Critic**: The learned state-value function $\hat{v}(s, w)$.

The TD error assesses the action:
$$\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)$$

**Pseudocode (One-step Actor-Critic):**
```python
Initialize theta, w
Loop forever (for each episode):
    Initialize S
    Loop while S is not terminal:
        A ~ pi(.|S, theta)
        Take action A, observe S', R
        delta = R + gamma * v_hat(S', w) - v_hat(S, w)
        w = w + alpha_w * delta * grad_v_hat(S, w)
        theta = theta + alpha_theta * delta * grad_ln_pi(A|S, theta)
        S = S'
```

---

## 13.7 Continuous Actions Parameterization

For continuous action spaces, we learn the statistics of a probability distribution, typically a **Gaussian Policy**.

The mean $\mu(s, \theta)$ and standard deviation $\sigma(s, \theta)$ are parameterized:
$$\mu(s, \theta) = \theta_\mu^\top x_\mu(s)$$
$$\sigma(s, \theta) = \exp(\theta_\sigma^\top x_\sigma(s))$$

The policy is defined by the density:
$$\pi(a|s, \theta) = \frac{1}{\sigma(s, \theta)\sqrt{2\pi}} \exp\left(-\frac{(a-\mu(s, \theta))^2}{2\sigma(s, \theta)^2}\right)$$

The eligibility vectors are:
$$\nabla_{\theta_\mu} \ln \pi(a|s, \theta) = \frac{a - \mu(s, \theta)}{\sigma(s, \theta)^2} x_\mu(s)$$
$$\nabla_{\theta_\sigma} \ln \pi(a|s, \theta) = \left(\frac{(a - \mu(s, \theta))^2}{\sigma(s, \theta)^2} - 1\right) x_\sigma(s)$$

---

## Summary
- **Policy Gradient Methods** learn $\pi(a|s)$ directly via [[Stochastic Gradient Descent]].
- They handle **continuous actions** and **stochastic optimal policies** better than action-value methods.
- The **Policy Gradient Theorem** provides the theoretical foundation, removing dependence on the state distribution gradient.
- **REINFORCE** is the Monte Carlo version; **Actor-Critic** adds bootstrapping to reduce variance at the cost of some bias.
