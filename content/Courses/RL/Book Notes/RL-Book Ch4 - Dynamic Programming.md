---
type: book-chapter
course: RL
book: "Reinforcement Learning: An Introduction (2nd ed.)"
chapter: 4
sections: ["4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8"]
topics:
  - "[[Dynamic Programming]]"
  - "[[Policy Evaluation]]"
  - "[[Policy Improvement]]"
  - "[[Policy Iteration]]"
  - "[[Value Iteration]]"
  - "[[Generalized Policy Iteration]]"
  - "[[Bellman Equation]]"
status: complete
---

# Chapter 4: Dynamic Programming

## Overview
**Dynamic Programming (DP)** refers to a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as a [[Markov Decision Process]] (MDP). 

> [!definition] Dynamic Programming
> A method for solving complex problems by breaking them down into simpler subproblems. In RL, it uses [[Value Function]]s to search for [[Optimal Policy|optimal policies]].

While classical DP is limited by the assumption of a perfect model and high computational expense, it provides the foundation for most RL methods. Most RL methods can be seen as attempts to achieve the same effect as DP but with less computation and without a perfect model.

---

## 4.1 Policy Evaluation (Prediction)
**Policy Evaluation** is the process of computing the state-value function $v_\pi$ for an arbitrary policy $\pi$.

### The Bellman Equation for $v_\pi$
For all $s \in \mathcal{S}$:
> [!formula] Bellman Equation for $v_\pi$
> $$v_\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

If the environment's dynamics are known, this is a system of $|\mathcal{S}|$ simultaneous linear equations. 

### Iterative Policy Evaluation
DP solves this iteratively. Starting with an initial approximation $v_0$ (e.g., all zeros), each successive approximation is obtained using the Bellman equation as an update rule:

$$v_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_k(s')]$$

> [!intuition] Expected Update
> This operation is called an **expected update** because it is based on an expectation over all possible next states rather than a sample.

### Algorithm: Iterative Policy Evaluation
```python
# Iterative Policy Evaluation, for estimating V ≈ v_π
# Input: π (policy), θ (threshold > 0)
# Initialize V(s) arbitrarily, V(terminal) = 0

Loop:
    Δ ← 0
    Loop for each s in S:
        v ← V(s)
        V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
        Δ ← max(Δ, |v - V(s)|)
until Δ < θ
```

> [!example] 4x4 Gridworld
> - **States**: 14 non-terminal, 1 terminal (shown in two corners).
> - **Actions**: Up, Down, Left, Right (deterministic).
> - **Reward**: -1 on all transitions.
> - **Policy**: Equiprobable random.
> - **Result**: Policy evaluation converges to the expected number of steps to the goal (negated).

---

## 4.2 Policy Improvement
Once we have $v_\pi$, we want to know if we can improve the policy. Suppose we consider changing the policy to take action $a$ in state $s$, and thereafter following $\pi$. The value of this behavior is:
$$q_\pi(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

### Policy Improvement Theorem
> [!formula] Policy Improvement Theorem
> Let $\pi$ and $\pi'$ be any pair of deterministic policies such that, for all $s \in \mathcal{S}$:
> $$q_\pi(s, \pi'(s)) \geq v_\pi(s)$$
> Then the policy $\pi'$ must be as good as, or better than, $\pi$:
> $$v_{\pi'}(s) \geq v_\pi(s)$$

### Greedy Policy Improvement
We can create a new policy $\pi'$ that is **greedy** with respect to $v_\pi$:
$$\pi'(s) = \arg\max_{a} q_\pi(s, a) = \arg\max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$
This process is called **[[Policy Improvement]]**. If the new policy is no better than the old one, then both must be optimal ($\pi = \pi^*$).

---

## 4.3 Policy Iteration
By alternating evaluation and improvement, we obtain a sequence of monotonically improving policies:
$$\pi_0 \xrightarrow{E} v_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_1} \xrightarrow{I} \pi_2 \xrightarrow{E} \dots \xrightarrow{I} \pi^* \xrightarrow{E} v^*$$

### Algorithm: Policy Iteration
```python
# Policy Iteration (using iterative policy evaluation)
# 1. Initialization: V(s) and π(s) arbitrarily

# 2. Policy Evaluation
Loop:
    Δ ← 0
    Loop for each s in S:
        v ← V(s)
        V(s) ← Σ_{s',r} p(s',r|s,π(s)) [r + γ V(s')]
        Δ ← max(Δ, |v - V(s)|)
until Δ < θ

# 3. Policy Improvement
policy_stable ← true
Loop for each s in S:
    old_action ← π(s)
    π(s) ← argmax_a Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
    if old_action ≠ π(s): policy_stable ← false
if policy_stable: return V, π; else go to 2
```

> [!example] Jack's Car Rental
> - **States**: Number of cars at two locations (max 20 each).
> - **Actions**: Move up to 5 cars overnight ($2/car).
> - **Reward**: $10 per rental.
> - **Dynamics**: Poisson distributions for requests and returns.
> - **Observation**: Policy iteration finds the optimal strategy in just a few sweeps.

---

## 4.4 Value Iteration
One drawback of policy iteration is that it requires multiple sweeps for each evaluation step. **[[Value Iteration]]** truncates policy evaluation to exactly one sweep.

### The Update Rule
Value iteration turns the [[Bellman Optimality Equation]] into an update rule:
> [!formula] Value Iteration Update
> $$v_{k+1}(s) = \max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma v_k(s')]$$

### Algorithm: Value Iteration
```python
# Value Iteration
# Initialize V(s) arbitrarily, V(terminal) = 0

Loop:
    Δ ← 0
    Loop for each s in S:
        v ← V(s)
        V(s) ← max_a Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
        Δ ← max(Δ, |v - V(s)|)
until Δ < θ

# Output deterministic policy π
π(s) = argmax_a Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
```

> [!example] Gambler's Problem
> - **Goal**: Reach $100.
> - **Action**: Stake $a \in \{0, \dots, \min(s, 100-s)\}$.
> - **Reward**: +1 if goal is reached, 0 otherwise.
> - **Result**: The optimal policy shows a "curious form" with jumps at 25, 50, and 75 due to the binary nature of the goal.

---

## 4.5 Asynchronous Dynamic Programming
Synchronous DP requires sweeping the entire state set, which is impossible for large state spaces (e.g., Backgammon with $10^{20}$ states).

**Asynchronous DP** algorithms update state values in any order, using whatever values are available.
- They do not require a global sweep.
- They must continue to update all states to guarantee convergence.
- They allow intermixing computation with real-time interaction (updating states as an agent visits them).

---

## 4.6 Generalized Policy Iteration (GPI)
**[[Generalized Policy Iteration]]** refers to the general idea of letting policy-evaluation and policy-improvement processes interact, regardless of granularity.

### The GPI Mechanism
- **Evaluation**: Movement toward making the value function consistent with the current policy.
- **Improvement**: Movement toward making the policy greedy with respect to the current value function.

> [!intuition] Competition and Cooperation
> Evaluation and improvement compete (one changes the basis for the other) but cooperate to find a joint solution where the [[Bellman Equation]] is satisfied.

**Diagram Representation (Literal Description):**
- Two lines in a 2D space: one representing $v = v_\pi$ (consistency) and another representing $\pi = \text{greedy}(v)$ (optimality).
- The processes drive the system toward these lines. The intersection is $(v^*, \pi^*)$.

---

## 4.7 Efficiency of Dynamic Programming
DP methods are polynomial in the number of states and actions ($n$ and $k$). This is exponentially faster than direct search in policy space ($k^n$ policies).

- **Curse of Dimensionality**: Large state spaces are a problem for the MDP itself, not specifically DP. 
- DP is generally more efficient than linear programming for very large state sets.

---

## 4.8 Summary
- **[[Dynamic Programming]]** solve MDPs using a perfect model.
- **[[Policy Evaluation]]**: Iteratively compute $v_\pi$.
- **[[Policy Improvement]]**: Make $\pi$ greedy w.r.t $v_\pi$.
- **[[Policy Iteration]]**: Alternate Evaluation and Improvement.
- **[[Value Iteration]]**: Combine Evaluation (1 sweep) and Improvement into one update.
- **[[Generalized Policy Iteration]] (GPI)**: The overarching framework where evaluation and improvement interact.
- **Bootstrapping**: DP updates estimates based on other estimates.

> [!tip] Key Takeaway
> All DP algorithms turn Bellman equations into assignment/update rules.
