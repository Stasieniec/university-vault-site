---
type: lecture
course: RL
week: 1
lecture: 2
book_sections: ["Ch 2.5", "Ch 3.4-3.8", "Ch 4"]
topics:
  - "[[Dynamic Programming]]"
  - "[[Policy Evaluation]]"
  - "[[Policy Improvement]]"
  - "[[Policy Iteration]]"
  - "[[Value Iteration]]"
  - "[[Bellman Equation]]"
  - "[[Bellman Optimality Equation]]"
  - "[[Generalized Policy Iteration]]"
status: complete
---

# RL Lecture 2: Dynamic Programming

Dynamic Programming (DP) refers to a collection of algorithms that can be used to compute optimal policies given a **perfect model** of the environment as a [[Markov Decision Process]] (MDP). While classical DP is limited by the requirement of a model and computational cost, it provides the theoretical foundation for almost all other RL methods, which can be viewed as approximations of DP.

> [!definition] Key Assumption
> DP assumes the environment is a **finite MDP** with known dynamics $p(s', r | s, a)$.

---

## 1. Unified Notation & Recap

To handle both episodic and continuing tasks simultaneously, we use a single notation for [[Returns]]:
- For episodic tasks, we consider termination as entering a **special absorbing state** that transitions only to itself with reward 0.
- This allows us to use the infinite sum notation with the possibility of $\gamma = 1$ if all episodes eventually terminate.

> [!formula] Discounted Return
> $$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$
> where:
> - $G_t$ — the total discounted return from time $t$.
> - $\gamma \in [0, 1]$ — [[Discount Factor]].
> - $R_{t+k+1}$ — reward at step $t+k+1$.

---

## 2. Policy Evaluation (Prediction)

**Policy Evaluation** is the computation of the state-value function $v_\pi$ for an arbitrary policy $\pi$.

> [!formula] Bellman Equation for $v_\pi$
> $$v_\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$
> **Term-by-term explanation:**
> - $v_\pi(s)$: Value of state $s$ under policy $\pi$.
> - $\sum_a \pi(a|s)$: Expectation over actions chosen by policy $\pi$.
> - $\sum_{s', r} p(s', r | s, a)$: Expectation over next states $s'$ and rewards $r$ given action $a$ (dynamics).
> - $[r + \gamma v_\pi(s')]$: Immediate reward plus discounted value of the next state.

### Iterative Policy Evaluation
If the dynamics are known, the Bellman equation forms a system of $|S|$ linear equations. We solve this iteratively.

> [!formula] Iterative Update Rule
> $$v_{k+1}(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_k(s')]$$
> This is an **expected update**. The sequence $\{v_k\}$ converges to $v_\pi$ as $k \to \infty$ (under $\gamma < 1$ or guaranteed termination).

#### Algorithm: Iterative Policy Evaluation
```pseudo
Algorithm: Iterative Policy Evaluation, for estimating V ≈ v_π
─────────────────────────────────────────────────────────────
Input: π (policy to be evaluated)
Parameter: θ > 0 (small threshold determining accuracy)
Initialize V(s) arbitrarily (e.g., 0), for all s ∈ S; V(terminal) = 0

Loop:
  Δ ← 0
  For each s ∈ S:
    v ← V(s)
    V(s) ← Σ_a π(a|s) Σ_{s',r} p(s', r|s, a) [r + γ V(s')]
    Δ ← max(Δ, |v - V(s)|)
until Δ < θ
```

> [!intuition] Backup Diagram for $v_\pi$
> ```ascii
>        (s)          <-- State being updated
>       / | \
>      o  o  o        <-- Actions (a) chosen by policy π
>     / \
>   (s') (s')         <-- Possible next states (s') from dynamics p
> ```
> The value flows back from the leaf nodes ($s'$) through the actions to the root ($s$).

---

## 3. Policy Improvement

Once we have $v_\pi$, we want to find a better policy $\pi'$.

> [!definition] Policy Improvement Theorem
> Let $\pi$ and $\pi'$ be any pair of deterministic policies such that, for all $s \in S$:
> $$q_\pi(s, \pi'(s)) \geq v_\pi(s)$$
> Then $\pi'$ must be as good as, or better than, $\pi$:
> $$v_{\pi'}(s) \geq v_\pi(s) \quad \forall s \in S$$

### Proof Sketch
We start with $v_\pi(s) \leq q_\pi(s, \pi'(s))$ and repeatedly expand the right side:
1. $v_\pi(s) \leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s]$
2. Apply the inequality again: $v_\pi(s) \leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1}, \pi'(S_{t+1})) | S_t = s]$
3. Continue expanding the expectation of rewards.
4. Eventually, the sum reaches $v_{\pi'}(s)$.

### Greedy Policy Improvement
A natural candidate for $\pi'$ is the **greedy policy** with respect to $v_\pi$:
$$\pi'(s) = \text{arg max}_a q_\pi(s, a) = \text{arg max}_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

---

## 4. Policy Iteration

**Policy Iteration** is the alternating sequence of Evaluation and Improvement.

> [!formula] The PI Loop
> $$\pi_0 \xrightarrow{E} v_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_1} \xrightarrow{I} \pi_2 \xrightarrow{E} \dots \xrightarrow{I} \pi_* \xrightarrow{E} v_*$$
> Because a finite MDP has only a finite number of deterministic policies, this process **must converge** to an optimal policy in finite steps.

#### Algorithm: Policy Iteration
```pseudo
Algorithm: Policy Iteration (for finding π ≈ π*)
─────────────────────────────────────────────────────────────
1. Initialization
   V(s) ∈ ℝ and π(s) ∈ A(s) arbitrarily for all s ∈ S

2. Policy Evaluation
   Loop:
     Δ ← 0
     For each s ∈ S:
       v ← V(s)
       V(s) ← Σ_{s',r} p(s', r|s, π(s)) [r + γ V(s')]
       Δ ← max(Δ, |v - V(s)|)
   until Δ < θ

3. Policy Improvement
   policy-stable ← true
   For each s ∈ S:
     old-action ← π(s)
     π(s) ← arg max_a Σ_{s',r} p(s', r|s, a) [r + γ V(s')]
     If old-action ≠ π(s), then policy-stable ← false
   If policy-stable, then stop; else go to step 2
```

---

## 5. Value Iteration

One drawback of Policy Iteration is that each evaluation step itself requires an iterative process. **Value Iteration** combines evaluation and improvement into a single sweep.

> [!formula] Value Iteration Update Rule
> $$v_{k+1}(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_k(s')]$$
> This turns the **Bellman Optimality Equation** into an update rule.

#### Algorithm: Value Iteration
```pseudo
Algorithm: Value Iteration, for estimating π ≈ π*
─────────────────────────────────────────────────────────────
Initialize V arbitrarily (e.g., V(s) = 0), V(terminal) = 0

Loop:
  Δ ← 0
  For each s ∈ S:
    v ← V(s)
    V(s) ← max_a Σ_{s',r} p(s', r|s, a) [r + γ V(s')]
    Δ ← max(Δ, |v - V(s)|)
until Δ < θ

Output a deterministic policy π, such that
π(s) = arg max_a Σ_{s',r} p(s', r|s, a) [r + γ V(s')]
```

> [!intuition] Relation to Policy Iteration
> Value Iteration is equivalent to Policy Iteration with only **one sweep** of policy evaluation between improvement steps.

---

## 6. Generalized Policy Iteration (GPI)

GPI is the general framework describing any interaction of policy evaluation and policy improvement.

- **Evaluation**: Makes the value function consistent with the current policy.
- **Improvement**: Makes the policy greedy with respect to the current value function.

> [!intuition] The GPI Diagram
> Imagine two lines (or manifolds) in the value-policy space:
> 1. $v = v_\pi$: where values are consistent with the policy.
> 2. $\pi = \text{greedy}(v)$: where the policy is optimal for the values.
> Evaluation pulls us toward line 1; Improvement pulls us toward line 2. They compete and eventually intersect at $(v_*, \pi_*)$.
>
> ```mermaid
> graph TD
>     E[Policy Evaluation] --> |Estimate V| I[Policy Improvement]
>     I --> |Greedy Policy| E
>     style E fill:#f9f,stroke:#333
>     style I fill:#ccf,stroke:#333
> ```

---

## 7. Asynchronous Dynamic Programming

A major drawback of classical DP is that it involves "sweeps" over the entire state set (updates all states). **Asynchronous DP** updates states in any order.

- **Key idea**: Use whatever values are available.
- **Benefits**: Can focus computation on "important" or frequently visited states. No need to wait for a full sweep to finish.
- **Requirement**: Must continue to update all states (none can be ignored forever) for convergence to $v_*$.

---

## 8. Summary of Bellman Equations

| Name | Formula |
| :--- | :--- |
| **Bellman Expectation ($v_\pi$)** | $v_\pi(s) = \sum_a \pi(a\|s) \sum_{s', r} p(s', r \| s, a) [r + \gamma v_\pi(s')]$ |
| **Bellman Expectation ($q_\pi$)** | $q_\pi(s, a) = \sum_{s', r} p(s', r \| s, a) [r + \gamma \sum_{a'} \pi(a'\|s') q_\pi(s', a')]$ |
| **Bellman Optimality ($v_*$)** | $v_*(s) = \max_a \sum_{s', r} p(s', r \| s, a) [r + \gamma v_*(s')]$ |
| **Bellman Optimality ($q_*$)** | $q_*(s, a) = \sum_{s', r} p(s', r \| s, a) [r + \gamma \max_{a'} q_*(s', a')]$ |

---

## 9. Examples from Lecture

### 9.1 Gridworld (Iterative Policy Evaluation)
A 4x4 grid where transitions have reward -1 until termination.
- Under a **random policy**, the values represent the negative expected number of steps to the exit.
- After a single policy improvement step, the greedy policy already becomes optimal for this simple grid.

### 9.2 Transition Graph Stochasticity
> [!example] Recycling Robot
> Arcs are labeled with $(p, r)$: probability $p$ and reward $r$. 
> - If `low` battery, action `search` might transition to `low` (prob $\beta$) or result in "rescue" (transition to `high`, prob $1-\beta$, reward -3).
> - This complexity is handled inherently by the expectation over $p(s', r | s, a)$.

> [!tip] Big Picture of Policy Learning
> 1. **Model-based**: Learn dynamics $p(s', r|s, a)$ then plan (DP).
> 2. **Model-free Value-based**: Learn $V(s)$ or $Q(s, a)$ directly (TD, Q-learning).
> 3. **Model-free Policy-based**: Directly optimize $\pi(a|s)$ (Policy Gradient).

---
**Related Concepts:**
- [[Markov Decision Process]]
- [[Bellman Equation]]
- [[Returns]]
- [[Policy Iteration]]
- [[Value Iteration]]
- [[Generalized Policy Iteration]]
- [[Optimality and Approximation]]

**Book Reference:** [[Sutton & Barto]] Ch 3.4-3.8, Ch 4.
