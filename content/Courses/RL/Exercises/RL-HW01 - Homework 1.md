---
type: exercise
course: RL
week: 1
source: "Homework 1 (Coding Assignment 1)"
concepts:
  - "[[Dynamic Programming]]"
  - "[[Policy Iteration]]"
  - "[[Value Iteration]]"
  - "[[Bellman Equation]]"
  - "[[Value Function]]"
  - "[[Policy]]"
status: complete
---

# RL-HW01: Homework 1 — Dynamic Programming

> [!tip] Exam Relevance
> Questions 2a-2f are classic exam-style problems on Bellman equations, policy iteration, and value iteration. Especially 2f (linear system form of Bellman equations) appears frequently.

## Question 1: Policy Iteration vs Value Iteration (2.0p)

**Q1:** In the lab you implemented value iteration and policy iteration. (a) For which algorithm do you expect a single iteration to run faster? (b) Which algorithm do you expect to take fewer total iterations?

### Solution

**(a) Single iteration faster: [[Value Iteration]]**
- In [[Value Iteration]], each iteration does **one sweep** through all states with a max operation: $V(s) \leftarrow \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')]$
- In [[Policy Iteration]], each iteration requires:
  1. **Policy evaluation**: Multiple sweeps until $V$ converges (many inner iterations!)
  2. **Policy improvement**: One sweep with argmax
- So a single PI iteration is much more expensive than a single VI iteration.

**(b) Fewer total iterations: [[Policy Iteration]]**
- Policy iteration converges in fewer **outer** iterations because each iteration fully evaluates the current policy before improving it.
- Value iteration makes small incremental progress each sweep.
- For finite MDPs, policy iteration converges in at most $|\mathcal{A}|^{|\mathcal{S}|}$ iterations (number of possible policies), but in practice converges much faster.

---

## Question 2a: Value in terms of Q (2.0p)

**Q:** Write the value $v^\pi(s)$ of a state $s$ under policy $\pi$ in terms of $\pi$ and $q^\pi(s,a)$. Give both stochastic and deterministic cases.

### Solution

> [!formula] Stochastic Policy
> $$v^\pi(s) = \sum_{a \in \mathcal{A}(s)} \pi(a|s) \, q^\pi(s, a)$$

> [!formula] Deterministic Policy
> $$v^\pi(s) = q^\pi(s, \pi(s))$$
> where $\pi(s)$ is the single action the deterministic policy selects in state $s$.

---

## Question 2b: Q-Value Iteration (1.0p)

**Q:** Rewrite the Value Iteration update (Eq. 4.10) in terms of Q-values.

### Solution

The standard Value Iteration update:
$$V_{k+1}(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V_k(s')]$$

> [!formula] Q-Value Iteration Update
> $$Q_{k+1}(s, a) = \sum_{s', r} p(s', r | s, a) \left[ r + \gamma \max_{a'} Q_k(s', a') \right]$$

> [!intuition] Why This Works
> Instead of iterating over $V$, we iterate over $Q$ directly. The $\max_{a'}$ in the target corresponds to acting optimally from the next state — same Bellman optimality structure, just applied to action values.

---

## Question 2c: Policy Evaluation for Q (1.0p)

**Q:** Rewrite the policy evaluation update (Eq. 4.4) to compute $Q^\pi(s,a)$ instead of $V^\pi(s)$. The answer should not contain $V$.

### Solution

> [!formula] Policy Evaluation for Q
> $$Q^\pi_{k+1}(s, a) = \sum_{s', r} p(s', r | s, a) \left[ r + \gamma \sum_{a'} \pi(a'|s') \, Q^\pi_k(s', a') \right]$$

The inner sum $\sum_{a'} \pi(a'|s') Q^\pi_k(s', a')$ replaces what would be $V^\pi(s')$, using the relationship $v^\pi(s') = \sum_{a'} \pi(a'|s') q^\pi(s', a')$.

---

## Question 2d: Policy Improvement for Q (1.0p)

**Q:** Rewrite the Policy Improvement step (p.80) in terms of $Q^\pi(s,a)$ instead of $V^\pi(s)$.

### Solution

> [!formula] Policy Improvement with Q
> $$\pi'(s) = \arg\max_a Q^\pi(s, a)$$

> [!intuition] Why Q is Easier
> With $V$: $\pi'(s) = \arg\max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V^\pi(s')]$ — requires knowing $p$.
> With $Q$: Just take the argmax — **no model needed!** This is why Q-values are preferred for model-free control.

---

## Question 2e: Policy Evaluation Differences (1.0p)

**Q:** The policy evaluation step on p.80 is different from the separate algorithm on p.75. What's the difference and why?

### Solution

- **Page 75 (standalone policy evaluation)**: Iterates until convergence ($\Delta < \theta$). Runs many sweeps to get an accurate $V^\pi$.
- **Page 80 (within policy iteration)**: May stop after a **single sweep** (or a few). This is because full convergence is unnecessary — even an approximate evaluation leads to policy improvement.

> [!tip] This is the GPI Idea
> [[Generalized Policy Iteration]]: you don't need perfect evaluation before improving. Any amount of evaluation + improvement progress drives you toward optimality. Value iteration is the extreme: one sweep of evaluation per improvement step.

---

## Question 2f: Bellman as Linear System (2.0p)

**Q:** For an MDP with two states, write the Bellman equations as a linear system $A \begin{bmatrix} V(s_1) \\ V(s_2) \end{bmatrix} = b$. What are $A$ and $b$?

### Solution

The [[Bellman Equation]] for a given policy $\pi$ (deterministic for simplicity):
$$V(s) = \sum_{s', r} p(s', r | s, \pi(s))[r + \gamma V(s')]$$

For two states, expanding:
$$V(s_1) = r(s_1, \pi(s_1)) + \gamma \left[ p(s_1|s_1, \pi(s_1)) V(s_1) + p(s_2|s_1, \pi(s_1)) V(s_2) \right]$$
$$V(s_2) = r(s_2, \pi(s_2)) + \gamma \left[ p(s_1|s_2, \pi(s_2)) V(s_1) + p(s_2|s_2, \pi(s_2)) V(s_2) \right]$$

Rearranging ($V(s) - \gamma \sum_{s'} p(s'|s,a) V(s') = r(s,a)$):

> [!formula] Linear System $Av = b$
> $$\begin{bmatrix} 1 - \gamma p(s_1|s_1,\pi(s_1)) & -\gamma p(s_2|s_1,\pi(s_1)) \\ -\gamma p(s_1|s_2,\pi(s_2)) & 1 - \gamma p(s_2|s_2,\pi(s_2)) \end{bmatrix} \begin{bmatrix} V(s_1) \\ V(s_2) \end{bmatrix} = \begin{bmatrix} r(s_1, \pi(s_1)) \\ r(s_2, \pi(s_2)) \end{bmatrix}$$

In compact form: $A = I - \gamma P_\pi$ and $b = r_\pi$.

> [!tip] This Generalizes
> For $n$ states: $v_\pi = (I - \gamma P_\pi)^{-1} r_\pi$. This is the closed-form solution to the Bellman equation. Only practical for small state spaces (matrix inversion is $O(n^3)$). See also [[LSTD]] which exploits this structure with function approximation.
