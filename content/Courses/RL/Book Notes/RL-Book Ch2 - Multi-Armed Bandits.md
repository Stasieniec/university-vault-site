---
type: book-chapter
course: RL
book: "Reinforcement Learning: An Introduction (2nd ed.)"
chapter: 2
sections: ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "2.10"]
topics:
  - "[[Multi-Armed Bandit]]"
  - "[[Epsilon-Greedy Policy]]"
  - "[[Optimistic Initial Values]]"
  - "[[Upper Confidence Bound]]"
  - "[[Exploration vs Exploitation]]"
status: complete
---

# Chapter 2: Multi-armed Bandits

## Overview
This chapter explores the **evaluative** aspect of [[Reinforcement Learning]] in a simplified **nonassociative** setting. Unlike supervised learning (instructive feedback), RL uses evaluative feedback which indicates how good an action was, but not if it was the best possible. The k-armed bandit problem serves as the primary framework to introduce the fundamental conflict between **exploration** and **exploitation**.

---

## 2.1 A k-armed Bandit Problem
> [!definition] **Multi-Armed Bandit Problem**
> You are faced repeatedly with a choice among $k$ different options (actions). After each choice, you receive a numerical reward from a stationary probability distribution depending on the action. The objective is to maximize the expected total reward over a time period (e.g., 1000 steps).

### Key Variables
- $A_t$: Action selected at time step $t$.
- $R_t$: Reward received at time step $t$.
- $q_*(a)$: True value (expected reward) of action $a$.
- $Q_t(a)$: Estimated value of action $a$ at time $t$.

> [!formula] **Action Value**
> $$q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a]$$

### Exploration vs Exploitation
- **Exploitation**: Choosing the **greedy** action (the one with the highest current estimate $Q_t(a)$) to maximize immediate reward.
- **Exploration**: Choosing non-greedy actions to improve estimates of their true values.
- **The Conflict**: You cannot explore and exploit simultaneously with a single action. Balancing this is a central challenge in RL.

---

## 2.2 Action-value Methods
Methods that estimate action values and use them for selection.

### Sample-average Method
> [!formula] **Sample-Average Estimation**
> $$Q_t(a) \doteq \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i=a}}$$
> Where $\mathbb{1}_{predicate}$ is 1 if true, 0 otherwise. By the Law of Large Numbers, $Q_t(a) \to q_*(a)$ as $t \to \infty$.

### Action Selection Rules
1. **Greedy**: $A_t \doteq \text{argmax}_a Q_t(a)$.
2. **$\varepsilon$-greedy**: Behave greedily most of the time, but with probability $\varepsilon$, select an action at random from all $k$ actions.
   - **Benefit**: Ensures all actions are sampled infinitely, so $Q_t(a)$ converges to $q_*(a)$.

---

## 2.3 The 10-armed Testbed
> [!example] **The Testbed Setup**
> A suite of 2000 randomly generated 10-armed bandit problems.
> - $q_*(a) \sim \mathcal{N}(0, 1)$
> - Rewards $R_t \sim \mathcal{N}(q_*(A_t), 1)$

**Results summary:**
- **Greedy ($\varepsilon=0$)**: Improves slightly faster initially but levels off early at a suboptimal level. Often gets stuck on a suboptimal action (the "unlucky" start).
- **$\varepsilon=0.1$**: Explores more, finds the optimal action sooner, but levels off at $1-\varepsilon$ (91%) optimal selection.
- **$\varepsilon=0.01$**: Improves more slowly but eventually outperforms $\varepsilon=0.1$.

---

## 2.4 Incremental Implementation
To avoid growing memory/computation requirements, we update averages incrementally.

> [!formula] **Incremental Update Rule**
> $$Q_{n+1} = Q_n + \frac{1}{n} [R_n - Q_n]$$
> **General Form:**
> $$\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize} [\text{Target} - \text{OldEstimate}]$$

### Pseudocode: Simple Bandit Algorithm
```python
Initialize, for a = 1 to k:
    Q(a) <- 0
    N(a) <- 0

Loop forever:
    # Action Selection
    if random() < epsilon:
        A <- random_action()
    else:
        A <- argmax(Q(a)) # break ties randomly
    
    # Execution
    R <- bandit(A)
    N(A) <- N(A) + 1
    
    # Update
    Q(A) <- Q(A) + (1/N(A)) * (R - Q(A))
```

---

## 2.5 Tracking a Nonstationary Problem
In nonstationary problems (reward distributions change over time), recent rewards should carry more weight.

> [!formula] **Constant Step-Size Update**
> $$Q_{n+1} \doteq Q_n + \alpha [R_n - Q_n]$$
> Results in an **exponential recency-weighted average**:
> $$Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^n \alpha(1-\alpha)^{n-i} R_i$$

### Convergence Conditions (Stochastic Approximation)
For estimates to converge with probability 1, the step-size sequence $\{\alpha_n(a)\}$ must satisfy:
1. $\sum_{n=1}^\infty \alpha_n(a) = \infty$ (Steps large enough to overcome initial conditions)
2. $\sum_{n=1}^\infty \alpha_n^2(a) < \infty$ (Steps small enough to ensure convergence)
*Note: Constant $\alpha$ violates the second condition, allowing the estimate to follow nonstationary changes.*

---

## 2.6 Optimistic Initial Values
Setting $Q_1(a)$ to a high value (e.g., +5 in the 10-armed testbed) forces the agent to explore.
- **Mechanism**: Any reward received is "disappointing" compared to the estimate, driving the agent to try all other actions.
- **Limitation**: Only helps with initial exploration; not useful for long-term nonstationarity.

---

## 2.7 Upper-Confidence-Bound (UCB) Action Selection
$\varepsilon$-greedy explores blindly. UCB explores by favoring actions that are uncertain.

> [!formula] **UCB Action Selection**
> $$A_t \doteq \text{argmax}_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$
> - **$c$**: Controls degree of exploration.
> - **$\sqrt{\dots}$**: Measure of uncertainty/variance in the estimate.
> - **Effect**: As $t$ increases, the term grows for all actions, ensuring every action is tried eventually. As $N_t(a)$ increases, the term shrinks.

---

## 2.8 Gradient Bandit Algorithms
Instead of estimating values, we learn numerical **preferences** $H_t(a)$.

### Soft-max distribution
> [!formula] **Action Probabilities**
> $$\Pr\{A_t = a\} \doteq \frac{e^{H_t(a)}}{\sum_{b=1}^k e^{H_t(b)}} \doteq \pi_t(a)$$

### Update Rule (Stochastic Gradient Ascent)
Upon receiving reward $R_t$:
> [!formula] **Preference Update**
> $$H_{t+1}(A_t) \doteq H_t(A_t) + \alpha (R_t - \bar{R}_t)(1 - \pi_t(A_t))$$
> $$H_{t+1}(a) \doteq H_t(a) - \alpha (R_t - \bar{R}_t)\pi_t(a) \quad \text{for } a \neq A_t$$
> **$\bar{R}_t$**: The average reward baseline. It reduces variance without changing the expected update.

---

## 2.9 Associative Search (Contextual Bandits)
In **Contextual Bandits**, the learner is given a "clue" or signal about the situation.
- **Goal**: Learn a **policy** (mapping from situation/context to action).
- **Position**: Intermediate between simple bandits (no context) and full RL (actions affect future state/context).

---

## Comparison Summary
| Method | Key Parameter | Exploration Strategy |
| :--- | :--- | :--- |
| **Greedy** | None | Exploitation only |
| **$\varepsilon$-greedy** | $\varepsilon$ | Random selection |
| **Optimistic Initial Values** | $Q_1$ | Initial "disappointment" forces trial |
| **UCB** | $c$ | Uncertainty-based |
| **Gradient Bandit** | $\alpha$ | Soft-max preferences relative to baseline |

> [!tip] **Summary of Parameter Study (Figure 2.6)**
> All algorithms show an "inverted-U" performance curve. UCB typically performs best on the stationary 10-armed testbed, but it is harder to generalize to large state spaces than $\varepsilon$-greedy or gradient methods.

---

## Key Takeaways
1. **Evaluative feedback** requires a balance of exploration and exploitation.
2. **Sample averages** are for stationarity; **constant step sizes** are for non-stationarity.
3. **Optimistic initialization** is a simple but limited trick for initial exploration.
4. **UCB** provides a more sophisticated exploration based on uncertainty.
5. **Gradient bandits** optimize preferences rather than estimating values directly.
6. **Baselines** in gradient methods dramatically speed up learning by reducing variance.
