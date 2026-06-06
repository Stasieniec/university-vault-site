---
type: concept
aliases: [action-value methods]
course: [RL]
tags: [foundations, evaluation, exam-topic]
status: complete
---

# Action-Value Methods

## Definition

> [!definition] Action-Value Methods
> **Action-value methods** estimate the **value** of each action — the expected reward (or return) of selecting it — and then use those estimates to drive **action selection**. The policy is *implicit*: it is derived from the value estimates (e.g. greedy or [[Epsilon-Greedy Policy|ε-greedy]] w.r.t. $Q_t(a)$), rather than being parameterised and learned directly. They are the canonical approach to the [[Multi-Armed Bandit]] problem and the conceptual ancestor of [[SARSA]] / [[Q-Learning]] in the full MDP setting.

## Intuition

> [!intuition] Estimate first, act second
> The core loop is: *keep a running estimate of how good each action is, then prefer the action that looks best (while still exploring).* You never store a policy explicitly — you store numbers $Q_t(a)$, and the policy is just "look at the numbers and pick".
>
> This is the natural contrast to [[Policy Gradient Methods]]: action-value methods learn $Q$ and **read off** a policy; policy-gradient methods skip the values and **adjust the policy parameters** directly. In a bandit, "value of action $a$" is simply the expected reward $q_*(a)$; in a full MDP it becomes the [[action-value function|action-value function]] $q_\pi(s,a)$.

## Mathematical Formulation

The **true value** of an action is its expected reward:

$$q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a]$$

The **sample-average estimate** averages the rewards actually received for $a$:

$$Q_t(a) \doteq \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i = a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i = a}}$$

To avoid storing all past rewards, this is computed with the **incremental update rule**:

$$Q_{n+1} = Q_n + \frac{1}{n}\,[R_n - Q_n]$$

which has the general "error-correction" form

$$\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize}\,[\text{Target} - \text{OldEstimate}]$$

where:
- $A_t$ — action selected at step $t$; $R_t$ — reward received at step $t$
- $q_*(a)$ — true (unknown) expected reward of action $a$
- $Q_t(a)$ — current estimate of $q_*(a)$ at step $t$
- $\mathbb{1}_{A_i = a}$ — indicator, $1$ if action $a$ was taken at step $i$, else $0$
- $N_t(a) = \sum_{i<t}\mathbb{1}_{A_i=a}$ — number of times $a$ has been selected
- $\frac{1}{n}$ — step size; the $n$-th selection of the action uses step $1/n$
- $[R_n - Q_n]$ — the **error** between the latest reward (target) and the current estimate

By the Law of Large Numbers, $Q_t(a) \to q_*(a)$ as each action is sampled infinitely often. For **nonstationary** problems, replace $1/n$ with a constant step size $\alpha \in (0,1]$:

$$Q_{n+1} = Q_n + \alpha\,[R_n - Q_n]$$

giving an **exponential recency-weighted average** (recent rewards weighted more heavily):

$$Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1-\alpha)^{n-i} R_i$$

## Key Properties / Variants

- **Selection rule is separate from estimation.** Estimation gives $Q_t(a)$; a *selection rule* turns it into behaviour:
  - **Greedy:** $A_t = \arg\max_a Q_t(a)$ — pure exploitation, can lock onto a suboptimal arm.
  - **[[Epsilon-Greedy Policy|ε-greedy]]:** greedy with prob. $1-\varepsilon$, uniform-random with prob. $\varepsilon$; guarantees every action is sampled infinitely often so $Q_t(a)\to q_*(a)$.
  - **[[Upper Confidence Bound|UCB]]:** $A_t = \arg\max_a\big[Q_t(a) + c\sqrt{\tfrac{\ln t}{N_t(a)}}\big]$ — directs exploration toward *uncertain* actions instead of exploring blindly.
  - **Optimistic Initial Values:** set $Q_1(a)$ high so early rewards "disappoint" and force trial of all actions; only aids *initial* exploration.
- **Step-size convergence (stochastic approximation):** estimates converge w.p. 1 iff $\sum_n \alpha_n(a) = \infty$ and $\sum_n \alpha_n^2(a) < \infty$. Sample-average ($1/n$) satisfies both; constant $\alpha$ violates the second on purpose, so it keeps tracking a moving target.
- **Contrast with preference-based methods:** **Gradient bandit** algorithms learn preferences $H_t(a)$ via a [[Softmax Policy|softmax]] and stochastic gradient ascent — they do *not* estimate action values, so they are *not* action-value methods (they are the bandit-level analogue of policy gradient).
- **Scaling up:** in a full [[Markov Decision Process|MDP]] the same "estimate values, derive policy" principle gives [[Temporal Difference Learning|TD]] control methods [[SARSA]] (on-policy) and [[Q-Learning]] (off-policy), where the target becomes a bootstrapped return rather than a single reward.

```pseudo
Algorithm: Simple Bandit (ε-greedy Action-Value Method)
─────────────────────────────────────────────────────────
Initialize, for a = 1..k:
    Q(a) ← 0          # value estimate
    N(a) ← 0          # selection count

Loop forever:
    # --- Action selection (policy derived from Q) ---
    With probability ε:   A ← random action
    Otherwise:            A ← argmax_a Q(a)   (ties broken randomly)

    # --- Take action, observe reward ---
    R ← bandit(A)

    # --- Incremental value update ---
    N(A) ← N(A) + 1
    Q(A) ← Q(A) + (1 / N(A)) · [R − Q(A)]
```

## Connections

- Core setting: [[Multi-Armed Bandit]] (action-value methods are its standard solution)
- Special case / scaled to: [[action-value function]] and [[Q(s a)]] in a full [[Markov Decision Process]]
- Selection rules: [[Epsilon-Greedy Policy]], [[Upper Confidence Bound]], Optimistic Initial Values
- Central tension: [[Exploration vs Exploitation]]
- MDP successors: [[SARSA]], [[Q-Learning]], [[Expected SARSA]] (value-based [[Temporal Difference Learning|TD]] control)
- Contrasted with: [[Policy Gradient Methods]] (parameterise the policy directly), [[Softmax Policy]] / gradient bandits (learn preferences, not values)

## Appears In

- [[RL-Book Ch13 - Policy Gradient Methods]]
- [[RL-Book Ch2 - Multi-Armed Bandits]]
- [[RL-L01 - Intro, MDPs & Bandits]]
- [[RL-ES01 - Exercise Set Week 1]]
