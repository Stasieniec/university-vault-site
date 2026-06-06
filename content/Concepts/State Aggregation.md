---
type: concept
aliases: []
course: [RL]
tags: [approximation, function-approximation, exam-topic]
status: complete
---

# State Aggregation

## Definition

> [!definition] State Aggregation
> **State aggregation** is the simplest form of [[Function Approximation]]: the state space is partitioned into disjoint **groups**, and a single estimated value is shared by every state in a group. The estimate for a state is just the component of the weight vector $\mathbf{w}$ assigned to its group; updating one state updates the value of every state in that group.

## Intuition

> [!intuition] One Weight Per Bucket
> Imagine 1000 ordered states divided into 10 buckets of 100. Instead of storing 1000 values, you store 10 — one per bucket. All states in a bucket are treated as indistinguishable. The learned value function therefore looks like a **staircase**: flat within each group, jumping only at group boundaries.
>
> This is exactly the [[Tabular Methods|tabular]] case "coarsened": with one group per state you recover an exact lookup table; with fewer groups you trade fidelity for compactness and generalization.

## Mathematical Formulation

State aggregation is a special case of [[Linear Function Approximation]] in which the feature vector $\mathbf{x}(s)$ is **one-hot**: it has a single $1$ in the position of $s$'s group and $0$ elsewhere.

> [!formula] Value Estimate
> $$\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s) = w_{g(s)}$$
>
> where:
> - $g(s) \in \{1, \ldots, d\}$ — the group (aggregate) that state $s$ belongs to
> - $\mathbf{x}(s)$ — one-hot feature vector, $x_i(s) = \mathbf{1}[\,g(s) = i\,]$
> - $\mathbf{w} \in \mathbb{R}^d$ — one weight per group ($d$ = number of groups $\ll |\mathcal{S}|$)

Because the feature is one-hot, the gradient is also one-hot and the [[Semi-Gradient Methods|semi-gradient]] update touches only the active group's weight:

> [!formula] Semi-Gradient Update
> $$w_{g(S_t)} \leftarrow w_{g(S_t)} + \alpha\big[\,U_t - \hat{v}(S_t, \mathbf{w})\,\big]$$
>
> where:
> - $U_t$ — update target (e.g., the Monte Carlo return $G_t$, or the bootstrapped [[TD error|TD]] target $R_{t+1} + \gamma\,\hat{v}(S_{t+1}, \mathbf{w})$)
> - $\alpha$ — step size
> - only $w_{g(S_t)}$ changes; $\nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w}) = \mathbf{x}(S_t)$ is one-hot

> [!formula] What It Converges To
> Under [[Monte Carlo Methods|Monte Carlo]] / gradient updates, each group's weight converges to the $\mu$-weighted average of the true values of the states in that group:
> $$w_i^\* = \frac{\sum_{s : g(s)=i} \mu(s)\, v_\pi(s)}{\sum_{s : g(s)=i} \mu(s)}$$
>
> where $\mu(s)$ is the [[On-Policy Distribution|on-policy distribution]] (fraction of time spent in $s$). Minimizing the [[Mean Squared Value Error|MSVE]] within a group thus yields a $\mu$-weighted mean.

## Key Properties / Variants

- **Special case of linear FA**: one-hot features make every result for [[Linear Function Approximation]] apply — in particular semi-gradient TD(0) converges to the [[TD Fixed Point]].
- **Staircase value function**: $\hat{v}$ is piecewise-constant across groups; it cannot represent variation *within* a group, so resolution is limited by the partition.
- **Distribution bias**: accuracy follows $\mu(s)$. In the 1000-state random-walk example, central states (visited most often) are approximated best, and the staircase is visibly skewed near the edges of each group.
- **Generalization vs. discrimination trade-off**: coarse groups generalize strongly (one update affects many states) but cannot distinguish states that need different values; fine groups discriminate but generalize little and approach the tabular cost.
- **No overlap (unlike [[Tile Coding]])**: groups are disjoint, so exactly one feature is active. Tile coding generalizes state aggregation by using *multiple overlapping* partitions, giving smoother generalization while keeping binary features.

```pseudo
Algorithm: Gradient MC with State Aggregation
──────────────────────────────────────────────
Input: aggregation map g(·) into d groups, step size α
Initialize w ∈ R^d  (e.g., w = 0)

Loop for each episode:
  Generate S0, A0, R1, ..., S_T following π
  Loop for t = 0, 1, ..., T-1:
    G ← return from step t  (Σ_{k} γ^k R_{t+1+k})
    i ← g(S_t)                      # active group
    w[i] ← w[i] + α (G - w[i])      # only this weight updates
```

> [!warning] Resolution Is Hard-Coded
> State aggregation cannot represent any structure finer than its partition: two states forced into the same group are permanently assigned the same value, no matter how much data you collect. Choosing a good aggregation is itself [[Feature Construction|feature engineering]], and a poor partition caps achievable accuracy regardless of $\alpha$ or sample count.

## Connections

- Special case of: [[Linear Function Approximation]] (one-hot features), hence of [[Function Approximation]]
- Coarsens: [[Tabular Methods]] (one group per state recovers the exact table)
- Generalized by: [[Tile Coding]] (multiple overlapping partitions)
- Trained with: [[Semi-Gradient Methods]], [[Monte Carlo Methods]] / [[Temporal Difference Learning]] targets
- Accuracy governed by: [[On-Policy Distribution]] $\mu(s)$ and [[Mean Squared Value Error|MSVE]]
- Partial-observability analogue: using observations as state ($S = O$) is state aggregation where the aggregation feature is the [[POMDP|observation]] over the latent state — see [[Partial Observability]]

## Appears In

- [[RL-L13 - Partial Observability]]
- [[RL-L05 - Tabular to Approximation]]
- [[RL-Book Ch9 - On-Policy Prediction with Approximation]]
