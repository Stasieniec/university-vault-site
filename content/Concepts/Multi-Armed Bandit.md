---
type: concept
aliases: [bandit, k-armed bandit, multi-armed bandit problem, MAB]
course: [RL]
tags: [foundations, exam-topic]
status: complete
---

# Multi-Armed Bandit

## Definition

> [!definition] $k$-Armed Bandit Problem
> A simplified RL problem with **only one state**. At each time step, you choose one of $k$ actions ("arms"), and receive a reward drawn from a stationary probability distribution that depends on the action selected. The goal: maximize total reward over time.

> [!intuition] The Name
> Named after slot machines ("one-armed bandits") in casinos. Imagine facing $k$ slot machines, each with an unknown payoff distribution. Which do you play, and how do you decide?

## Action Values

> [!formula] True Action Value
> $$q_*(a) = \mathbb{E}[R_t \mid A_t = a]$$
> 
> The true expected reward for action $a$. Unknown to the agent.

> [!formula] Sample-Average Estimate
> $$Q_t(a) = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i=a}}$$
> 
> Average of rewards received for action $a$ so far. Converges to $q_*(a)$ by Law of Large Numbers.

## Action Selection Methods

### Greedy
$$A_t = \arg\max_a Q_t(a)$$
Pure exploitation. Can get stuck on suboptimal action.

### [[Epsilon-Greedy Policy|ε-Greedy]]
Pick greedy action with probability $1-\varepsilon$, random action with probability $\varepsilon$.

### [[Upper Confidence Bound|UCB]]
$$A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$
Adds an exploration bonus that shrinks as an action is tried more often. $c$ controls exploration degree. More principled than ε-greedy — preferentially explores uncertain actions.

### Optimistic Initial Values
Initialize $Q_0(a)$ high (e.g., +5 when rewards are around 0). Encourages initial exploration because early actual rewards will be "disappointing," causing the agent to try other actions.

### Gradient Bandit
Learn a preference $H_t(a)$ for each action, select via softmax:
$$\pi_t(a) = \frac{e^{H_t(a)}}{\sum_b e^{H_t(b)}}$$

Update:
$$H_{t+1}(a) = H_t(a) + \alpha(R_t - \bar{R}_t)(\mathbb{1}_{A_t=a} - \pi_t(a))$$
where $\bar{R}_t$ is the average reward baseline.

## Incremental Update

> [!formula] Incremental Action-Value Update
> $$Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]$$
> 
> General form: $\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize} \times [\text{Target} - \text{OldEstimate}]$

For **nonstationary** problems, use constant step-size $\alpha$ instead of $1/n$:
$$Q_{n+1} = Q_n + \alpha[R_n - Q_n]$$
This gives exponentially decaying weights to old rewards (more weight on recent).

## Relation to Full RL

> [!tip] Bandits as Special Case
> A bandit is a **1-state [[Markov Decision Process|MDP]]**. There's no state transition, no delayed reward, no sequential planning. It isolates the [[Exploration vs Exploitation]] problem.

## Connections

- Special case of: [[Markov Decision Process]] (single state)
- Core problem: [[Exploration vs Exploitation]]
- Selection methods: [[Epsilon-Greedy Policy]], [[Upper Confidence Bound]]
- Extended to: Contextual bandits (state-dependent), full MDPs

## Appears In

- [[RL-L01 - Intro, MDPs & Bandits]]
- [[RL-Book Ch2 - Multi-Armed Bandits]]
- [[RL-ES01 - Exercise Set Week 1]]
