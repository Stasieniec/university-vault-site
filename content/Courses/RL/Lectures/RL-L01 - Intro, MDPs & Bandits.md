---
type: lecture
course: RL
week: 1
lecture: 1
book_sections: ["Ch 1.1-1.4", "Ch 1.6", "Ch 2.1-2.4", "Ch 2.6-2.7", "Ch 3.1-3.3"]
topics:
  - "[[Reinforcement Learning]]"
  - "[[Multi-Armed Bandit]]"
  - "[[Markov Decision Process]]"
  - "[[Value Function]]"
  - "[[Policy]]"
  - "[[Return]]"
  - "[[Exploration vs Exploitation]]"
  - "[[Epsilon-Greedy Policy]]"
  - "[[Upper Confidence Bound]]"
status: complete
---

# RL Lecture 1: Introduction, MDPs & Bandits

## 1. Introduction to Reinforcement Learning

> [!definition] Reinforcement Learning (RL)
> **Reinforcement Learning** is a computational approach to learning from interaction. It focuses on goal-directed learning where an **agent** learns what to do—how to map situations to actions—so as to maximize a numerical **reward** signal.

### 1.1 Key Characteristics
Unlike other machine learning paradigms, RL is distinguished by:
1. **Trial-and-error search**: The learner is not told which actions to take but must discover which yield the most reward by trying them.
2. **Delayed reward**: Actions may affect not only the immediate reward but also the next situation and, through that, all subsequent rewards.

### 1.2 RL vs. Other Paradigms
- **[[Supervised Learning]]**: Learning from a training set of labeled examples provided by a supervisor. In RL, the agent must learn from its own experience in uncharted territory.
- **[[Unsupervised Learning]]**: Finding hidden structure in unlabeled data. RL is focused on maximizing a reward signal rather than finding structure.

### 1.3 Elements of Reinforcement Learning
Beyond the agent and environment, we identify four main sub-elements:
1. **[[Policy]]**: A mapping from perceived states of the environment to actions to be taken (the agent's behavior).
2. **[[Reward Signal]]**: Defines the goal; a single number sent by the environment at each time step that the agent seeks to maximize in the long run.
3. **[[Value Function]]**: Specifies what is good in the long run. The value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state.
4. **[[Model of the Environment]]**: (Optional) Something that mimics the behavior of the environment, used for planning.

> [!intuition] Rewards vs. Values
> Rewards are immediate (like pleasure/pain), while values are fartsighted (long-term desirability). We seek actions that lead to states of highest **value**, not necessarily highest immediate reward.

---

## 2. Multi-Armed Bandits

The **k-armed bandit problem** is a simplified RL setting that involves only a single state (nonassociative), focusing purely on the evaluative aspect of feedback.

### 2.1 The Problem Formulation
You are faced repeatedly with a choice among $k$ different actions. After each choice, you receive a numerical reward from a stationary probability distribution.

> [!formula] Action Value
> The true value of an action $a$, denoted $q_*(a)$, is the expected reward given that $a$ is selected:
> $$q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a]$$
> where:
> - $q_*(a)$: True value of action $a$.
> - $A_t$: Action selected at time $t$.
> - $R_t$: Reward received at time $t$.

### 2.2 Action-Value Methods
We estimate $q_*(a)$ using the **sample-average** method:
$$Q_t(a) \doteq \frac{\text{sum of rewards when } a \text{ taken prior to } t}{\text{number of times } a \text{ taken prior to } t}$$

### 2.3 Incremental Implementation
To avoid storing all rewards, we update the average incrementally:

> [!formula] Incremental Update Rule
> $$Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]$$
> General form:
> $$\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize} \big[ \text{Target} - \text{OldEstimate} \big]$$

### 2.4 Exploration vs. Exploitation
- **Exploitation**: Selecting the **greedy action** (the one with the highest $Q_t(a)$) to maximize immediate reward.
- **Exploration**: Selecting non-greedy actions to improve estimates of their values.

#### Epsilon-Greedy Policy
An **[[\epsilon-greedy policy]]** selects the greedy action with probability $1-\epsilon$, and a random action with probability $\epsilon$.

#### Optimistic Initial Values
Setting initial estimates $Q_1(a)$ to a high value (higher than any likely reward) encourages exploration. The agent is "disappointed" by initial rewards and tries all actions before settling.

#### Upper Confidence Bound (UCB)
UCB selects actions based on both their estimated value and the uncertainty in that estimate.

> [!formula] UCB Action Selection
> $$A_t \doteq \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$
> where:
> - $t$: Total number of time steps.
> - $N_t(a)$: Number of times action $a$ has been selected.
> - $c$: Confidence level (controls exploration).
> - The square-root term represents the uncertainty/variance.

---

## 3. Markov Decision Processes (MDPs)

> [!definition] Markov Decision Process (MDP)
> A formalization of sequential decision-making where actions influence not just immediate rewards, but also subsequent states. An MDP is defined by its **states**, **actions**, **rewards**, and **dynamics**.

### 3.1 The Agent-Environment Interface
- At each time step $t$, the agent receives state $S_t \in \mathcal{S}$.
- The agent takes action $A_t \in \mathcal{A}(s)$.
- The environment responds with a reward $R_{t+1} \in \mathcal{R}$ and a new state $S_{t+1} \in \mathcal{S}$.

> [!figure] Agent-Environment Interaction
> The process generates a trajectory: $S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots$
> ```mermaid
> graph LR
>     Agent -- Action A_t --> Environment
>     Environment -- Reward R_{t+1} --> Agent
>     Environment -- State S_{t+1} --> Agent
> ```

### 3.2 Transition Dynamics
The dynamics of a finite MDP are completely defined by the probability:
> [!formula] Dynamics Function
> $$p(s', r \mid s, a) \doteq \Pr\{S_t = s', R_t = r \mid S_{t-1} = s, A_{t-1} = a\}$$
> This function defines the probability of transitioning to state $s'$ with reward $r$, given the current state $s$ and action $a$.

### 3.3 Goals and Returns
The agent's goal is to maximize the **expected return** $G_t$.

> [!formula] Discounted Return
> $$G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$
> where $\gamma \in [0, 1]$ is the **discount factor**.
> - $\gamma = 0$: Agent is "myopic" (maximizes immediate reward).
> - $\gamma \to 1$: Agent is "farsighted" (weights future rewards heavily).

---

## 4. Value Functions & Bellman Equations

### 4.1 Policies
A **[[Policy]]** $\pi(a|s)$ is the probability of taking action $a$ in state $s$.

### 4.2 State-Value and Action-Value Functions
- **State-value function** $v_\pi(s)$: Expected return starting from state $s$ following policy $\pi$.
  $$v_\pi(s) \doteq \mathbb{E}_\pi [G_t \mid S_t = s]$$
- **Action-value function** $q_\pi(s, a)$: Expected return starting from $s$, taking action $a$, then following $\pi$.
  $$q_\pi(s, a) \doteq \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]$$

### 4.3 The Bellman Equation for $v_\pi$
The Bellman Equation expresses a recursive relationship between the value of a state and its successor states.

> [!formula] Bellman Equation
> $$v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right]$$
> **Intuition**: The value of the current state is the expected immediate reward plus the discounted value of the next state, averaged over all possible actions and outcomes.

### 4.4 Bellman Optimality Equations
The optimal value functions $v_*$ and $q_*$ satisfy the **Bellman Optimality Equations**, which don't depend on a specific policy but assume the best action is always taken.

> [!formula] Bellman Optimality Equation for $v_*$
> $$v_*(s) = \max_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_*(s') \right]$$

> [!formula] Bellman Optimality Equation for $q_*$
> $$q_*(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \max_{a'} q_*(s', a') \right]$$

> [!example] Gridworld
> A simple 5x5 grid where certain cells (A, B) provide large rewards and transport the agent to other locations (A', B'). Moving off the grid results in a reward of -1 and keeps the agent in place. Under an equiprobable random policy (north, south, east, west), states near the center have higher values, while those near edges have lower (often negative) values due to the risk of hitting the boundary.

> [!example] Golf
> The state is the location of the ball.
> - **Rewards**: -1 for each stroke until the ball is in the hole.
> - **State Value $v_{\text{putt}}(s)$**: Negative of the number of strokes to the hole using only a putter. Contours represent regions from which 1, 2, or 3 putts are needed.
> - **Action Value $q_*(s, \text{driver})$**: Value after taking the first shot with a driver, then following an optimal policy (using driver or putter as appropriate). Driving allows reaching the green faster but with more risk/uncertainty.

---

## 5. Key Figures & Diagrams

### 5.1 Backup Diagrams
Backup diagrams are graphical summaries of value function updates.
- **State-value backup ($v_\pi$)**: From a state $s$ (open circle), branch to possible actions $a$ (solid circles) via $\pi(a|s)$, then to next states $s'$ (open circles) via $p(s',r|s,a)$.
- **Action-value backup ($q_\pi$)**: From a state-action pair $(s,a)$ (solid circle), branch to possible next states $s'$ (open circles), then to next actions $a'$ (solid circles).

### 5.2 The 10-Armed Testbed
This figure (Sutton & Barto Fig 2.1) visualizes the distributions of rewards for a set of 10 arms. Each arm has a mean reward $q_*(a)$ (sampled from a normal distribution) and actual rewards are sampled around that mean. This testbed is used to compare epsilon-greedy methods ($0, 0.01, 0.1$) showing that non-zero epsilon performs better in the long run by avoiding suboptimal convergence.

---
## Summary Takeaways
- **RL Problem**: Goal-directed interaction with an environment to maximize cumulative reward.
- **Bandits**: Simplest case; one state, focus on exploration ($\epsilon$-greedy, UCB, Optimistic Init).
- **MDPs**: Full case; actions affect state transitions. Handled via value functions and policies.
- **Bellman Equation**: The core recursive tool for evaluating policies by linking current and future values. 
- **Exploration/Exploitation**: The central dilemma of RL. 
