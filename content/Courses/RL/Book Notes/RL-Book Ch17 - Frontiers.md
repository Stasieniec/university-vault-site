---
type: book-chapter
course: RL
book: "Reinforcement Learning: An Introduction (2nd ed.)"
chapter: 17
sections: ["17.1", "17.2", "17.3", "17.4", "17.5", "17.6"]
topics:
  - "[[Reinforcement Learning]]"
  - "[[Markov Decision Process]]"
  - "[[Function Approximation]]"
status: complete
---

# RL-Book Chapter 17: Frontiers

## Overview
Chapter 17 explores the "frontiers" of [[Reinforcement Learning]], touching on topics that extend beyond the standard [[Markov Decision Process]] framework. It covers general value functions, temporal abstraction (options), the critical transition from states to observations (partial observability), reward design, and the open challenges facing the field.

---

## 17.1 General Value Functions and Auxiliary Tasks
The concept of a [[Value Function]] is generalized to **General Value Functions (GVFs)**, which predict arbitrary signals rather than just reward.

> [!definition] Cumulant
> The signal being predicted in a GVF is called the **cumulant**, denoted $C_t \in \mathbb{R}$.

> [!formula] General Value Function (GVF)
> $$v_{\pi, \gamma, C}(s) = \mathbb{E} \left[ \sum_{k=t}^{\infty} C_{k+1} \prod_{i=t+1}^{k} \gamma(S_i) \mid S_t=s, A_{t:\infty} \sim \pi \right]$$
> where $\gamma: \mathcal{S} \to [0, 1]$ is a state-dependent termination/[[Discount Factor]] function.

- **Auxiliary Tasks**: Learning to predict multiple signals (e.g., pixel changes, future rewards) forces the agent to develop robust internal representations that often accelerate learning on the main task.
- **Pavlovian Control**: Built-in reflexes triggered by learned predictions (e.g., a robot heading to a charger when it predicts low battery).

---

## 17.2 Temporal Abstraction via Options
MDPs can be applied at multiple time scales. **Options** formalize higher-level actions that persist over multiple time steps.

> [!definition] Option
> An option $\omega$ is a pair $\langle \pi_\omega, \gamma_\omega \rangle$ consisting of a [[Policy]] $\pi_\omega$ and a termination function $\gamma_\omega$.

- **Option Models**: Consist of two parts:
    1. **Reward part**: Expected cumulative reward during execution.
    2. **State-transition part**: The discounted probability of terminating in a specific state.
- **Bellman Equation for Options**:
    $$v_\pi(s) = \sum_{\omega \in \Omega(s)} \pi(\omega|s) \left[ r(s, \omega) + \sum_{s'} p(s'|s, \omega) v_\pi(s') \right]$$

---

## 17.3 Observations and State (Detailed)
Throughout the book, it was assumed the agent perceives the environment's state directly. In reality, agents receive **observations** which provide only partial information.

### Partial Observability
While [[Function Approximation]] can implicitly handle some partial observability (by choosing parameters that don't depend on hidden variables), an explicit treatment is required for complex environments.

### History and State
- **History ($H_t$)**: The sequence of all past actions and observations: $H_t = A_0, O_1, A_1, O_2, \dots, A_{t-1}, O_t$.
- **State ($S_t$)**: A compact summary of the history, $S_t = f(H_t)$.

> [!definition] Markov State
> A state $S_t$ is **Markov** if it summarizes all information in the history necessary for predicting future observations.
> Formally, $f(h) = f(h') \implies p(\tau | h) = p(\tau | h')$ for any future test $\tau$.

### The State-Update Function
To remain compact and computable, the state must be updated incrementally:
$$S_{t+1} = u(S_t, A_t, O_{t+1})$$

### Two Approaches to Representation
1. **POMDPs (Partially Observable MDPs)**:
    - Assumes a "latent" environment state $X_t$ that is never directly seen.
    - The agent maintains a **Belief State**: a probability distribution over possible latent states.
    - Updated via **Bayes' Rule**:
    > [!formula] Belief State Update
    > $$u(s, a, o)[i] = \frac{\sum_{x=1}^d s[x]p(i, o|x, a)}{\sum_{x', o'} \sum_{x=1}^d s[x]p(x', o'|x, a)}$$
    - *Critique*: Scales poorly and relies on unobservable semantics ($X_t$).

2. **PSRs (Predictive State Representations)**:
    - Grounded in observable data.
    - State is defined as a vector of probabilities for "core tests" (specific future action-observation sequences).

### Approximate State
In practice, $S_t$ is rarely perfectly Markov. Common heuristics:
- **Immediate Observation**: $S_t = O_t$.
- **$k$-th Order History**: $S_t = (O_t, A_{t-1}, O_{t-1}, \dots, O_{t-k})$.
- **Feature-based**: Multiple GVFs/auxiliary tasks provide features for the state representation.

> [!intuition] The Heuristic of Representation
> A state that is good for predicting many different things (auxiliary tasks) is likely to contain the information necessary for predicting reward and making optimal decisions.

---

## 17.4 Designing Rewards
[[Reinforcement Learning]] depends heavily on the reward signal, which is the designer's way of communicating the goal.

- **Sparse Rewards**: The "plateau problem" where the agent wanders without feedback.
- **Value Function Initialization**: A way to guide learning without changing the reward:
    $$\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s) + v_0(s)$$
- **Shaping**: Gradually changing the reward signal or task dynamics to lead the agent toward the goal (Skinner).
- **Inverse RL**: Learning the reward signal by observing an expert's behavior.

> [!warning] Reward Design Risks
> Agents may find "loopholes" or unexpected ways to maximize reward that violate the designer's intent.

---

## 17.5 Remaining Issues
1. **Incremental Deep Learning**: Overcoming "catastrophic interference" in online settings.
2. **Representation Learning**: Automating the construction of the state-update function.
3. **Planning with Learned Models**: Scaling Dyna-like architectures to complex function approximation.
4. **Automated Task Selection**: How agents can choose their own subgoals/auxiliary tasks.
5. **Curiosity**: Using "intrinsic reward" to drive exploration and learning progress.

---

## 17.6 Future of AI
- **Complete Agents**: Transitioning from superhuman performance in narrow domains to interactive, generalist agents.
- **Safety**: Ensuring optimization doesn't lead to dangerous unintended consequences.
- **Prometheus vs. Pandora**: AI as a tool that can either solve global challenges (fire) or release new perils (the box).

---

## Summary
Chapter 17 shifts from *how* to learn given a state, to *what* to learn (GVFs, Options) and *how* to represent the environment (Observations, State-Update). The fundamental message is that the future of RL lies in an agent-centric view where representation, tasks, and goals are discovered and curated by the agent itself.
