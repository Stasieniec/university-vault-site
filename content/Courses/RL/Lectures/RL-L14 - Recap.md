---
type: lecture
course: RL
week: 7
lecture: 14
book_sections: []
topics:
  - "[[Reinforcement Learning]]"
status: complete
---

# RL Lecture 14 - Recap & Exam Preparation

## Overview & Motivation

This is the final recap lecture covering all 13 previous lectures. The goal is to sketch the coherence of all lecture topics and provide a unified view of the methods studied throughout the course. This note is structured as an **exam-preparation resource** with comparison tables, convergence guarantees, and method taxonomies.

> [!warning]
> The recap cannot cover all 14 lectures in 90 minutes. **Topics outside of the recap can be on the exam too.** Study each lecture's material independently.

> [!tip] Exam: What to Know
> **Know the advantages, disadvantages, and limitations of each method, and the situations where a certain method should be preferred.** A cheat sheet with key update equations is provided during the exam. Many algorithms have variants (Q- and V-version, importance weights, etc.) -- the cheat sheet has only the most important ones.

---

## The Big Picture

The central problem in RL: **find an optimal policy $\pi^*$** (or evaluate a given policy) from data.

$$D = \{(s_i, a_i, r_i, s'_i)\}_{i=1 \ldots N}$$

When MDP dynamics $p(s', r \mid s, a)$ are known, we use [[Dynamic Programming]]. When we only have data from the MDP, we use **reinforcement learning**.

### Three Approaches to Learning Policies

| Approach | What We Learn | Methods |
|---|---|---|
| **Value-based** | $V(s)$ or $Q(s,a)$, derive policy from values | [[Monte Carlo Methods\|MC]], [[Temporal Difference Learning\|TD]], [[Q-Learning]], [[SARSA]], [[Deep Q-Network (DQN)\|DQN]], [[Conservative Q-Learning (CQL)\|CQL]] |
| **Policy-based** | $\pi(a \mid s)$ directly | [[REINFORCE]], [[Policy Gradient Theorem\|PGT]], [[Actor-Critic]], [[Deterministic Policy Gradient\|DPG]]/DDPG, [[Soft Actor-Critic (SAC)\|SAC]] |
| **Model-based** | $p(s' \mid s, a)$ and $r(s, a)$, then plan | [[Dyna]], [[Monte Carlo Tree Search (MCTS)\|MCTS]], AlphaGo Zero |

Other approaches: Decision Transformer, Decision Diffuser.

---

## Taxonomy of RL Methods

```
                        RL Methods
                            |
            ________________|________________
           |                |                |
      Value-Based      Policy-Based     Model-Based
           |                |                |
    MC, TD, Q-learning  REINFORCE       Dyna, MCTS
    SARSA, DQN, CQL    PGT, AC, DPG    AlphaGo Zero
                        SAC
```

### Model-Free vs Model-Based

| | Model-Free | Model-Based |
|---|---|---|
| **What it learns** | Value function or policy directly | Transition model $p(s'\mid s,a)$ and reward $r(s,a)$ |
| **Planning** | No explicit planning | Uses model for planning / simulation |
| **Sample efficiency** | Lower | Higher (can generate synthetic data) |
| **Model errors** | No model bias | Model errors compound |
| **Examples** | [[Q-Learning]], [[SARSA]], [[REINFORCE]], [[Actor-Critic]] | [[Dyna]], [[Monte Carlo Tree Search (MCTS)\|MCTS]], AlphaGo Zero |

---

## Lecture-by-Lecture Summary

### L1: MDPs, Bandits, Exploration vs Exploitation

- [[Markov Decision Process]]: states, actions, transitions, rewards, discount factor
- [[Multi-Armed Bandit]]: simplified RL without state transitions
- [[Exploration vs Exploitation]]: fundamental tension in RL
- Methods: [[Upper Confidence Bound\|UCB]], [[Epsilon-Greedy Policy|epsilon-greedy]], [[Optimistic Initial Values]]

### L2: Dynamic Programming

- [[Dynamic Programming]]: requires known model $p(s',r \mid s,a)$
- [[Policy Iteration]]: alternate [[Policy Evaluation]] and [[Policy Improvement]]
- [[Value Iteration]]: combine evaluation and improvement into one step
- [[Generalized Policy Iteration]]: general framework unifying both

### L3: Monte Carlo Methods

- [[Monte Carlo Methods]]: learn from complete episodes
- [[First-Visit MC]]: use first visit to each state per episode
- [[Monte Carlo Control]]: MC + policy improvement
- [[Exploring Starts]]: ensure all state-action pairs are visited
- Key: **unbiased** estimates, but **high variance** and must wait for episode end

### L4: Temporal Difference Learning

- [[Temporal Difference Learning]]: learn from incomplete episodes via [[Bootstrapping]]
- [[TD(0)]]: one-step TD prediction
- [[SARSA]]: on-policy TD control
- [[Q-Learning]]: off-policy TD control
- [[Expected SARSA]]: reduces variance of SARSA
- Key: **biased** (bootstrapping) but **low variance**, updates every step

### L5: From Tabular to Approximation

- [[Function Approximation]]: generalize beyond tabular methods
- [[Gradient Descent]] / [[Stochastic Gradient Descent]]: optimize function parameters
- [[Linear Function Approximation]]: $\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)$
- [[Feature Construction]], [[Tile Coding]]: designing features for linear methods
- [[Neural Network Function Approximation]]: nonlinear function approximation

### L6: On-Policy TD with Approximation

- [[Semi-Gradient Methods]]: gradient only through the estimate, not the target
- [[Episodic Semi-Gradient Control]]: semi-gradient SARSA for control
- Key issue: semi-gradient methods are not true gradient methods

### L7: Off-Policy RL with Approximation

- [[Deadly Triad]]: function approximation + bootstrapping + off-policy = instability
- [[Off-Policy Divergence]]: Baird's counterexample
- [[Gradient-TD Methods]]: GTD2 -- true gradient methods that converge off-policy
- [[LSTD]]: least-squares TD (no step size needed, batch method)
- [[Importance Sampling]]: correcting for distribution mismatch
- Errors: [[Bellman Error]], [[TD Error]], Value Error (VE), Projected Bellman Error (PBE)

### L8: Deep RL (Value-Based)

- [[Deep Q-Network (DQN)]]: Q-learning with deep neural networks
- Key tricks: [[Experience Replay]], [[Target Network]]
- [[Conservative Q-Learning (CQL)]]: for [[Offline Reinforcement Learning]]

### L9: Policy Gradient Methods

- [[Policy Gradient Methods]]: optimize policy parameters directly
- [[REINFORCE]]: Monte Carlo policy gradient
- Log-ratio trick / score function estimator
- [[Softmax Policy]] for discrete actions, [[Gaussian Policy]] for continuous actions

### L10: Advanced Policy Search

- [[Policy Gradient Theorem]]: replaces returns with $Q$-function
- [[Actor-Critic]]: critic reduces variance of policy gradient
- [[Advantage Function]] / [[Generalized Advantage Estimation\|GAE]]: baseline subtraction
- [[Deterministic Policy Gradient]]: off-policy, no importance sampling needed
- [[Natural Policy Gradient]]: accounts for policy space geometry

### L11: SAC, Decision Transformer, Decision Diffuser

- [[Soft Actor-Critic (SAC)]]: maximum entropy RL, balances exploration and exploitation
- Decision Transformer: sequence modeling approach to RL (uses transformers)
- Decision Diffuser: diffusion models for decision-making

### L12: Model-Based RL

- [[Dyna]]: integrates learning and planning, generates simulated experience
- [[Monte Carlo Tree Search (MCTS)]]: tree search with rollout-based evaluation
- AlphaGo Zero: combines MCTS with deep neural network (policy + value)
- Key questions: How to learn the model? When to update? How to update?

### L13: POMDPs

- Partially Observable MDPs: agent does not observe full state
- Exact methods: full history (not compact), belief states (requires known model), predictive state representations (most compact, learnable)
- Approximate methods: recent observations (easy, loses long-term info), end-to-end learning with RNNs (general but tricky, data-hungry)

---

## Value-Based Methods: MC vs TD

> [!summary] MC vs TD Comparison

| Property | [[Monte Carlo Methods\|MC]] | [[Temporal Difference Learning\|TD]] |
|---|---|---|
| **Bias** | Unbiased | Biased (bootstrapping) |
| **Variance** | High | Low |
| **Episode requirement** | Must wait for episode end | Updates every step |
| **Bootstrapping** | No | Yes |
| **Works in continuing tasks** | No (episodic only) | Yes |
| **Sensitivity to initial values** | Less sensitive | More sensitive |
| **Convergence (tabular)** | Converges to $v_\pi$ | Converges to $v_\pi$ |

---

## On-Policy vs Off-Policy

| Property | On-Policy | Off-Policy |
|---|---|---|
| **Complexity** | Simpler | More complex |
| **Generality** | Specific case | More general |
| **Convergence** | Often converges faster | Often large variance or slow convergence |
| **Data usage** | Only data from current policy | Can reuse data, use data from other sources |
| **Policy type** | Generally needs non-greedy policy | Allows greedy target policy |
| **Examples** | [[SARSA]], MC control | [[Q-Learning]], [[Deep Q-Network (DQN)\|DQN]], [[Deterministic Policy Gradient\|DPG]] |

---

## Evaluation Methods (Value Prediction)

The unified view of evaluation methods along two axes (Sutton and Barto figure):

- **Width** (sampling): from one sample (TD) to all samples (DP)
- **Depth** (bootstrapping): from one-step (TD) to full return (MC)

Key methods: Gradient [[Monte Carlo Methods|MC]], semi-gradient [[TD(0)]], [[Gradient-TD Methods|GTD2]], [[LSTD]]

---

## Control Methods

| | On-Policy | Off-Policy |
|---|---|---|
| **Tabular** | [[SARSA]], [[Monte Carlo Control\|MC Control]] | [[Q-Learning]] |
| **With Approximation** | Semi-gradient [[SARSA]] | [[Deep Q-Network (DQN)\|DQN]], [[Conservative Q-Learning (CQL)\|CQL]] |
| **DP (known model)** | [[Policy Iteration\|Policy Evaluation]] | [[Value Iteration]] |
| **Off-policy with IS** | (SARSA with importance weights) | -- |

---

## Convergence with Function Approximation

> [!warning] This table is critical for the exam. Know which combinations converge and which do not.

### Convergence Guarantees

| Method | Tabular (On/Off) | Linear On-Policy | Nonlinear On-Policy | Linear Off-Policy | Nonlinear Off-Policy |
|---|---|---|---|---|---|
| **Gradient MC** | Yes | Yes (global) | Yes (local) | Yes (global) | Yes (local) |
| **Semi-gradient TD** | Yes | Yes (global) | **No C!** | **No C!** | **No C!** |
| **Gradient TD** (e.g. GTD2) | Yes | Yes (global) | Yes (local) | Yes (global) | Yes (local) |
| **LSTD** | Yes | Yes (global) | N.A. | Yes (global) | N.A. |

Notes:
- \* with appropriate step-size schedule
- \*\* linear convergence assumes features are independent with a single solution
- "No C!" = No convergence guarantee
- "local" = converges to local optimum only (nonlinear case)
- "global" = converges to global optimum

> [!tip] Exam Pattern
> **Semi-gradient TD diverges** in off-policy settings with function approximation. This is the [[Deadly Triad]]: function approximation + bootstrapping + off-policy. Gradient TD methods (GTD2) fix this by using true gradients.

### Convergence to Which Error?

| Method | Error Minimized |
|---|---|
| **Gradient MC** | VE (Value Error): $\overline{\text{VE}}(\mathbf{w}) = \sum_s \mu(s) [v_\pi(s) - \hat{v}_\mathbf{w}(s)]^2$ |
| **Semi-gradient TD** | PBE (Projected Bellman Error) -- when it converges |
| **Gradient TD** (GTD2) | PBE (Projected Bellman Error) |
| **LSTD** | PBE (Projected Bellman Error) |

---

## Errors in Value Function Approximation

> [!summary] Error Hierarchy (Lecture 7)
> - **Value Error (VE)**: $\overline{\text{VE}}(\mathbf{w}) = \| v_\pi - \hat{v}_\mathbf{w} \|_\mu^2$ -- difference between true and estimated value
> - **Bellman Error (BE)**: $\bar{\delta}_\mathbf{w}(s) = \mathbb{E}_{a \sim \pi}[\mathbb{E}_{s'}[R_{t+1} + \gamma \hat{v}_\mathbf{w}(S_{t+1}) - \hat{v}_\mathbf{w}(S_t)] \mid S_t = s]$
> - **TD Error**: $\delta_\mathbf{w}(S_t, A_t, S_{t+1}) = R_{t+1} + \gamma \hat{v}_\mathbf{w}(S_{t+1}) - \hat{v}_\mathbf{w}(S_t)$ -- sample of Bellman error
> - **Projected Bellman Error (PBE)**: $\overline{\text{PBE}}(\mathbf{w}) = \| \Pi(\hat{v}_\mathbf{w} + \bar{\delta}) - \hat{v}_\mathbf{w} \|_\mu^2$ -- BE projected onto representable space

---

## Semi-Gradient Methods: Why "Semi"?

[[Semi-Gradient Methods]] take the gradient only through the estimate $\hat{v}_\mathbf{w}(s)$, **not through the bootstrapping target** $R_{t+1} + \gamma \hat{v}_\mathbf{w}(S_{t+1})$. This means they are not true gradient methods and lack the convergence guarantees of true gradient descent. They converge on-policy with linear function approximation but can diverge off-policy.

---

## Policy-Based Methods

### Taxonomy of Policy Methods

| Category | Methods | Key Idea |
|---|---|---|
| **Actor only** | [[REINFORCE]] (original & v2), Finite differences | Policy gradient without a critic |
| **Actor-Critic** | [[Policy Gradient Theorem\|PGT]]-based [[Actor-Critic]], [[Deterministic Policy Gradient\|DPG]], [[Soft Actor-Critic (SAC)\|SAC]] | Policy gradient with a learned value function |
| **Critic only** | [[Q-Learning]], [[Deep Q-Network (DQN)\|DQN]], [[Conservative Q-Learning (CQL)\|CQL]] | Derive policy from learned value function |
| **Other** | Decision Transformer, Decision Diffuser | Sequence modeling / generative approaches |

### Methods and Action/Policy Types

| Method | Discrete Actions | Continuous Actions | Stochastic Policies | Deterministic Policies |
|---|---|---|---|---|
| **Stochastic PG** | Yes | Yes | Yes (behavior + target) | No |
| **Deterministic PG** | No (no gradients) | Yes | Only behavior policy | Yes (target policy) |
| **Critic-only evaluation** | Yes | Yes | Behavior or target | Only target policy |
| **Critic-only control** | Yes | No (how to extract policy?) | -- | -- |

> [!tip] Key Distinction
> **Stochastic policy gradients** work for both discrete and continuous actions but require stochastic policies. **Deterministic policy gradients** require continuous actions but avoid importance sampling and can learn a true greedy policy.

---

## RL Methods Landscape

The RL methods landscape can be organized along two axes (from Jan Peters):

- **x-axis**: Parametrized (or given) transition model (model-free to model-based)
- **y-axis**: Parametrized $Q$ vs parametrized $\pi$

| | Model-Free | Model-Based |
|---|---|---|
| **Parametrized $Q$** | [[Q-Learning]], [[Deep Q-Network (DQN)\|DQN]], ... | -- |
| **Both $Q$ and $\pi$** | [[Actor-Critic]], [[Soft Actor-Critic (SAC)\|SAC]] | Model-based policy search |
| **Parametrized $\pi$** | [[REINFORCE]] | -- |
| **Planning** | -- | [[Dyna]], AlphaGo, Pure planning |

---

## Model-Based Learning (Lecture 12)

Key questions addressed:
- **Why** do model-based RL? Sample efficiency, ability to plan ahead
- **How to learn** the model? Supervised learning of transitions and rewards
- **When to update** the policy? After each real step, after batches, etc.
- **How to update** the policy? Using simulated experience ([[Dyna]]) or tree search ([[Monte Carlo Tree Search (MCTS)\|MCTS]])
- **AlphaGo Zero**: leverages planning both "ahead of" and "while" acting (MCTS during play, neural network training from self-play)

---

## POMDPs (Lecture 13)

State update functions for internal states in Partially Observable MDPs:

| Method | Type | Compact? | Markovian? | Notes |
|---|---|---|---|---|
| **Full history** | Exact | No | Yes | Trivially Markovian but grows without bound |
| **Belief state** | Exact | Moderate | Yes | Easy to interpret, requires known model |
| **Predictive state** | Exact | Most compact | Yes | Model learnable from data |
| **Recent observation(s)** | Approximate | Yes | No | Easy, but loses long-term dependencies |
| **End-to-end (RNN)** | Approximate | Yes | Learned | General, but RNN training is tricky and data-hungry |

---

## Key Recurring Themes

> [!summary] Themes That Run Through the Entire Course

| Theme | Description | Relevant Lectures |
|---|---|---|
| [[Bias-Variance Trade-off]] | MC is unbiased/high variance; TD is biased/low variance | L3, L4, L5, L6, L10 |
| [[On-Policy vs Off-Policy]] | Learning from own policy vs learning from other data | L3, L4, L7, L8, L10 |
| [[Exploration vs Exploitation]] | Balancing trying new actions vs using known good ones | L1 and throughout |
| Tabular vs Approximation | Exact methods vs generalization with function approximation | L5, L6, L7 |
| Model-free vs Model-based | Learning from experience directly vs learning a model and planning | L3-L11 vs L12 |
| Experimentation & Reproducibility | Proper experimental methodology in RL research | L10 |

---

## Quick Reference: When to Use Which Method

> [!tip] Decision Guide for the Exam

| Situation | Recommended Method | Why |
|---|---|---|
| Known model, small state space | [[Dynamic Programming]] ([[Value Iteration]]) | Exact solution, no sampling needed |
| Unknown model, episodic, small state space | [[Monte Carlo Methods\|MC]] or [[Temporal Difference Learning\|TD]] (tabular) | Simple, guaranteed convergence |
| Unknown model, continuing tasks | [[Temporal Difference Learning\|TD]] methods | MC requires episode end |
| Large/continuous state space | [[Function Approximation]] + TD/MC | Generalization needed |
| Off-policy with function approximation | [[Gradient-TD Methods\|GTD2]] or [[LSTD]] | Avoids [[Deadly Triad]] divergence |
| Continuous action space | [[Policy Gradient Methods]] or [[Deterministic Policy Gradient\|DPG]] | Q-learning cannot easily maximize over continuous actions |
| Need stochastic policy | [[REINFORCE]], [[Actor-Critic]] | Built-in exploration |
| Need deterministic optimal policy | [[Deterministic Policy Gradient\|DPG]] / DDPG | No importance sampling needed |
| Maximum entropy / robust exploration | [[Soft Actor-Critic (SAC)\|SAC]] | Entropy-regularized objective |
| Sample efficiency critical | Model-based ([[Dyna]], [[Monte Carlo Tree Search (MCTS)\|MCTS]]) | Can generate synthetic experience |
| Partial observability | POMDP methods (belief states, RNNs) | Full state not available |
| Offline data only | [[Conservative Q-Learning (CQL)\|CQL]], Decision Transformer | Cannot collect new data |

---

## Other Important Topics

> [!warning] Do Not Forget
> - **Exploration vs exploitation** (throughout the course, especially Lecture 1)
> - **Experimentation, evaluation & reproducibility** (Lecture 10): proper methodology for comparing RL algorithms, statistical significance, hyperparameter sensitivity

---

## References

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
- Lecture slides 1--14, Herke van Hoof, University of Amsterdam.
