---
type: book-chapter
course: RL
book: "Reinforcement Learning: An Introduction (2nd ed.)"
chapter: 16
sections: ["16.1", "16.2", "16.3", "16.4", "16.5", "16.6", "16.7", "16.8"]
topics:
  - "[[Deep Q-Network (DQN)]]"
  - "[[Experience Replay]]"
  - "[[Target Network]]"
  - "[[Monte Carlo Tree Search (MCTS)]]"
  - "[[Deep Reinforcement Learning]]"
  - "[[Convolutional Neural Networks]]"
status: complete
---

# Chapter 16: Applications and Case Studies

## Overview
Chapter 16 presents a series of reinforcement learning applications, ranging from historical milestones to modern superhuman systems. It illustrates the transition from classical methods to **[[Deep Reinforcement Learning]]**, emphasizing how domain knowledge is incorporated and how representation issues are critical to success.

---

## 16.1 TD-Gammon (Tesauro, 1992-1995)
One of the most impressive early applications. TD-Gammon used **[[Temporal Difference Learning]]** (specifically $TD(\lambda)$) with a non-linear multi-layer neural network trained by backpropagating **[[TD Error]]**.

- **Key Feature**: Self-play. It learned to play by playing against itself, starting from random weights.
- **Success**: Reached world-class levels and changed how human Grandmasters play certain opening positions.

---

## 16.2 Samuel's Checkers Player (1950s-60s)
A seminal precursor that used heuristic search and a form of TD learning.
- **Method**: Lookahead search with a scoring polynomial (linear function approximation).
- **Rote Learning**: Stored board positions and their backed-up values to effectively increase search depth.
- **Learning by Generalization**: Updated weights toward the minimax value of a search.

---

## 16.3 Watson's Daily-Double Wagering
IBM Watson used RL for its wagering strategy in *Jeopardy!*.
- **Mechanism**: Compared action values $q(s, \text{bet})$ estimating the probability of a win.
- **Computation**: 
  $$q(s, \text{bet}) = p_{DD} \times \hat{v}(S_W + \text{bet}, \dots) + (1 - p_{DD}) \times \hat{v}(S_W - \text{bet}, \dots)$$
- **Note**: Used models of human opponents rather than self-play due to the asymmetric nature of the game and imperfect information.

---

## 16.4 Optimizing Memory Control
RL applied to scheduling DRAM commands (precharge, activate, read, write).
- **Agent**: Used **Sarsa** with linear **Function Approximation** (tile coding).
- **Reward**: +1 for read/write, 0 otherwise (Objective: maximize throughput).
- **Result**: Significant latency reduction and improved execution speed by adapting to workload patterns online.

---

## 16.5 Human-level Video Game Play (Atari/DQN)
> [!IMPORTANT] 
> This section marks the breakthrough of **[[Deep Q-Network (DQN)]]**, demonstrating that a single architecture can learn different tasks directly from raw pixels.

### The DQN Architecture
DQN combines Q-learning with a **[[Convolutional Neural Networks]]** (CNN).
- **Input**: "Raw" $84 \times 84 \times 4$ image stacks (luminance) from the game emulator.
- **Agent**: A deep CNN that outputs estimated optimal action values $Q^*(s,a)$.

### Stabilizing Deep RL
Deep RL is notoriously unstable. DQN introduced two critical mechanisms to solve this:

1. **[[Experience Replay]]**:
   - Stores transitions $(s_t, a_t, r_{t+1}, s_{t+1})$ in a replay buffer.
   - Updates are performed on mini-batches sampled uniformly at random.
   - **Benefit**: Breaks temporal correlations and improves data efficiency.

2. **[[Target Network]]**:
   - Uses a separate, slowly-syncing network to compute the target $Q$-values.
   - **Update Rule**:
     $$w_{t+1} = w_t + \alpha \left[ R_{t+1} + \gamma \max_a \tilde{q}(S_{t+1}, a, w_t^-) - \hat{q}(S_t, A_t, w_t) \right] \nabla \hat{q}(S_t, A_t, w_t)$$
   - where $w_t^-$ are the weights of the target network (syncing with $w_t$ every $C$ steps).

> [!FORMULA] **DQN Loss Function**
> The network is trained by minimizing the mean-squared error of the Bellman residual:
> $$L_i(\theta_i) = \mathbb{E}_{s,a,r,s'} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_i^-) - Q(s, a; \theta_i) \right)^2 \right]$$

---

## 16.6 Mastering the Game of Go (AlphaGo & AlphaGo Zero)
> [!IMPORTANT]
> Go was long considered the "Holy Grail" of AI due to its high branching factor ($\approx 250$) and lack of a simple evaluation function.

### AlphaGo (2016)
Combined Supervised Learning (SL), Reinforcement Learning (RL), and **[[Monte Carlo Tree Search (MCTS)]]**.

- **Policy Networks**: Deep CNNs trained to predict human moves (SL) then improved via **[[Policy Gradient Methods]]** (RL).
- **Value Network**: Trained to predict the winner of games played by the RL policy.
- **APV-MCTS**: Asynchronous Policy and Value MCTS.
  - **Node Evaluation**: $v(s) = (1 - \eta)v_\theta(s) + \eta G$ (Mixing value network $v_\theta$ and rollout return $G$).

### AlphaGo Zero (2017)
The "Zero" represents **zero human data**. It learned exclusively through **Self-Play**.

- **Architecture**: A single "two-headed" residual CNN that outputs both move probabilities $p$ and position value $v$.
- **MCTS as Policy Improvement**: MCTS is treated as a powerful policy improvement operator. The search probabilities $\pi$ are used as targets for the policy head.
- **Differences from AlphaGo**:
  - No rollouts (only the value head of the network).
  - No human features (only raw stone placements).
  - Single network for policy and value.

> [!FORMULA] **AlphaGo Zero Training Objective**
> The network parameters $\theta$ are updated to minimize the loss:
> $$(p, v) = f_\theta(s)$$
> $$L = (z - v)^2 - \pi^\top \log p + c \|\theta\|^2$$
> Where $z$ is the winner of the game (+1 or -1), $\pi$ are search probabilities from MCTS, and $c$ is a regularization constant.

---

## 16.7 Personalized Web Services
- **Problem**: Contextual Bandits (associative RL) vs. MDP formulation.
- **Life-Time Value (LTV)**: RL agents prioritize sequences of interactions (funnels) rather than immediate click-through rates (CTR).
- **Evaluation**: Used **Off-policy evaluation** to provide high-confidence performance guarantees before deployment.

---

## 16.8 Thermal Soaring
- **Task**: Glider soaring in turbulent air.
- **Observation**: Vertical wind acceleration and torques (gradient info) were more critical than wind velocity for staying within thermals.
- **Algorithm**: Sarsa with **Function Approximation**.

---

## Summary
- **Representation is key**: The shift from hand-crafted features (Samuel, TD-Gammon 1.0) to learned features (DQN, AlphaGo Zero) represents the "Deep RL" revolution.
- **Stability Mechanisms**: Experience Replay and Target Networks are fundamental for training deep non-linear approximators.
- **MCTS + Deep Learning**: The combination of decision-time planning (search) and learned value/policy functions (intuition) is currently the most successful recipe for complex zero-sum games.
