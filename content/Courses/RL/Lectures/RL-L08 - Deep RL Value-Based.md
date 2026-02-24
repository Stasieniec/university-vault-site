---
type: lecture
course: RL
week: 4
lecture: 8
book_sections: ["Ch 16.5"]
topics:
  - "[[Deep Q-Network (DQN)]]"
  - "[[Experience Replay]]"
  - "[[Target Network]]"
  - "[[Conservative Q-Learning (CQL)]]"
  - "[[Offline Reinforcement Learning]]"
  - "[[Neural Network Function Approximation]]"
  - "[[Atari Games]]"
status: complete
---

# RL Lecture 8: Deep RL (Value-Based Methods)

## 1. Why Deep RL?
Traditional Reinforcement Learning (tabular or linear function approximation) faces significant hurdles in complex environments:
- **Curse of Dimensionality**: Tabular methods (e.g., standard Q-learning) cannot scale to high-dimensional state spaces (like pixels).
- **Manual Feature Engineering**: Linear function approximation requires expert-designed features, which are often task-specific and brittle.
- **Limited Representational Power**: Linear models cannot capture the non-linear relationships often required for complex tasks (e.g., visual perception).

**Transition to Deep RL:**
Deep Reinforcement Learning replaces manual feature engineering with **Representation Learning**. By using deep neural networks (especially CNNs), agents can learn features directly from raw input (pixels) end-to-end.

---

## 2. Deep Q-Network (DQN)
Introduced by Mnih et al. (2015), **DQN** was the first algorithm to achieve human-level performance across a diverse range of tasks (Atari 2600 games) using the same architecture and hyperparameters.

### 2.1 Pre-processing
To handle raw pixels efficiently, DQN applies several task-agnostic steps:
1.  **Downscaling & Grayscale**: Images are reduced to $84 \times 84$ resolution and converted to grayscale to save memory.
2.  **Frame Stacking**: The agent receives a stack of the **4 most recent frames** as input. This provides a "short memory" allowed the agent to infer velocity and direction (e.g., whether a ball is moving up or down).
3.  **Reward Clipping**: Rewards are clipped to $[-1, 0, 1]$ to stabilize the gradients across different games.

### 2.2 Architecture
The DQN architecture is a deep Convolutional Neural Network (CNN):
- **Input**: $84 \times 84 \times 4$ (pre-processed frames).
- **Conv Layers**: Three convolutional layers with **ReLU** activations to extract spatial features.
- **Fully Connected Layers**: One or more FC layers to map features to action values.
- **Output**: A separate output for **each possible action**. This layout is more efficient than taking an action as input, as it computes all action values in a single forward pass.

![DQN Architecture Illustration|600](https://miro.medium.com/max/1400/1*uCCv9oP6U5S8y7vOth2kZw.png)
> [!info] **Figure 1: DQN Convolutional Architecture**
> The network takes the 4 most recent $84 \times 84$ frames as input. It consists of three convolutional layers (extracting features like ball positions and motion) followed by fully connected layers that output a $Q$-value for each action. Note the output layout: all actions are computed in parallel.

### 2.3 Key Techniques for Stability
Training deep neural networks with Q-learning is notoriously unstable. DQN solves this using two main techniques:

#### [[Experience Replay]]
- **Mechanism**: Transitions $(S_t, A_t, R_{t+1}, S_{t+1})$ are stored in a large buffer (circular queue).
- **Training**: At each step, a small **random minibatch** is sampled from the buffer for the update.
- **Why it helps**:
  1.  **Breaks Correlations**: Successive transitions in an episode are highly correlated. Randomized sampling makes the data look more like the i.i.d. data used in supervised learning.
  2.  **Data Efficiency**: Experiences are reused multiple times for learning.

#### [[Target Network]]
- **Mechanism**: DQN maintains two networks: the **Online Network** ($w$) and the **Target Network** ($\tilde{w}$).
- **Update**: The target network weights are kept fixed and only synchronized with the online network every $C$ steps ($\tilde{w} \gets w$).
- **Bellman Update**:
  $$y_j = R_j + \gamma \max_{a'} \hat{q}(S'_{j}, a', \tilde{w})$$
- **Why it helps**: In standard Q-learning, the target changes with every weight update, leading to feedback loops and oscillations. A fixed target provides a stable "ground truth" to move towards.

---

## 3. The DQN Algorithm
The loss function for DQN is the mean squared error (MSE) of the Bellman residual:
$$L(w) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( R + \gamma \max_{a'} \hat{q}(s', a', \tilde{w}) - \hat{q}(s, a, w) \right)^2 \right]$$

> [!abstract] **Algorithm: DQN with Experience Replay**
> 1.  Initialize replay memory $D$ to capacity $N$
> 2.  Initialize action-value function $Q$ with random weights $\theta$
> 3.  Initialize target action-value function $\hat{Q}$ with weights $\theta^- = \theta$
> 4.  **For** episode = 1 to $M$ **do**:
>     1. Initialize state $s_1$ and pre-process $\phi_1 = \phi(s_1)$
>     2. **For** $t = 1$ to $T$ **do**:
>        1. With probability $\epsilon$ select random action $a_t$, else $a_t = \arg\max_a Q(\phi_t, a; \theta)$
>        2. Execute $a_t$, observe reward $r_t$ and image $x_{t+1}$
>        3. Pre-process $\phi_{t+1} = \phi(s_t, a_t, x_{t+1})$
>        4. Store transition $(\phi_t, a_t, r_t, \phi_{t+1})$ in $D$
>        5. Sample minibatch $(\phi_j, a_j, r_j, \phi_{j+1})$ from $D$
>        6. Set $y_j = r_j + \gamma \max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^-)$ (if not terminal)
>        7. Perform gradient descent step on $(y_j - Q(\phi_j, a_j; \theta))^2$
>        8. Every $C$ steps, set $\theta^- \gets \theta$

---

## 4. DQN Extensions
Several institutional improvements have been proposed:
- **Double DQN**: Mitigates overestimation bias by decoupling action selection from evaluation.
- **Prioritized Experience Replay**: Samples transitions with higher TD error more frequently.
- **Dueling DQN**: Splits the network into $V(s)$ and $A(s, a)$.
- **Rainbow DQN**: Combines all-of-the-above (plus multi-step, distributional, and noisy layers) for drastically better performance.

![Rainbow Convergence|500](https://miro.medium.com/v2/resize:fit:1400/1*C687W_A2u2kS8-7Xp8x5jw.png)
> [!info] **Figure 2: The "Rainbow" of Improvements**
> Learning curves show that combining individual "tricks" (Double DQN, Dueling, etc.) leads to significantly faster and more stable convergence compared to vanilla DQN.

---

## 5. Atari Results & Significance
- **Human-Level Performance**: DQN outperformed humans on 22/49 games. 
- **Significance**: Proved that a single architecture could learn diverse skills (reflexes in *Breakout*, precision in *Space Invaders*, management in *Seaquest*) solely from pixels and score.
- **Failure Cases**: Struggles on games requiring long-term planning/exploration (e.g., *Montezuma's Revenge*).

---

## 6. Offline Reinforcement Learning
**Offline RL** is the task of learning a policy from a **fixed dataset** $D$ collected by some behavior policy $\beta$, without further interaction.

### 6.1 The Challenge: Distribution Shift
Standard Q-learning fails because the agent evaluates actions that are Out-of-Distribution (**OOD**).

> [!example] **Visualization of the Problem**
> Imagine an MDP with three actions:
> 1. $a_1$ (Well-seen): Real Q=10, Estimated Q=10.5
> 2. $a_2$ (Rarely seen): Real Q=8, Estimated Q=5
> 3. $a_3$ (Unseen/OOD): Real Q=8, **Estimated Q=11** (Error due to FA)
>
> Standard Q-learning will calculate $\max_a Q(s, a)$ and pick action $a_3$, even though it is sub-optimal. This overestimated error propagates through the Bellman backup, causing the value function to "explode."

---

## 7. Conservative Q-Learning (CQL)
Proposed by Kumar et al. (2020), **CQL** addresses this by learning a **conservative** (pessimistic) $Q$-function.

### 7.1 Key Idea: Expected Pessimism
Instead of point-wise guarantees, CQL ensures that the **expected** $Q$-value under the learned policy is a lower bound. 
$$V^\pi_{learned}(s) \le V^\pi_{true}(s) \text{ (with high probability)}$$

### 7.2 The CQL Loss Function
CQL adds a regularizer that penalizes $Q$-values for actions where $\pi(a|s) > \beta(a|s)$:
$$\min_Q \alpha \left( \mathbb{E}_{s \sim D, a \sim \pi(a|s)}[Q(s,a)] - \mathbb{E}_{s \sim D, a \sim \beta(a|s)}[Q(s,a)] \right) + \text{Loss}_{Bellman}$$

- **Practical Implementation**: For discrete actions, the first term is implemented using a `logsumexp` over all actions.
- **Result**: Proved to guarantee under-estimation, which prevents the policy from "tripping" over OOD action errors. Experiments show CQL performs significantly better than other offline methods on small, noisy datasets.

![CQL Underestimation Table|600](https://miro.medium.com/v2/resize:fit:1200/1*dI6qY_fU6sKByvDqY7lEfg.png)
> *Figure: CQL empirically shows lower (conservative) values compared to standard Q-learning or ensembles, which tend to diverge.*

---
*Reference: Mnih et al. (2015), Kumar et al. (2020), Sutton & Barto (2018) Ch 16.5.*
