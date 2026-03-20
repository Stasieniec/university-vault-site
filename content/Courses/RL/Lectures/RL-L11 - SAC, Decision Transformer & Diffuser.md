---
type: lecture
course: RL
week: 6
lecture: 11
book_sections: []
topics:
  - "[[Soft Actor-Critic (SAC)]]"
  - "[[Maximum Entropy RL]]"
  - "[[Decision Transformer]]"
  - "[[Decision Diffuser]]"
  - "[[Offline Reinforcement Learning]]"
status: complete
---

# RL Lecture 11 - SAC, Decision Transformer & Diffuser

## Overview & Motivation

This lecture has two major parts. First, we wrap up policy gradient methods by introducing **Soft Actor-Critic (SAC)**, a high-performing deep RL method that augments the standard RL objective with an entropy bonus to encourage exploration and robustness. Second, we take a fundamentally different perspective on policy learning: instead of estimating value functions or computing policy gradients, we ask **why not just imitate the good bits of logged trajectories?** This leads to the [[Decision Transformer]] and [[Decision Diffuser]], which recast RL as a supervised sequence modeling or generative modeling problem.

The core tension addressed: value-based methods ([[DQN]], [[DDPG]]) are sample-efficient but brittle; stochastic [[Policy Gradient Methods]] are robust but need on-policy samples. SAC gets the best of both worlds. The Decision Transformer/Diffuser sidestep value estimation entirely.

---

## Part 1: Soft Actor-Critic (SAC)

### Motivation

> [!intuition]
> Three key observations motivate SAC:
> 1. **Off-policy learning** is important for sample efficiency, but maximizing a Q-function directly ([[DQN]], [[DDPG]]) is **brittle** -- small errors in Q can lead to catastrophic policy changes
> 2. **Stochastic policy gradients** (REINFORCE and similar) need **on-policy samples**, making them sample-inefficient
> 3. We want an **off-policy stochastic actor-critic** method that combines the benefits of both approaches

The goal is to develop a method with:
- **Off-policy learning** (sample-efficient, can reuse data from a [[Experience Replay|replay buffer]])
- **Stochastic policies** (robust, good exploration)
- **Stable Q-function optimization** (less brittle than pure Q-maximization)

---

### The Maximum Entropy RL Objective

SAC augments the standard RL objective with an **entropy term** that encourages the policy to remain stochastic:

> [!formula]
> **SAC Augmented Objective**:
> $$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \, H(\pi(\cdot | s_t)) \right]$$
>
> where $H(\pi(\cdot|s))$ is the entropy of the policy at state $s$:
> $$H(\pi(\cdot|s)) = \mathbb{E}_{a \sim \pi} [-\log \pi(a|s)]$$

Expanding the entropy term explicitly:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \, \mathbb{E}_{a' \sim \pi(a'|s_t)} [-\log \pi(a'|s_t)] \right]$$

Which simplifies to:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) - \alpha \log \pi(a_t | s_t) \right]$$

> [!definition]
> **[[Maximum Entropy RL]]**: A framework where the agent maximizes both expected cumulative reward and the entropy of its policy. The entropy bonus acts as a built-in regularizer that prevents premature convergence to deterministic policies.

### The Role of Temperature $\alpha$

The parameter $\alpha$ (temperature) controls the tradeoff between reward maximization and entropy:

| High $\alpha$ | Low $\alpha$ |
|---|---|
| More exploration | More exploitation |
| More random/stochastic policy | More greedy/deterministic policy |
| Prioritizes entropy | Prioritizes reward |

> [!tip]
> The lecture notes assume $\alpha = 1$ for simplicity, since different values of $\alpha$ are equivalent to rescaling the reward $r$. In practice, SAC can **automatically tune** $\alpha$ during training.

### Why Entropy Matters

> [!intuition]
> The entropy objective does more than just encourage exploration at the current state. Because entropy appears inside the sum over all timesteps, the policy is also incentivized to **reach future states where it can maintain high entropy**. This means the agent seeks out states where it has many viable options, leading to more robust behavior.

---

### Soft Policy Iteration (Tabular Case)

Analogous to regular [[Dynamic Programming|policy iteration]], SAC iterates between two steps:
1. **Soft value iteration** (policy evaluation)
2. **Soft policy improvement**

#### Soft Bellman Equation

The soft Bellman operator modifies the standard Bellman equation to include the entropy bonus:

> [!formula]
> **Soft Bellman Operator**:
> $$\mathcal{T}^\pi Q(s, a) = r(s, a) + \gamma \, \mathbb{E}_{s' \sim p} \left[ \mathbb{E}_{a' \sim \pi} \left[ Q(s', a') - \alpha \log \pi(a'|s') \right] \right]$$

This defines the **soft value function**:

$$V_{\text{soft}}(s) = \mathbb{E}_{a \sim \pi} \left[ Q_{\text{soft}}(s, a) - \alpha \log \pi(a|s) \right]$$

So the soft Q-function satisfies:

$$Q_{\text{soft}}(s, a) = r(s, a) + \gamma \, \mathbb{E}_{s'} \left[ V_{\text{soft}}(s') \right]$$

> [!intuition]
> The soft value function $V_{\text{soft}}(s)$ incorporates the entropy bonus: a state is valuable not just because of high expected reward, but also because the agent has many good action choices there (high entropy).

#### Soft Policy Improvement

The soft policy improvement step finds the policy that minimizes the KL divergence to an energy-based policy derived from the current Q-function:

> [!formula]
> **Soft Policy Improvement**:
> $$\pi_{\text{new}} = \arg\min_{\pi' \in \Pi} D_{\text{KL}} \left( \pi'(\cdot|s) \;\Big\|\; \frac{\exp(Q^{\pi_{\text{old}}}(s, \cdot))}{Z^{\pi_{\text{old}}}(s)} \right)$$
>
> where $Z^{\pi_{\text{old}}}(s)$ is a partition function (normalizing constant).

> [!intuition]
> This update says: make the new policy as close as possible to the Boltzmann distribution induced by the current Q-function. Actions with high Q-values get high probability, but the KL divergence constraint keeps the policy spread out (stochastic).

---

### Deep SAC: Scaling Beyond Tabular

In the deep learning setting, we parameterize three networks:
- **Q-networks**: $Q_{w_1}(s, a)$ and $Q_{w_2}(s, a)$ (two Q-networks)
- **Value network**: $V_\psi(s)$ (not strictly necessary, but reduces variance by avoiding a sampling step)
- **Policy network**: $\pi_\phi(a|s)$

All networks are updated with alternating gradient-based updates.

#### Loss Functions

> [!formula]
> **Value Network Loss** (sample from current policy, not replay buffer):
> $$J_V(\psi) = \mathbb{E}_{s_t \sim \mathcal{D}} \left[ \frac{1}{2} \left( V_\psi(s_t) - \mathbb{E}_{a_t \sim \pi_\phi} \left[ Q_\theta(s_t, a_t) - \alpha \log \pi_\phi(a_t|s_t) \right] \right)^2 \right]$$
>
> **Q-Network Loss** (using [[Target Network|target network]] parameters $\bar{\psi}$):
> $$J_Q(\theta) = \mathbb{E}_{(s_t, a_t) \sim \mathcal{D}} \left[ \frac{1}{2} \left( Q_\theta(s_t, a_t) - \left( r(s_t, a_t) + \gamma \, \mathbb{E}_{s_{t+1} \sim p} \left[ V_{\bar{\psi}}(s_{t+1}) \right] \right) \right)^2 \right]$$

#### Policy Loss

The policy gradient is derived from the KL divergence objective:

$$\nabla_\phi J_\pi(\phi) = \nabla_\phi \, \mathbb{E}_{s_t \sim \mathcal{D}} \left[ D_{\text{KL}} \left( \pi_\phi(\cdot|s_t) \;\Big\|\; \frac{\exp(Q_\theta(s_t, \cdot))}{Z_\theta(s_t)} \right) \right]$$

Expanding:

$$= \nabla_\phi \, \mathbb{E}_{s_t \sim \mathcal{D}} \left[ \mathbb{E}_{a \sim \pi_\phi} \left[ \log \pi_\phi(\cdot|s_t) - Q_\theta(s_t, \cdot) + \text{const.} \right] \right]$$

> [!tip]
> The partition function $Z_\theta(s_t)$ is a constant with respect to $\phi$, so it disappears in the gradient. This is a key simplification.

One could compute this gradient with REINFORCE-style estimators, but we can do better using the **reparameterization trick**.

---

### The Reparameterization Trick

> [!definition]
> **[[Reparameterization Trick]]**: Instead of sampling $a \sim \pi_\phi(\cdot|s)$ directly (which blocks gradient flow), we write the action as a deterministic function of the state and an independent noise variable:
> $$a = f_\phi(\epsilon; s), \quad \epsilon \sim \mathcal{N}(0, I)$$
> This allows gradients to flow through the sampling operation.

Using reparameterization, the policy gradient becomes:

> [!formula]
> **Reparameterized Policy Gradient**:
> $$\nabla_\phi J_\pi(\phi) = \mathbb{E}_{s_t \sim \mathcal{D}} \, \mathbb{E}_{\epsilon \sim \mathcal{N}} \Big[ \nabla_\phi \log \pi_\phi(a_s|s_t) + \big( \nabla_{a_t} \log \pi_\phi(a_s|s_t) - \nabla_{a_t} Q_\theta(s_t, a_t) \big) \nabla_\phi f_\phi(\epsilon_t; s_t) \Big]$$

This expression has two interpretable components:
- **Entropy maximization term**: $\nabla_\phi \log \pi_\phi$ -- pushes the policy to increase entropy
- **DDPG-like term**: $\nabla_{a_t} Q_\theta(s_t, a_t) \cdot \nabla_\phi f_\phi(\epsilon_t; s_t)$ -- pushes the policy toward actions with high Q-values, analogous to [[DDPG]]

> [!intuition]
> The reparameterization trick makes SAC's policy update similar to DDPG (using known Q-function gradients) while additionally maximizing entropy. This is much lower variance than REINFORCE-style gradient estimation.

---

### Full SAC Algorithm

> [!formula]
> **SAC Algorithm (Deep Version)**:
> ```
> Initialize: Q_w1, Q_w2, V_psi, pi_phi, target V_psi_bar
> Initialize replay buffer D
>
> for each iteration do:
>     # Environment interaction
>     for each environment step do:
>         a ~ pi_phi(a|s)           # Sample action from policy
>         s' ~ p(s'|s, a)           # Step environment
>         D <- D ∪ {(s, a, r, s')}  # Store transition in replay buffer
>     end for
>
>     # Gradient updates (one or multiple steps)
>     for each gradient step do:
>         Sample minibatch from D
>
>         # Update V network
>         psi <- psi - lambda_V * nabla_psi J_V(psi)
>
>         # Update Q networks (both)
>         w_i <- w_i - lambda_Q * nabla_w_i J_Q(w_i)   for i = 1, 2
>
>         # Update policy
>         phi <- phi - lambda_pi * nabla_phi J_pi(phi)
>
>         # Update target network (soft update)
>         psi_bar <- tau * psi + (1 - tau) * psi_bar
>     end for
> end for
> ```

Key implementation details:
- **Two Q-functions**: $Q_{w_1}$ and $Q_{w_2}$. The **minimum** of the two is used to counter optimistic Q-value overestimation (optimization bias), similar to the twin critics in TD3
- **[[Target Network]]**: Used for stable Q-learning targets
- **[[Experience Replay]]**: Off-policy data is sampled from a replay buffer
- In practice, **1 environment step** is taken per iteration, with one or multiple gradient steps using data from the replay buffer

---

### SAC Results

> [!summary]
> **Empirical Performance**:
> - **More consistent than DDPG**: SAC avoids the instability and brittleness of pure Q-maximization (DDPG results shown in bottom row of comparison plots)
> - **Faster learning than PPO**: The off-policy nature gives SAC better sample efficiency
> - **Automatic temperature scheduling**: The automatically tuned temperature (shown in blue in experiments) works about as well as manually tuned temperature per task (shown in orange)
> - **Real-world robotics**: SAC also works on real-world robotics tasks including walking and manipulation from pixels

### SAC Conclusions

> [!summary]
> **SAC Key Properties**:
> - Off-policy stochastic [[Actor-Critic]] method
> - Entropy-maximizing loss keeps the policy stochastic, which:
>   - Increases **robustness** to perturbations
>   - Improves **exploration**
> - Combines benefits of stochastic policy gradients and Q-function maximization:
>   - **Less-greedy update** (more robust) with stochastic exploration
>   - **Limited maximization of a Q-function** (sample efficient)

---

## Part 2: RL as Supervised Learning

### Motivation: Why Not Just Imitate?

> [!intuition]
> RL approaches based on learning value functions and/or policies tend to be somewhat "fiddly" and don't always work. This is especially true in the **[[Offline Reinforcement Learning|offline RL]]** case, where we learn from a fixed dataset without further environment interaction. (Recall the difficulties with [[Conservative Q-Learning (CQL)]]!)
>
> Meanwhile, transformers and diffusion models trained with supervised learning in language and vision tend to perform **more robustly**. Can we leverage the strength of supervised learning for RL?

The naive approach -- behavioral cloning (just imitate all demonstrations) -- has clear limitations:
- Needs **high-quality** demonstrations
- Cannot in principle do **better** than the demonstrations

But what if we only have **mediocre** demonstrations (e.g., from exploration)?

### Key Idea: Imitate the Good Bits

We could attempt to **only imitate the good trajectories** and/or only the "good bits" of trajectories. This is not a new idea:

- **Reward-weighted regression** and **PoWER** up-weight trajectories with positive returns (but assume linear policies)
- **Upside-down RL** and **reward-conditioned policies** explored conditioning policies on specific desired returns

The breakthrough in the deep learning era: combining **conditioning on (good) rewards** with modern deep learning architectures (transformers, diffusion models):
- [[Decision Transformer]]
- [[Decision Diffuser]]

---

## Decision Transformer

### Core Idea

> [!definition]
> **[[Decision Transformer]]**: Treats RL as a **sequence modeling** problem rather than a value estimation problem. It predicts actions autoregressively as a function of the trajectory so far and the **desired return-to-go**, using a GPT-style transformer architecture.

The key insight: instead of estimating $Q(s,a)$ or $V(s)$ and deriving a policy, directly **predict what action to take** given the trajectory history and a target return level.

### Trajectory Preprocessing

The trajectory is represented as an interleaved sequence of returns-to-go, states, and actions:

> [!formula]
> **Trajectory Representation**:
> $$\tau = (G_0, s_0, a_0, \, G_1, s_1, a_1, \, \ldots, \, G_{T-1}, s_{T-1}, a_{T-1})$$
>
> where $G_t = \sum_{t'=t}^{T} r_{t'}$ is the **return-to-go** (sum of future rewards from timestep $t$ onwards).

> [!intuition]
> The return-to-go $G_t$ tells the model "this is the total reward achieved from this point onwards." By conditioning on a **desired** return-to-go at test time, we can control the performance level of the generated behavior.

### Architecture

- **GPT architecture**: A causally masked transformer (autoregressive)
- **Input**: Return-to-go, state, and action tokens from the last $K$ timesteps (context window)
- **Output**: Predicted next action
- **Training loss**: Cross-entropy (for discrete actions) or mean squared error (for continuous actions)

```
Input tokens:  [G_0] [s_0] [a_0]  [G_1] [s_1] [a_1]  ...  [G_t] [s_t] [?]
                 |     |     |       |     |     |            |     |     |
              [Causal Transformer with positional embeddings]
                                                              |     |     |
Output:                                               predict action a_t
```

### Inference (Test Time)

At test time:
1. Start with initial state $s_0$ and a **desired return-to-go** $G_0$ (e.g., the maximum return seen during training)
2. The model predicts action $a_0$
3. Execute $a_0$, observe $r_0$ and $s_1$
4. Update return-to-go: $G_1 = G_0 - r_0$
5. Feed $(G_0, s_0, a_0, G_1, s_1)$ to predict $a_1$
6. Continue autoregressively

> [!tip]
> By setting the desired return-to-go to a high value, the model generates behavior that achieves high rewards. This is analogous to prompting a language model -- you "prompt" the policy with the performance level you want.

### Results & Discussion

- **Good performance** on several offline RL benchmarks
- **Better than training on only the best trajectories**: Other (mediocre) trajectories help generalization
- **Context helps**: $K=30$ or $K=50$ outperforms $K=1$, suggesting the environment appears non-Markov or that policy history provides useful hints for future actions
- **Robust to sparse reward settings**: Because it directly models returns rather than bootstrapping value estimates
- **No value optimization**: No need for pessimism or regularization (unlike [[Conservative Q-Learning (CQL)|CQL]])

> [!warning]
> The Decision Transformer is primarily designed for **[[Offline Reinforcement Learning|offline RL]]** -- it learns from a fixed dataset of logged trajectories without online interaction with the environment.

---

## Decision Diffuser

### Core Idea

> [!definition]
> **[[Decision Diffuser]]**: Uses a **diffusion model** to generate future state trajectories with desired properties, then derives actions using an inverse dynamics model. Unlike the Decision Transformer, it can condition on **multiple types of guidance** beyond just return: constraints, skills, and their combinations.

### Architecture & Approach

The Decision Diffuser operates in two stages:

1. **Trajectory generation**: A diffusion model generates a future state trajectory $(s_t, s_{t+1}, \ldots, s_{t+H})$
2. **Action extraction**: A separate **inverse dynamics model** $f_\phi$ maps consecutive state pairs to actions: $a_t = f_\phi(s_t, s_{t+1})$

> [!intuition]
> Why generate only states and not actions?
> - In robotics domains, **states tend to be continuous and smooth** (positions, velocities)
> - **Actions can be jerky or discrete** (torques, motor commands)
> - Diffusion models work better on smooth, continuous data
> - The inverse dynamics model handles the state-to-action mapping separately

### Conditioning with Classifier-Free Guidance

The key advantage of the Decision Diffuser is its flexible conditioning mechanism. During the reverse (denoising) process of the diffusion model, **classifier-free guidance** adds a bias toward trajectories with desired properties:

> [!formula]
> **Conditioning Types**:
> - **Maximize return**: Condition the noise prediction on $R = 1$ (the maximum normalized return)
> - **Satisfy constraints**: Condition on constraint identity (one-hot encoding), e.g., $\text{blockheight}(i) > \text{blockheight}(j)$
> - **Compose skills**: Condition on skill identity (one-hot encoding)

With classifier-free guidance, during training the conditioning information is provided with probability $p$ and dropped with probability $1-p$. At inference, the model can interpolate between the conditioned and unconditioned predictions to steer generation.

### Training Details

> [!formula]
> **Training Components**:
> - **Diffusion model** $\epsilon_\theta$: Predicts the noise applied in the forward diffusion process, conditioned on desired properties (with probability $p$)
> - **Inverse dynamics model** $f_\phi$: Predicts actions from consecutive state pairs
>
> **Architecture**:
> - $\epsilon_\theta$: **Temporal U-Net**, with conditioning information projected to a latent vector $z$ via an MLP
> - $f_\phi$: MLP
>
> **Low-temperature sampling**: During denoising at inference time, **reduce the variance of predicted noise** to get more deterministic, high-quality trajectories

> [!intuition]
> Low-temperature sampling is analogous to reducing the temperature in language model generation -- it makes the output more focused and less random, favoring the most likely trajectories.

### Results

- **Offline RL**: Competitive with or outperforms baselines such as behavioral cloning, [[Conservative Q-Learning (CQL)]], and [[Decision Transformer]]
- **Constraints**: Better at satisfying single constraints than baselines, and the **only model that can combine multiple constraints** (e.g., $\text{blockheight}(i) > \text{blockheight}(j)$ AND $\text{blockheight}(k) > \text{blockheight}(l)$)
- **Skills**: Manages to generate behavior somewhat "in between" separately learned skills

---

## Related Work: Decision Transformer / Diffuser Family

Several works are closely related:

| Method | Key Difference |
|---|---|
| **Upside-down RL** | Single state input (no sequence model) |
| **Trajectory Transformer** | Similar to (concurrent with) Decision Transformer, but also uses state and return predictions |
| **Diffuser** (Janner et al.) | Uses **classifier guidance** (gradient of estimated returns) instead of classifier-free guidance; predicts **state-action pairs** instead of states only |

---

## Conclusions on Decision Transformer / Diffuser

> [!summary]
> - **Promising alternative** to methods that estimate values (critic-only or actor-critic)
> - Open question: Do they share weaknesses of methods based on **Monte Carlo returns**? (They rely on observed returns rather than bootstrapped estimates)
> - Mostly aimed at **[[Offline Reinforcement Learning|offline RL]]** -- whether they are equally promising for online RL remains an open question

---

## The Big Picture: RL Methods Landscape

```
                         RL Methods
                             |
             ________________|________________
            |                                 |
      Action-value based                Policy-based
            |                                 |
    SARSA, Q-learning, MC          Policy Gradient Methods
    Gradient MC                   /       |        |       \
    Semi-gradient TD           REINFORCE  PGT  Actor-Critic  DPG/DDPG
    GTD2, etc                    |                |
                              G(PO)MDP           SAC
                                                  |
        Critic only <---------> Actor-Critic <---------> Actor only
                                                  |
                                    Decision Transformer
                                    Decision Diffuser
```

### How to Learn Policies -- Three Paradigms

Given data $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1 \ldots N}$:

1. **Value-based**: Learn $V(s)$ or $Q(s,a)$, derive policy from values
2. **Policy-based**: Directly optimize $\pi(a|s)$ using policy gradients or supervised learning
3. **Model-based**: Learn dynamics $p(s'|s,a)$ and reward $r(s,a)$, then plan or learn value/policy using the model

---

## What You Should Know

> [!summary]
> **Exam-relevant takeaways from this lecture**:
>
> 1. **Soft Actor-Critic**:
>    - Off-policy stochastic actor-critic
>    - How does the chosen loss (entropy-augmented objective) result in a **robust** and **efficient** method?
>    - The entropy term keeps the policy stochastic (robustness, exploration) while Q-function gradients drive efficient learning
>
> 2. **Main properties of policy improvement using supervised learning**:
>    - Can imitate only the good parts of trajectories by conditioning on desired returns
>    - No need for value function bootstrapping, pessimism, or regularization
>
> 3. **Trajectory preprocessing**:
>    - Representing trajectories as sequences of $(G_t, s_t, a_t)$ triples with return-to-go
>
> 4. **Main idea behind Decision Transformer, Decision Diffuser, and their differences**:
>    - Decision Transformer: autoregressive action prediction conditioned on return-to-go
>    - Decision Diffuser: diffusion-based trajectory generation with flexible conditioning (returns, constraints, skills) + inverse dynamics for actions

---

## New Concepts to Explore

The following concepts are introduced or referenced and warrant deeper study:

- [[Soft Actor-Critic (SAC)]] - Off-policy maximum entropy actor-critic method
- [[Maximum Entropy RL]] - Framework augmenting RL with entropy bonuses
- [[Decision Transformer]] - RL via autoregressive sequence modeling conditioned on returns
- [[Decision Diffuser]] - RL via conditional diffusion trajectory generation
- [[Offline Reinforcement Learning]] - Learning from fixed datasets without online interaction
- [[Reparameterization Trick]] - Gradient-friendly sampling via deterministic transformations of noise
- [[Classifier-Free Guidance]] - Conditioning mechanism for diffusion models without a separate classifier
- [[Inverse Dynamics Model]] - Predicting actions from consecutive state pairs
- [[Reward-Weighted Regression]] - Weighting policy updates by trajectory returns
- [[Upside-Down RL]] - Conditioning policies on desired returns (precursor to Decision Transformer)

---

## References

- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML.
- Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., ... & Levine, S. (2018). *Soft Actor-Critic Algorithms and Applications*. arXiv preprint arXiv:1812.05905.
- Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., & Mordatch, I. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling*. NeurIPS.
- Ajay, A., Du, Y., Gupta, A., Tenenbaum, J. B., Jaakkola, T. S., & Agrawal, P. (2023). *Is Conditional Generative Modeling all you need for Decision Making?* ICLR.
- Janner, M., Li, Q., & Levine, S. (2021). *Offline Reinforcement Learning as One Big Sequence Modeling Problem*. NeurIPS.
- Janner, M., Du, Y., Tenenbaum, J. B., & Levine, S. (2022). *Planning with Diffusion for Flexible Behavior Synthesis*. ICML.
- Peters, J., & Schaal, S. (2007). *Reinforcement Learning by Reward-Weighted Regression for Operational Space Control*. ICML.
- Srivastava, R. K., Shyam, P., Muber, F., Elber, M., & Schmidhuber, J. (2019). *Training Agents using Upside-Down Reinforcement Learning*. CoRR.
- Kumar, A., Peng, X. B., & Levine, S. (2019). *Reward-Conditioned Policies*. arXiv preprint arXiv:1912.13465.
