---
type: lecture
course: RL
week: 6
lecture: 12
book_sections: ["Ch 8.1", "Ch 8.2", "Ch 8.8", "Ch 8.10", "Ch 8.11", "Ch 8.13", "Ch 16.6"]
topics:
  - "[[Model-Based Reinforcement Learning]]"
  - "[[Dyna]]"
  - "[[Monte Carlo Tree Search (MCTS)]]"
  - "[[AlphaGo Zero]]"
  - "[[Rollout Algorithm]]"
status: complete
---

# RL Lecture 12 - Model-Based Reinforcement Learning

## Overview & Motivation

This lecture introduces the third major approach to learning policies in RL: **learning a dynamics model** of the environment and using it for planning. We have previously seen value-based methods (learn $V(s)$ or $Q(s,a)$) and policy-based methods (directly optimize $\pi(a|s)$). Model-based RL instead learns the transition dynamics $p(s'|s,a)$ and reward function $r(s,a)$, then uses this learned model as a cheap simulator to derive a policy through planning.

The core motivation is **sample efficiency**: real-world data is often expensive, risky, or time-consuming to obtain. A learned model can serve as a proxy for the real system, enabling the agent to generate unlimited simulated experience for planning.

> [!intuition]
> Think of model-based RL as "learning the rules of the game" rather than just "learning what to do." Once you understand how the world works, you can reason about consequences without actually taking actions.

---

## Big Picture: Three Approaches to Learning Policies

Given a dataset of experience $D = \{(s_i, a_i, r_i, s'_i)\}_{i=1 \dots N}$, there are three fundamental approaches:

| Approach | What is Learned | Category |
|---|---|---|
| **Value-based** | $V(s)$, $Q(s,a)$ | Model-free RL |
| **Policy-based** | $\pi(a \mid s)$ directly | Model-free RL |
| **Model-based** | $p(s' \mid s, a)$, $r(s,a)$ then derive $V$, $Q$, or $\pi$ | Model-based RL |

Previously covered methods fall into two categories:
- **Planning methods** that require knowledge of the MDP (e.g., [[Dynamic Programming]]: policy iteration, value iteration)
- **Learning methods** that directly learn value functions or policies from data (Monte Carlo, TD, policy gradient -- all model-free)

Model-based RL bridges these: it **learns** the model from data, then uses **planning** with that learned model.

> [!definition]
> **Planning** (in this lecture): Any process that uses a model (learned or given) rather than real data to obtain or improve a policy.

---

## Why Use Dynamics Models?

A learned dynamics model approximates the real system:

$$s, a \xrightarrow{\text{Real system}} s', r \quad \approx \quad s, a \xrightarrow{\text{Dynamics model}} \hat{s}', \hat{r}$$

The learned model serves as a **cheap proxy** for the real environment.

**Dynamics models are useful when:**
- Real data is **limited, time-consuming, or expensive** to obtain (e.g., robotics, autonomous driving)
- You want access to **internal gradients** or **probability distributions** of the dynamics
- You want to reason about **counterfactuals** (what would have happened under a different action)

> [!warning]
> If real data is plentiful and cheap, model-free techniques are often better. Models introduce **modelling errors** that can compound and lead to poor policies. Only use model-based RL when the benefits of sample efficiency outweigh the cost of model inaccuracy.

### Trade-offs

- **Model-based methods** make fuller use of experience: **lower sample complexity**
- **Model-free methods** are simpler and **not affected by modelling errors**
- The two approaches **can be combined** (as in Dyna)

---

## Model Learning

### The Learning Problem

Given experience data $D = \{(s_i, a_i, r_i, s'_i)\}$, learn:
- **Transition model**: $\hat{p}(s' | s, a)$ -- predicting the next state given current state and action
- **Reward model**: $\hat{r}(s, a)$ -- predicting the reward for a state-action pair

This is fundamentally a **supervised learning** problem: the inputs are $(s, a)$ and the targets are $(s', r)$.

### Tabular Model Learning

In the simplest case (deterministic environments with discrete states), the model is a lookup table:

| action \ state | A | B |
|---|---|---|
| right | B, 0 | G\*, +1 |
| left | \*\* | \*\* |

\*G indicates goal/terminal state; \*\*Unvisited state-action pairs are undefined.

For example, given experience trajectory $\tau = [A, \text{right}, 0, B, \text{right}, +1]$, the model stores the observed transitions directly.

**Assumption**: Transitions are deterministic (each $(s,a)$ maps to exactly one $s'$).

### Alternatives for Model Learning

For **stochastic** or **large/continuous** environments:
- **Count-based**: Learn transition probabilities as the fraction of observed transitions for each $(s,a)$ pair
- **Function approximation**: Use neural networks, Gaussian processes, etc.
  - **Discrete states**: Train with a classification loss
  - **Continuous states**: Train with a regression loss (e.g., MSE)

---

## Using the Model: Background vs. Decision-Time Planning

Once a model is learned, there are two fundamentally different ways to use it:

### Background Planning

> [!definition]
> **Background planning**: Use the model to learn a good policy for *any* state, ahead of acting in the world. Once you have the policy, use it to act without further planning.

- Plan "ahead of" acting
- Updates a global value function or policy using simulated experience
- Example: **[[Dyna]]-Q**

### Decision-Time Planning

> [!definition]
> **Decision-time planning**: Plan from the *current state* at the time a decision must be made. Only consider the part of the state space relevant to the current situation.

- Plan "while" acting
- Think of playing chess: you reason ahead from your current position, not from every possible position
- You make or refine a plan at every turn
- Examples: **[[Rollout Algorithm]]**, **[[Monte Carlo Tree Search (MCTS)]]**

> [!tip]
> Decision-time planning is the approach used in AlphaGo. It can be used with both learned and given transition models.

---

## Dyna-Q: Background Planning with a Learned Model

### Core Idea

[[Dyna]]-Q integrates learning, planning, and acting in a single architecture. After each real interaction with the environment, the agent:
1. Updates its value function using the real experience (direct RL)
2. Updates its model using the real experience (model learning)
3. Performs $n$ additional planning steps using simulated experience from the model

> [!intuition]
> Dyna-Q is conceptually similar to [[Experience Replay]]: both reuse past experience to perform additional updates. The difference is that experience replay stores real transitions, while Dyna generates *new* simulated transitions from a learned model.

### Dyna-Q Algorithm

> [!formula]
> **Dyna-Q Algorithm**:
>
> Initialize $Q(s,a)$ and $\text{Model}(s,a)$ for all $s, a$
>
> Loop forever:
> 1. $s \leftarrow$ current state
> 2. $a \leftarrow \epsilon\text{-greedy}(s, Q)$
> 3. Execute $a$, observe $r, s'$
> 4. **Direct RL**: $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
> 5. **Model learning**: $\text{Model}(s,a) \leftarrow (s', r)$ (assuming deterministic)
> 6. **Planning**: Repeat $n$ times:
>    - $s_{\text{sim}} \leftarrow$ random previously observed state
>    - $a_{\text{sim}} \leftarrow$ random action previously taken in $s_{\text{sim}}$
>    - $s'_{\text{sim}}, r_{\text{sim}} \leftarrow \text{Model}(s_{\text{sim}}, a_{\text{sim}})$
>    - $Q(s_{\text{sim}}, a_{\text{sim}}) \leftarrow Q(s_{\text{sim}}, a_{\text{sim}}) + \alpha [r_{\text{sim}} + \gamma \max_{a'} Q(s'_{\text{sim}}, a') - Q(s_{\text{sim}}, a_{\text{sim}})]$

Step 4 is standard [[Q-Learning]]. Steps 5--6 are the model-based additions. The planning steps (step 6) use the same Q-learning update rule but applied to simulated transitions generated by the model.

### Dyna-Q Example

Consider a simple two-state MDP with states $A$ and $B$:
- $A \xrightarrow{\text{right}, r=0} B \xrightarrow{\text{right}, r=+1} G$ (terminal)

After the first real episode, the model stores these transitions. During subsequent episodes, the agent can perform planning updates on these stored transitions, effectively gaining additional "experience" without interacting with the real environment.

### Performance Characteristics

- Comparisons between Q-learning and Dyna-Q are made for **equal amounts of real experience**
- Dyna-Q learns faster (in terms of real samples) because it extracts more from each interaction
- Model-based RL typically takes **more compute time** for the same amount of real experience:
  - Time to learn the model
  - Time to update the policy using simulated samples
- **If real samples are expensive**, model-based is usually better
- **In terms of total updates** (real + simulated), there is no consistent winner

### Three Key Design Questions

The Dyna architecture raises three fundamental questions:

1. **How to learn the model?** (tabular, function approximation, etc.)
2. **When to plan?** (background planning vs. decision-time planning)
3. **How to plan and update?** (which RL algorithm to apply to simulated data)

### How to Plan and Update

Several approaches:
- Like Dyna-Q: generate samples from the model, then apply any model-free method ([[Q-Learning]], SARSA, policy gradient, etc.)
- Use model-provided **gradients** or **distributions** for more powerful updates (e.g., value-iteration-like updates)
- More extreme: full [[Dynamic Programming]] or other planners to find the optimal value function or policy

> [!warning]
> **Don't trust the model too much** -- it likely has errors! Model errors compound, especially in long rollouts. Using simulated data uncritically can lead to policies that exploit model inaccuracies rather than solving the real task.

---

## The Game of Go: Motivation for Decision-Time Planning

Before introducing rollout algorithms and MCTS, consider the game of Go:

- Place 1 stone per turn (or pass); surround opponent stones to capture territory
- **Branching factor** $b = 250$, **depth** $d = 250$, giving $b^d \approx 10^{170}$ possible game states
- Compare with chess: $b = 35$, $d = 80$, $b^d \approx 10^{80}$
- The key difficulty is **evaluating positions** -- determining who is winning from a given board state
- The board is non-Markov: need (short) history

The **game tree** represents all possible futures. A naive approach (dynamic programming from leaves) is computationally infeasible due to the tree's enormous size ($10^{170}$).

---

## Rollout Algorithms (Book 8.10)

### Core Idea

A [[Rollout Algorithm]] estimates the value of each action from the current state by simulating many trajectories ("rollouts") using a **rollout policy** $b$ (which can be random or a simple heuristic).

### Algorithm

> [!formula]
> **Rollout Algorithm**:
>
> Play $K$ games/rollouts using rollout policy $b$ from state $s_t$
>
> For each rollout $i$, observe trajectory $\tau_i = (s_t, a, s_{t+1}, \dots)$ and outcome $v_i \in \{-1, +1\}$ (or real-valued return)
>
> Estimate action values:
> $$\hat{q}_b(s_t, a) = \frac{\sum_{i: \tau_i = (s_t, a, \dots)} v_i}{\sum_{i: \tau_i = (s_t, a, \dots)} 1} = \text{mean}(\text{value for trajectories starting with } s_t, a)$$
>
> Select action: $a^* = \arg\max_a \hat{q}_b(s_t, a)$

### Pseudocode

```
Initialize s_0
For each timestep t:
    For each planning step k:
        Select an action a (e.g., uniformly at random)
        Perform a rollout starting at (s_t, a) using policy b
    Estimate q_b(s_t, a) from the rollouts for each possible a
    Execute a* = argmax_a q_b(s_t, a) and observe s_{t+1}   # only this action is really executed!
```

### Properties

- With enough rollouts, selecting $a^* = \arg\max_a \hat{q}_b(s_t, a)$ is **at least as good** as following $b$ directly (policy improvement)
- But in general, **not the same as the optimal policy** -- only approximate
- The approximation improves with more rollouts

### Limitations

- Rollout policy $b$ can be very bad -- could we use experience to choose better actions?
- Starts from scratch after each action is selected and executed -- could we re-use relevant trajectories?

These limitations motivate **Monte Carlo Tree Search**.

---

## Monte Carlo Tree Search (MCTS) (Book 8.11)

### Core Idea

[[Monte Carlo Tree Search (MCTS)]] addresses the limitations of rollout algorithms by:
- Building only a **small but most relevant** subtree of the full game tree
- Doing so **incrementally** through repeated simulations
- **Re-using** information from previous simulations via an expanding search tree

MCTS performs "look-ahead search" -- reasoning about consequences ("what if?") using simulation.

### The Four Phases of MCTS

Each MCTS iteration consists of four phases:

#### 1. Selection

Follow the **tree policy** $\pi_{\text{tree}}$ through the existing tree from the root to a leaf node.

The tree policy balances **exploitation** (choosing actions that look good) with **exploration** (trying less-visited actions). A common choice is **UCB1** (Upper Confidence Bound, Chang et al., 2005):

> [!formula]
> **UCB1 Tree Policy**:
> $$\pi_{\text{tree}}(s) = \arg\max_a \left[ Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}} \right]$$
>
> where:
> - $N(s)$ = number of visits to node/state $s$
> - $N(s, a)$ = number of times action $a$ was selected in state $s$
> - $Q(s, a)$ = estimated value of taking action $a$ in state $s$
> - $c$ = exploration constant controlling the exploration-exploitation tradeoff

This is the same [[Upper Confidence Bound]] principle used in multi-armed bandits, applied to each node in the tree.

#### 2. Expansion

- **Expand** the leaf node by adding one (or all) child nodes to the tree
- This grows the tree incrementally, focusing on the most promising/visited regions
- Why expand? To increase the resolution of the search in promising areas

#### 3. Simulation (Rollout)

- From the newly expanded node, perform a **rollout** using a simple rollout policy $b$ (e.g., random) until the end of the game/episode
- Observe the reward/outcome $v$ (e.g., $v \in \{-1, +1\}$ for win/loss)

#### 4. Backup

- **Update values** for all state-action pairs along the path from the expanded node back to the root
- For each $(s, a)$ in the trajectory from root to the leaf:
  - $W(s, a) \leftarrow W(s, a) + v$ (accumulate total value; note: $+v$ or $-v$ depending on which player)
  - $N(s, a) \leftarrow N(s, a) + 1$ (increment visit count)
  - $Q(s, a) = \frac{W(s, a)}{N(s, a)}$ (update mean value)

### MCTS Algorithm

> [!formula]
> **MCTS Algorithm**:
>
> ```
> Initialize s_0
> For each timestep t:
>     For each planning step k (within budget):
>         Perform SELECTION with π_tree to traverse tree to a leaf
>         Perform EXPANSION to add new node(s)
>         Perform SIMULATION (rollout) from new node with policy b
>         Perform BACKUP to update Q for all ancestors
>     Execute a* = π_final(s_t) and observe s_{t+1}   # only this action is really executed!
> ```

### Final Action Selection

After the planning budget is exhausted, select the action to actually execute using $\pi_{\text{final}}$. Common choices:

- **Pick highest $Q(s,a)$**: Can be sensitive to outliers
- **Pick highest $N(s,a)$**: More robust; the most-visited action is typically the most promising
- **Softmax/proportional**: $\pi_{\text{final}}(a|s) \propto N(s, a)^{1/\tau}$ where $\tau$ is a temperature parameter

> [!tip]
> Selecting the most-visited action $N(s,a)$ is often preferred over highest $Q(s,a)$ because visit counts are more robust to noisy value estimates and outlier rollouts.

### Key Properties

- With high enough budget, $Q$ evaluates a policy that can be **much better than $b$**
- The final chosen action can be seen as a **policy improvement** over that evaluated policy
- After executing an action and observing the next state, **parts of the tree can be reused** (the subtree rooted at the chosen child becomes the new root)

> [!intuition]
> MCTS is "all planning" -- it reasons about the future without executing actions on the real game board. Only after the planning budget runs out does it commit to a single action. This is pure decision-time planning.

---

## AlphaGo Zero: MCTS + Neural Networks (Book 16.6)

### Motivation

Standard MCTS does not use neural networks. The rollout policy $b$ can be random, and the tree policy uses UCB1 without any learned prior. **[[AlphaGo Zero]]** (Silver et al., 2017) enhances MCTS with neural networks that guide the search.

### Neural Network Roles

A neural network $f_\theta$ (with parameters $\theta$) provides two outputs:
1. **Policy network** $\pi_\theta(a|s)$: A prior distribution over actions, used to **guide tree search** (limiting the **width** of the search)
2. **Value network** $V_\theta(s)$: An evaluation of leaf nodes, used to **replace or augment rollouts** (limiting the **depth** of the search)

> [!intuition]
> Without neural networks, MCTS must search broadly (many actions) and deeply (long rollouts to terminal states). The policy network tells the tree where to look (reducing width), and the value network tells it when to stop looking (reducing depth).

### Modified Selection: Adding a Prior Policy

In AlphaGo Zero, the UCB selection formula is modified to incorporate the neural network policy as a prior:

> [!formula]
> **AlphaGo Zero Modified UCB**:
> $$\pi_{\text{tree}}(s) = \arg\max_a \left[ Q(s, a) + c \cdot \pi_\theta(a|s) \cdot \frac{\sqrt{N(s)}}{N(s, a) + 1} \right]$$
>
> where:
> - $Q(s, a)$ = mean action value from tree search
> - $\pi_\theta(a|s)$ = neural network policy prior
> - $N(s)$ = visit count of parent node
> - $N(s, a)$ = visit count of child (state-action pair)
> - $c$ = exploration constant

The key difference from standard UCB1: the exploration bonus is **weighted by the neural network policy prior** $\pi_\theta(a|s)$. Actions that the neural network considers promising receive a larger exploration bonus, directing the search toward more promising branches.

### Leaf Evaluation

Instead of performing a full rollout to the end of the game, AlphaGo Zero uses the **value network** $V_\theta(s)$ to evaluate leaf nodes directly. This dramatically reduces computation per MCTS iteration.

### Architecture: Two Networks or One?

AlphaGo Zero uses a **single neural network** with two output heads:
- One head outputs the policy $\pi_\theta(a|s)$
- One head outputs the value $V_\theta(s)$

The shared body allows features learned for one task to benefit the other.

### Training the Neural Networks

> [!formula]
> **AlphaGo Zero Training**:
>
> **Policy head**: Train to predict the MCTS output (visit count distribution)
> - Loss: **Cross-entropy** between $\pi_\theta(a|s)$ and the MCTS search probabilities $\hat{\pi}_{\text{MCTS}}(a|s) \propto N(s,a)$
>
> **Value head**: Train to predict the game outcome
> - Loss: **MSE** (Monte Carlo) between $V_\theta(s)$ and the actual game outcome $z \in \{-1, +1\}$
> - $\mathcal{L}_V = (V_\theta(s) - z)^2$
>
> **Self-play loop**:
> 1. Play games using MCTS (guided by current neural network)
> 2. Collect training data: $(s, \hat{\pi}_{\text{MCTS}}, z)$ for each position
> 3. Train neural network on collected data
> 4. Evaluate if new network is stronger; if so, replace old network
> 5. Repeat

### What Do Neural Networks Bring?

**Benefits**:
- Don't need to start search from scratch -- the network provides informed priors
- **Limit width** of search (policy network focuses on promising actions)
- **Limit depth** of search (value network evaluates positions without full rollouts)

**Costs**:
- What if the neural network policy is wrong? It may bias the search away from good moves
- Neural network evaluation is more computationally costly than a simple random rollout policy

> [!warning]
> The learned policy and value functions only **help** tree search -- they don't directly select actions! The final action is always selected based on the MCTS tree statistics ($N(s,a)$ or $Q(s,a)$), not directly from the neural network output.

---

## Big Picture: Taxonomy of Planning Approaches

|  | **Known Model** | **Learned Model** |
|---|---|---|
| **Rollout algorithm / MCTS** | Planning (e.g., AlphaGo with known rules) | Model-based RL with decision-time planning |
| **Background planning using MF RL tools** | Planning (e.g., DP) | Model-based RL using MF RL tools (e.g., Dyna) |

### Unified View

Another useful perspective organizes methods along two axes:

- **Horizontal axis**: Whether a (parameterized or given) transition model is used
  - "Model-free" methods (left): Q-learning, actor-critic, REINFORCE, etc.
  - "Model-based" methods (right): Dyna, MCTS, etc.
- **Vertical axis**: What is parameterized
  - Value only (critic only): Q-learning, ...
  - Actor-critic: methods that learn both policy and value
  - Policy only (actor only): REINFORCE, ...

AlphaGo sits in the model-based + actor-critic region: it uses a known model (game rules) with MCTS, guided by learned policy and value networks.

---

## Summary & Key Takeaways

> [!summary]
> **Core Contributions of This Lecture**:
>
> 1. **Model-Based RL**: Learn transition model $p(s'|s,a)$ and reward $r(s,a)$ from data, then use the model for planning. Beneficial when real data is expensive; problematic when model errors are significant.
>
> 2. **Dyna-Q**: Integrates direct RL, model learning, and background planning. After each real step, performs $n$ additional planning steps using simulated experience from the learned model. Conceptually similar to [[Experience Replay]].
>
> 3. **Background vs. Decision-Time Planning**: Background planning (Dyna) learns a policy for all states ahead of time. Decision-time planning (rollouts, MCTS) plans from the current state when a decision is needed.
>
> 4. **Rollout Algorithms**: Estimate action values by simulating trajectories from the current state using a rollout policy. Selecting $\arg\max_a \hat{q}(s,a)$ is a policy improvement over the rollout policy.
>
> 5. **MCTS**: Incrementally builds a search tree through four phases -- Selection (UCB1), Expansion, Simulation (rollout), Backup. The tree policy balances exploration and exploitation. More efficient than pure rollouts by reusing information.
>
> 6. **AlphaGo Zero**: Enhances MCTS with neural networks. The policy network $\pi_\theta(a|s)$ limits search width (prior in selection). The value network $V_\theta(s)$ limits search depth (leaf evaluation). Networks trained via self-play with cross-entropy (policy) and MSE (value) losses.

---

## What You Should Know

- **Dyna**: The idea, model learning, background planning
- **When model-based RL might be beneficial** (expensive data, need gradients/distributions) **vs. problematic** (model errors compound, extra compute)
- **Decision-time planning**: What is a [[Rollout Algorithm]]
- **MCTS**: The 4 phases (selection, expansion, simulation, backup), UCB tree policy, final action selection
- **AlphaGo Zero**: How neural networks guide tree search (policy as prior, value for leaf evaluation), how to learn the neural networks (cross-entropy on MCTS output, MSE on game outcome)

---

## New Concepts to Explore

The following concepts are introduced but require deeper study:

- [[Model-Based Reinforcement Learning]] - Learning dynamics models and using them for planning
- [[Dyna]] - Architecture integrating learning, planning, and acting
- [[Monte Carlo Tree Search (MCTS)]] - Incremental tree search with selection, expansion, simulation, backup
- [[AlphaGo Zero]] - MCTS enhanced with neural network policy and value priors
- [[Rollout Algorithm]] - Decision-time planning via simulated trajectories
- [[Background Planning]] - Planning ahead of acting to learn a global policy
- [[Decision-Time Planning]] - Planning from the current state at the time of action selection
- [[Upper Confidence Bound]] - Exploration strategy balancing exploitation and uncertainty

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). Chapters 8 and 16.6.
- Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017). *Mastering the Game of Go without Human Knowledge*. Nature.
- Chang, H. S., Fu, M. C., Hu, J., & Marcus, S. I. (2005). *An Adaptive Sampling Algorithm for Solving Markov Decision Processes*. Operations Research.
- Coulom, R. (2006). *Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search*. CG.
