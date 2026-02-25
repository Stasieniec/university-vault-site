---
type: book-chapter
course: RL
book: "Reinforcement Learning: An Introduction (2nd ed.)"
chapter: 1
sections: ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7"]
topics:
  - "[[Reinforcement Learning]]"
  - "[[Markov Decision Process]]"
  - "[[Value Function]]"
  - "[[Policy]]"
  - "[[Reward Signal]]"
  - "[[Exploration vs Exploitation]]"
status: complete
---

# Chapter 1: Introduction

## Overview
This chapter introduces the fundamental concepts of **Reinforcement Learning (RL)**, a computational approach to learning from interaction to achieve goals. Unlike other machine learning paradigms, RL focuses on an agent's direct sensorimotor connection to its environment, mapping situations to actions to maximize a numerical reward signal.

---

## 1.1 Reinforcement Learning
Reinforcement Learning is defined by several core characteristics that distinguish it from other fields:

> [!definition] Reinforcement Learning
> RL is learning what to do—how to map situations to actions—so as to maximize a numerical **reward signal**. The learner is not told which actions to take but must discover them through trial and error.

### Key Distinguishing Features
1. **Trial-and-Error Search**: The agent must explore various actions to discover which yield the most reward.
2. **Delayed Reward**: Actions may affect not only the immediate reward but also future situations and all subsequent rewards.

### Comparison with Other ML Paradigms
- **[[Supervised Learning]]**: Learning from a training set of labeled examples provided by a supervisor. Not adequate for learning from interaction because it's impractical to obtain representative examples of all situations an agent might face.
- **[[Unsupervised Learning]]**: Typically about finding hidden structure in unlabeled data. RL differs because it is specifically trying to maximize a reward signal rather than searching for structure.

> [!warning] Exploration vs exploitation
> One of the unique challenges in RL is the trade-off between **[[Exploration vs Exploitation]]**. To obtain reward, an agent must **exploit** what it already knows, but to discover better actions, it must **explore** actions it has not selected before. Neither can be pursued exclusively.

---

## 1.2 Examples
The chapter provides several diverse examples of RL in action:
- A master chess player making a move based on foresight and intuition.
- An adaptive controller in a petroleum refinery optimizing trade-offs in real-time.
- A gazelle calf learning to run shortly after birth.
- A mobile robot deciding between searching for trash or returning to its charging station based on battery levels.
- A human (Phil) preparing breakfast, involving complex web of conditional behaviors and goal-subgoal relationships.

---

## 1.3 Elements of Reinforcement Learning
A reinforcement learning system consists of four main subelements:

1.  **[[Policy]]**: The agent's way of behaving at a given time; a mapping from perceived states to actions.
2.  **[[Reward Signal]]**: Defines the goal. A single number sent by the environment on each time step. The agent's sole objective is to maximize the total reward over the long run.
3.  **[[Value Function]]**: Specifies what is good in the long run. The *value* of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state.
4.  **Model of the Environment** (Optional): Mimics the behavior of the environment, allowing for inferences (planning) about how the environment will behave.

> [!intuition] Reward vs. Value
> Rewards are immediate and primary (like pleasure or pain). Values are secondary and farsighted (like a judgment of a state based on its potential for future reward). We choose actions based on **value** judgments to obtain the most **reward** over the long run.

---

## 1.4 Limitations and Scope
RL relies heavily on the concept of **state**—the information available to the agent about its environment. This book focuses on the decision-making issues (what action to take) rather than how the state signal is constructed.

### RL vs. Evolutionary Methods
- **Evolutionary Methods**: (e.g., genetic algorithms) search the space of behaviors directly by evaluating entire policies over many interactions.
- **RL Methods**: Learn while interacting with the environment, taking advantage of the structure of individual behavioral interactions.

---

## 1.5 An Extended Example: Tic-Tac-Toe
To contrast RL with classical methods like minimax or dynamic programming, the authors use Tic-Tac-Toe.

### The RL Approach (using a Value Function)
1.  **Set up a table**: One entry for each possible game state, representing the probability of winning from that state.
    - Wins = 1.0; Losses/Draws = 0.0; Others (initially) = 0.5.
2.  **Play games**: 
    - **Greedy moves**: Choose the move leading to the state with the highest value.
    - **Exploratory moves**: Occasionally choose a random move to see new states.
3.  **Back up values**: Update the value of previous states based on the value of subsequent states using **Temporal Difference (TD)** learning.

### Algorithm: Temporal Difference Update (Tabular)
```pseudocode
For each greedy move from state St to state St+1:
    V(St) <- V(St) + alpha * [V(St+1) - V(St)]
```
- $V(S_t)$: Current estimate of the value of state $S_t$.
- $\alpha$ (alpha): A small positive fraction called the **step-size parameter**.
- $[V(S_{t+1}) - V(S_t)]$: The **temporal difference** (error) between estimates at successive times.

> [!example] Tic-Tac-Toe Learning
> An RL player can learn to set up multi-move traps for an opponent without an explicit search tree, simply by refining its value estimates through experience.

---

## 1.6 Summary
- RL is a computational approach to automating goal-directed learning and decision-making.
- It is distinguished by its emphasis on learning from direct interaction without exemplary supervision.
- It uses the formal framework of **[[Markov Decision Process]] (MDPs)** to define interactions in terms of states, actions, and rewards.
- Value functions are the key to efficient search in the policy space.

---

## 1.7 Early History of Reinforcement Learning
Modern RL emerged from the convergence of three independent threads:
1.  **Trial and Error (Psychology)**: Evolved from the "Law of Effect" (Thorndike) and animal learning theories.
2.  **Optimal Control (Mathematics/Engineering)**: Focused on minimizing/maximizing measures of system behavior using value functions and **[[Dynamic Programming]]**.
3.  **Temporal-Difference Methods**: Methods driven by the difference between successive estimates (e.g., Samuel's checkers program).

---

## Key Takeaways
- RL is about maximizing a **reward signal** through interaction.
- The **[[Exploration vs Exploitation]]** trade-off is fundamental.
- **[[Value Function]]** estimation is arguably the most important component of RL algorithms.
- RL methods can achieve the effects of planning/foresight without an explicit model of the environment.

## Related Notes
- [[Reinforcement Learning]]
- [[Markov Decision Process]]
- [[Value Function]]
- [[Policy]]
- [[Reward Signal]]
- [[Exploration vs Exploitation]]
- [[Temporal Difference Learning]]
- [[Dynamic Programming]]
