---
type: concept
aliases: [AlphaGo Zero, AlphaZero, Alpha Go Zero]
course: [RL]
tags: [deep-rl, planning, exam-topic]
status: complete
---

# AlphaGo Zero

> [!definition] AlphaGo Zero
> A system that combines [[Monte Carlo Tree Search (MCTS)]] with deep neural networks to play Go (and later Chess and Shogi) at superhuman level. The neural network provides a **policy prior** $\pi_\theta(a|s)$ to guide tree search and a **value estimate** $V_\theta(s)$ to evaluate leaf nodes, replacing random rollouts.

## Key Innovation: Neural Network Guided MCTS

Standard MCTS uses random rollouts to estimate values and UCB1 for selection. AlphaGo Zero replaces both with neural networks:

> [!formula] Modified Tree Policy (Selection)
> $$\pi_{\text{tree}}(s) = \arg\max_a \left[ Q(s,a) + c \cdot \frac{\pi_\theta(a|s) \cdot \sqrt{N(s)}}{N(s,a) + 1} \right]$$
>
> where:
> - $Q(s,a)$ — average value from simulations through this state-action
> - $\pi_\theta(a|s)$ — neural network policy prior (replaces uniform prior in UCB1)
> - $N(s)$ — visit count of state $s$
> - $N(s,a)$ — visit count of action $a$ in state $s$
> - $c$ — exploration constant

## What Neural Networks Provide

| MCTS Component | Standard MCTS | AlphaGo Zero |
|----------------|---------------|--------------|
| **Selection** | UCB1 (uniform prior) | UCB with $\pi_\theta(a|s)$ prior |
| **Simulation** | Random rollout to terminal state | $V_\theta(s)$ evaluation at leaf |
| **Effect on width** | Explores all branches | Focuses on high-prior actions (limits width) |
| **Effect on depth** | Must rollout to end | Value function limits depth |

## Training

The neural network is trained through **self-play**:

1. **Play games** using MCTS (with current neural network) to select moves
2. **Collect training data**: for each position, record:
   - State $s$
   - MCTS search probabilities $\vec{\pi}_{\text{MCTS}}$ (based on visit counts)
   - Game outcome $z \in \{-1, +1\}$
3. **Train neural network**:
   - **Policy head**: Cross-entropy loss — train $\pi_\theta(a|s)$ to match MCTS output $\vec{\pi}_{\text{MCTS}}$
   - **Value head**: MSE loss — train $V_\theta(s)$ to predict game outcome $z$ (Monte Carlo target)
4. **Evaluate**: check if new network beats previous version
5. Repeat

> [!intuition] The Virtuous Cycle
> Better neural networks → better MCTS search → better training targets → even better neural networks. The policy network learns to "distill" the improved policy that MCTS computes, and the value network learns from game outcomes to evaluate positions more accurately.

## Architecture

A single neural network with two heads:
- **Input**: Board state $s$
- **Shared body**: Deep residual network
- **Policy head**: Outputs $\pi_\theta(a|s)$ — probability over all legal moves
- **Value head**: Outputs $V_\theta(s) \in [-1, 1]$ — estimated probability of winning

## Key Properties

- **No human knowledge**: learns entirely from self-play (no expert games)
- **No random rollouts**: neural network evaluation replaces simulation phase
- **Search still matters**: the neural network guides but doesn't replace MCTS — MCTS provides the actual decision and generates improved training targets
- **Cost**: computationally very expensive (thousands of TPUs for training)

## Connections

- Builds on [[Monte Carlo Tree Search (MCTS)]] — the core planning algorithm
- Uses [[Upper Confidence Bound]] ideas for tree selection
- Part of [[Model-Based Reinforcement Learning]] — requires a game model (known rules)
- Demonstrates the power of combining [[Deep Reinforcement Learning]] with planning

## Appears In

- [[RL-L12 - Model-Based RL]]
- [[RL-Book Ch16 - Applications and Case Studies]] (§16.6)
- Silver et al., "Mastering the game of Go without human knowledge" (2017)
