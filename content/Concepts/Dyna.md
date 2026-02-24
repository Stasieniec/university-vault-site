---
type: concept
aliases: [Dyna architecture, Dyna-Q]
course: [RL]
tags: [planning, exam-topic]
status: complete
---

# Dyna

## Definition

> [!definition] Dyna
> **Dyna** is a model-based reinforcement learning architecture that integrates learning, planning, and acting. It uses real experience to simultaneously improve a policy/value function (direct RL) and learn a **model** of the environment. This model is then used to generate simulated experience for **planning** (indirect RL).

## Intuition

> [!intuition] Learning from the Simulated Past
> In traditional RL (like Q-learning), you only learn from what actually happened. In Dyna, you use your experience to build a mental mirror of the world (the Model). After every real step you take, you pause and imagine $n$ simulated steps using that model. This "imaginary" experience updates your value function just like real experience does, making learning much more sample-efficient.

## Update Logic

Dyna-Q combines direct Q-learning with random replay from the model:

1. **Direct RL**: $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \max_a Q(S', a) - Q(S, A)]$
2. **Model Learning**: $Model(S, A) \leftarrow (R, S')$ (storing the transition)
3. **Planning**: Repeat $n$ times:
   - Select a previously visited state $S$ and action $A$ at random.
   - Query model: $R, S' \leftarrow Model(S, A)$
   - Update $Q$: $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \max_a Q(S', a) - Q(S, A)]$

## Key Components

- **Direct RL**: Improving value functions/policies from real experience.
- **Model Learning**: Learning the transition and reward dynamics: $p(s', r | s, a)$.
- **Planning**: Improving value functions/policies using simulated experience from the model.
- **Search Control**: The process of selecting starting states and actions for the simulated experience.

## Dyna-Q vs Dyna-Q+

| Variant | Strategy | Purpose |
|---------|----------|---------|
| **Dyna-Q** | Randomly samples $(s, a)$ | Standard planning. |
| **Dyna-Q+** | Adds a "curiosity" reward $r + \kappa \sqrt{\tau}$ | Encourages exploration in changing environments where $\tau$ is time since last visit. |

## Connections

- Extends: [[Q-Learning]] (by adding planning)
- Compared with: [[Monte Carlo Tree Search (MCTS)]] (a different planning approach)
- Component of: Model-based RL
- Concept: [[Experience Replay]] (Dyna is a form of model-based replay)

## Appears In

- [[RL-L02 - Planning and Learning with Tabular Methods]] (mentioned)
- Future Planning Lecture
- [[RL-Book Ch8 - Planning and Learning with Tabular Methods]]
