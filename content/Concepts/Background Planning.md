---
type: concept
aliases: []
course: [RL]
tags: [model-based-rl, planning, exam-topic]
status: complete
---

# Background Planning

## Definition

> [!definition] Background Planning
> **Background planning** uses a model of the environment to improve a **global** value function or policy for *any* state, **ahead of** acting in the world. Planning runs "in the background" — interleaved with (or between) real interactions — and the resulting policy is then used to act **without further planning** at decision time. It is one of the two ways to use a learned model in [[Model-Based Reinforcement Learning]], the other being [[Decision-Time Planning]].

## Intuition

> [!intuition] Plan Ahead, Act Fast
> Imagine studying every position in a game before you ever sit down to play, so that when it's your turn you already know what to do everywhere. Background planning spreads model-based computation across *all* states gradually, building up a value function or policy that applies globally. Contrast this with [[Decision-Time Planning]] (e.g. a chess player reasoning forward from the *current* board): there you plan only the part of the state space relevant *right now*, at the moment a decision is required.
>
> The mechanism is simple: treat experience *imagined* by the model exactly like real experience, and feed it into ordinary model-free updates. Each simulated transition is "extra" learning squeezed out of the model.

## Mathematical Formulation

Background planning generates simulated transitions $(s, a, r, s')$ from a learned model and applies a standard model-free update to them. With [[Q-Learning]] as the planner (the [[Dyna]]-Q case), the planning update for each simulated sample is:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[\, \underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{from simulated experience}} - Q(s,a)\,\right]$$

where:
- $(s,a)$ — a previously observed state-action pair, sampled by **search control** (in Dyna-Q, uniformly at random from visited pairs)
- $r, s' \leftarrow \text{Model}(s,a)$ — reward and next state *predicted by the model*, not the real environment
- $\alpha$ — step size; $\gamma$ — discount factor
- $\max_{a'} Q(s',a')$ — bootstrapped estimate of optimal future value

The defining feature is that $r$ and $s'$ come from the model $\hat{p}(s'\mid s,a)$, $\hat{r}(s,a)$ rather than from real interaction. The **same** update is used for real experience (direct RL); the only difference is the data source. This is why background planning is, in Sutton & Barto's words, "just learning applied to simulated experience."

## Key Properties / Variants

- **Global, not local**: improves the value function / policy over the whole (visited) state space, not just the current state.
- **Plan "ahead of" acting**: the policy is ready before action selection, so acting is cheap — no per-decision search.
- **Planner-agnostic**: any model-free method can be the planner — [[Q-Learning]], [[SARSA]], policy gradient — or even full [[Dynamic Programming]] / value-iteration-style sweeps if the model gives full distributions.
- **Search control** determines *which* simulated states/actions to update; uniform-random is the basic Dyna-Q choice, prioritized sweeping focuses updates where they matter most.
- **Sample efficiency**: extracts more value from each real interaction (lower real-sample complexity) at the cost of extra compute per real step.
- **Conceptually like [[Experience Replay]]**: both reuse the past for extra updates — but replay stores *real* transitions, whereas background planning generates *new* simulated ones from the model.
- **Canonical example**: the [[Dyna]] architecture (Dyna-Q).

The generic background-planning loop (Dyna-style) interleaves direct RL, model learning, and planning:

```pseudo
Algorithm: Background Planning (Dyna-style loop)
──────────────────────────────────────────────────
Initialize Q(s,a) and Model(s,a) for all s,a

Loop forever (per real step):
  s ← current state
  a ← ε-greedy(s, Q)
  Execute a, observe r, s'

  # (1) Direct RL — real experience
  Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') − Q(s,a)]

  # (2) Model learning — store the transition
  Model(s,a) ← (r, s')

  # (3) Background planning — n simulated updates
  Repeat n times:
    s_sim ← random previously visited state        # search control
    a_sim ← random action previously taken in s_sim
    r_sim, s'_sim ← Model(s_sim, a_sim)             # simulated, not real
    Q(s_sim,a_sim) ← Q(s_sim,a_sim)
                     + α[r_sim + γ max_{a'} Q(s'_sim,a') − Q(s_sim,a_sim)]

  s ← s'
```

> [!warning] Don't Trust the Model Too Much
> Background planning is only as good as the model. Model errors **compound** when simulated experience drives many updates, and the policy can come to **exploit model inaccuracies** rather than solve the real task. If real data is plentiful and cheap, pure model-free learning is often safer than aggressive background planning.

## Connections

- One of two model usages in: [[Model-Based Reinforcement Learning]]
- Contrasted with: [[Decision-Time Planning]] (local, per-state, at action time)
- Canonical instance: [[Dyna]] (Dyna-Q)
- Planners used: [[Q-Learning]], [[SARSA]], [[Dynamic Programming]]
- Closely related idea: [[Experience Replay]] (real vs. simulated transitions)

## Appears In

- [[RL-L12 - Model-Based RL]]
- [[RL-Book Ch8 - Planning and Learning]]
