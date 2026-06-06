---
type: concept
aliases: [Inverse Dynamics, IDM]
course: [RL]
tags: [model-based, offline-rl, deep-rl, exam-topic]
status: complete
---

# Inverse Dynamics Model

## Definition

> [!definition] Inverse Dynamics Model
> An **inverse dynamics model** $f_\phi$ predicts the **action** that caused a transition between two consecutive states: given a state $s_t$ and the next state $s_{t+1}$, it outputs the action $a_t$ that connects them, $a_t = f_\phi(s_t, s_{t+1})$.
>
> It is the **inverse** of a forward [[Model of the Environment|dynamics model]] $p(s_{t+1} \mid s_t, a_t)$: the forward model predicts the next state from a state-action pair, whereas the inverse model recovers the action from a state pair.

## Intuition

> [!intuition] States are smooth, actions are jerky
> In the [[Decision Diffuser]], the diffusion model generates only a sequence of **future states**, not actions. Why? In control/robotics domains, states (positions, velocities) tend to be **continuous and smooth**, which is exactly the kind of signal diffusion models excel at generating. Actions (torques, motor commands) are often **jerky, discrete, or high-frequency** and are harder to model directly.
>
> The inverse dynamics model cleanly separates concerns: a generative model decides *where to go* (the state trajectory), and the inverse dynamics model figures out *how to get there* (the action between each consecutive state pair). To execute a planned trajectory, you simply read off each action $a_t = f_\phi(s_t, s_{t+1})$.

## Mathematical Formulation

> [!formula] Inverse Dynamics Model
> $$a_t = f_\phi(s_t, s_{t+1})$$
>
> Trained by supervised regression on logged transitions $(s_t, a_t, s_{t+1}) \in \mathcal{D}$:
> $$\mathcal{L}(\phi) = \mathbb{E}_{(s_t, a_t, s_{t+1}) \sim \mathcal{D}} \left[ \left\| a_t - f_\phi(s_t, s_{t+1}) \right\|^2 \right]$$
>
> where:
> - $f_\phi$ — the inverse dynamics network (an MLP in the Decision Diffuser), parameters $\phi$
> - $s_t, s_{t+1}$ — consecutive states (the *input* to the model)
> - $a_t$ — the action that produced the transition (the regression *target*)
> - $\mathcal{D}$ — the offline dataset of logged trajectories

This contrasts with the forward dynamics model $p(s_{t+1} \mid s_t, a_t)$, which maps a state-action pair to the next state and is used for planning in [[Model-Based RL]].

## Key Properties / Variants

- **Separates generation from control**: lets a [[Decision Diffuser]] generate a clean state trajectory $(s_t, s_{t+1}, \dots, s_{t+H})$ and then extract executable actions post-hoc — sidestepping the difficulty of modeling jerky action signals with a diffusion process.
- **Pure supervised learning**: trained by regression on $(s_t, a_t, s_{t+1})$ triples, with **no value bootstrapping, pessimism, or regularization** required — fitting the broader theme of RL-as-supervised-learning in this lecture.
- **Architecture**: a simple MLP suffices; in the Decision Diffuser the trajectory generator $\epsilon_\theta$ is a Temporal U-Net while $f_\phi$ is an MLP.
- **Difference from the Diffuser (Janner et al.)**: the original Diffuser generates **state-action pairs** jointly (no separate inverse model), whereas the Decision Diffuser generates **states only** and relies on the inverse dynamics model for actions.
- **Identifiability caveat**: a single $(s_t, s_{t+1})$ pair may be reachable by more than one action (or none) in stochastic or partially observable environments, which can make the inverse map ambiguous.

How the inverse dynamics model is used at deployment within the Decision Diffuser:

```pseudo
Procedure: Act via Inverse Dynamics (Decision Diffuser inference)
─────────────────────────────────────────────────────────────────
Given: trained diffusion model ε_θ, inverse dynamics model f_φ,
       current state s_t, desired property (e.g. return R = 1)

1. Generate a state trajectory by reverse diffusion (denoising),
   conditioned on the desired property via classifier-free guidance:
       (s_t, s_{t+1}, ..., s_{t+H}) ← Denoise(ε_θ | s_t, R)
   (use low-temperature sampling for more deterministic plans)

2. Extract the next action from the first consecutive state pair:
       a_t ← f_φ(s_t, s_{t+1})

3. Execute a_t in the environment, observe new state, and replan.
```

## Connections

- Inverse of: [[Model of the Environment]] / forward dynamics in [[Model-Based RL]]
- Core component of: [[Decision Diffuser]]
- Pairs with: [[Classifier-Free Guidance]] (steers the generated state trajectory)
- Alternative trajectory model: [[Decision Transformer]] (predicts actions autoregressively instead)
- Setting: [[Offline Reinforcement Learning]]

## Appears In

- [[RL-L11 - SAC, Decision Transformer & Diffuser]]
