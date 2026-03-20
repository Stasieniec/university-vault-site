---
type: concept
aliases: [Decision Diffuser, decision diffusion]
course: [RL]
tags: [deep-rl, offline-rl, generative-models]
status: complete
---

# Decision Diffuser

> [!definition] Decision Diffuser
> An [[Offline Reinforcement Learning]] method that uses **diffusion models** to generate future state trajectories, then derives actions via a separately trained **inverse dynamics model**. Unlike [[Decision Transformer]], it generates states rather than actions directly, and can flexibly combine multiple conditioning signals (returns, constraints, skills).

## Core Idea

> [!intuition] Generating Trajectories, Not Actions
> Instead of directly predicting actions, the Decision Diffuser generates a plan — a sequence of future states $(s_t, s_{t+1}, \ldots, s_{t+H})$. A separate inverse dynamics model $f_\phi$ then computes which action to take: $a_t = f_\phi(s_t, s_{t+1})$. This separation allows flexible conditioning on different objectives.

## Architecture

- **Diffusion model** $\epsilon_\theta$: temporal U-Net that generates state trajectories
  - Conditioning information projected to a latent vector $z$ via MLP
  - Conditions with probability $p$ (for classifier-free guidance)
- **Inverse dynamics model** $f_\phi$: MLP that predicts $a_t$ from $(s_t, s_{t+1})$

## Training

The diffusion model is trained to predict the noise added during the forward diffusion process (standard denoising objective). The inverse dynamics model is trained separately with supervised learning on $(s_t, s_{t+1}) \to a_t$ pairs.

## Conditioning (Classifier-Free Guidance)

The diffusion model can be conditioned on:
- **Returns**: generate high-reward trajectories
- **Constraints**: e.g., block height requirements like $\text{blockheight}(i) > \text{blockheight}(j)$
- **Skills**: generate behavior that interpolates between learned skills

Uses **classifier-free guidance** — during training, conditioning is randomly dropped (replaced with null conditioning) with some probability, so at inference the model can trade off conditioned vs unconditioned generation.

## Low-Temperature Sampling

During denoising at inference, the variance of the predicted noise is **reduced** (low-temperature sampling), producing more deterministic and focused trajectory plans.

## Results

- Competitive with or outperforms baselines (behavioral cloning, [[Conservative Q-Learning (CQL)]], [[Decision Transformer]]) on offline RL benchmarks
- Better at single constraints than baselines
- Only model that can effectively combine multiple constraints
- Can generate behavior 'in between' separately learned skills

## Key Differences from Decision Transformer

| Aspect | [[Decision Transformer]] | Decision Diffuser |
|--------|------------------------|-------------------|
| **Generative model** | Autoregressive transformer | Diffusion model |
| **Generates** | Actions directly | State trajectories |
| **Action derivation** | Direct output | Inverse dynamics model |
| **Conditioning** | Return-to-go only | Returns, constraints, skills |
| **Multi-objective** | Limited | Natural (classifier-free guidance) |

## Connections

- Alternative to [[Decision Transformer]] for offline RL
- Uses diffusion models (from generative AI / image generation)
- Related to "Diffuser" (Janner et al., 2022) which uses classifier guidance and predicts state-action pairs
- Builds on denoising diffusion probabilistic models (DDPM)
- Part of the broader trend of applying sequence/generative models to RL

## Appears In

- [[RL-L11 - SAC, Decision Transformer & Diffuser]]
- Ajay et al., "Is Conditional Generative Modeling all you need for Decision-Making?" (2022)
