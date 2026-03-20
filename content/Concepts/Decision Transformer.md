---
type: concept
aliases: [Decision Transformer, DT]
course: [RL]
tags: [deep-rl, offline-rl, sequence-modeling]
status: complete
---

# Decision Transformer

> [!definition] Decision Transformer
> An approach to [[Offline Reinforcement Learning]] that casts RL as a **sequence modeling** problem. Instead of estimating value functions or computing policy gradients, it trains a GPT-style autoregressive transformer on trajectories, conditioning on **desired return-to-go** to generate actions.

## Core Idea

> [!intuition] RL as Sequence Modeling
> Instead of asking "what action maximizes expected future reward?", the Decision Transformer asks: "given that I want total return $G$, and I'm in state $s$, what action should I take?" This reframes RL as a **conditional generation** problem — no Bellman equations, no temporal difference learning.

## Trajectory Representation

Trajectories are preprocessed into triples of (return-to-go, state, action):

$$\tau = (\hat{G}_0, s_0, a_0, \hat{G}_1, s_1, a_1, \ldots, \hat{G}_T, s_T, a_T)$$

where $\hat{G}_t = \sum_{t'=t}^{T} r_{t'}$ is the **return-to-go** (total remaining reward from timestep $t$).

## Architecture

- Uses a **GPT** (causal transformer) architecture
- Input: sequence of $(\hat{G}_t, s_t, a_t)$ triples, each embedded and fed as tokens
- Each modality (return, state, action) has its own linear embedding layer
- Positional encoding shared across the triple at each timestep
- Output: predicted action $a_t$ given context $(\hat{G}_0, s_0, a_0, \ldots, \hat{G}_t, s_t)$
- Trained with standard cross-entropy (discrete) or MSE (continuous) loss on actions

## Inference (Test Time)

1. Set desired return-to-go $\hat{G}_0$ to a target performance level
2. Observe current state $s_0$
3. Model predicts action $a_0$
4. Execute $a_0$, observe $r_0$, $s_1$
5. Update: $\hat{G}_1 = \hat{G}_0 - r_0$
6. Repeat

> [!tip] Controlling Performance
> By conditioning on different return-to-go values, you can control the agent's behavior: high $\hat{G}$ produces expert-level behavior, lower $\hat{G}$ produces more conservative behavior.

## Key Properties

- **No value estimation**: avoids issues with bootstrapping, deadly triad, etc.
- **Offline**: learns entirely from logged data, no environment interaction needed
- **Simple training**: standard supervised learning (next-token prediction)
- **Hindsight conditioning**: learns from suboptimal data by conditioning on actual achieved returns
- **Limitation**: shares weaknesses of [[Monte Carlo Methods]] — relies on full trajectory returns, not bootstrapped estimates

## Comparison with Standard RL

| Aspect | Standard RL | Decision Transformer |
|--------|-------------|---------------------|
| **Objective** | Maximize expected return | Predict actions given desired return |
| **Training** | TD/MC + policy gradients | Supervised (next-token prediction) |
| **Value functions** | Required | Not needed |
| **Bellman equations** | Core component | Not used |
| **Data** | On-policy or off-policy | Offline dataset |
| **Architecture** | Various | Transformer (GPT) |

## Connections

- Alternative to value-based [[Offline Reinforcement Learning]] (e.g., [[Conservative Q-Learning (CQL)]])
- Related to [[Decision Diffuser]] (uses diffusion instead of transformers)
- Related to "Upside-Down RL" (single state input, no sequence model)
- Related to "Trajectory Transformer" (concurrent work, also predicts states and returns)
- Builds on transformer/GPT architecture from NLP

## Appears In

- [[RL-L11 - SAC, Decision Transformer & Diffuser]]
- Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling" (2021)
