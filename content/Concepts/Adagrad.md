---
type: concept
aliases: [AdaGrad]
course: [RL]
tags: [optimization, deep-rl, exam-topic]
status: complete
---

# Adagrad

## Definition

> [!definition] Adagrad (Adaptive Gradient Algorithm)
> **Adagrad** is an adaptive learning-rate optimization method that gives **each parameter its own learning rate**, scaled inversely to the square root of the **sum of all past squared gradients** for that parameter. Parameters with frequently large gradients get small effective steps; rarely-updated (sparse) parameters get large steps.

## Intuition

> [!intuition] Per-Parameter Rates from Accumulated History
> Plain [[Gradient Descent]] uses a single global step size $\alpha$ for every weight. But in problems with sparse or differently-scaled features, some directions need big steps and others tiny ones. Adagrad tracks, for each parameter, how much gradient "energy" it has seen so far (the running sum of squared gradients $G_t$). A parameter that has accumulated large gradients is in a steep, frequently-active direction, so its step is shrunk; a parameter rarely touched keeps a large step. This automatic per-coordinate scaling is what made Adagrad effective for sparse data (e.g. text/NLP features).

## Mathematical Formulation

For each parameter $w$, Adagrad accumulates the squared gradients and divides the global step size by their square root.

> [!formula] Adagrad Update
> $$G_t = G_{t-1} + g_t^2 \qquad\qquad w \leftarrow w - \frac{\alpha}{\sqrt{G_t} + \epsilon}\, g_t$$
>
> where:
> - $g_t = \dfrac{\partial L}{\partial w}$ — current gradient of the loss w.r.t. parameter $w$
> - $G_t$ — running **sum of all squared gradients** up to step $t$ (i.e. $G_t = \sum_{\tau=1}^{t} g_\tau^2$), one accumulator per parameter
> - $\alpha$ — global (base) learning rate
> - $\epsilon$ — small constant for numerical stability (avoids division by zero, e.g. $10^{-8}$)

The factor $\dfrac{\alpha}{\sqrt{G_t}+\epsilon}$ is the **effective per-parameter learning rate**. Because $G_t$ is monotonically non-decreasing, this effective rate only ever shrinks over time.

## Key Properties / Variants

- **Per-parameter adaptation**: each weight gets an individual learning rate without manual tuning.
- **Great for sparse features**: infrequently-updated parameters retain large effective steps, so rare-but-informative features still learn quickly.
- **Decaying learning rate (the core weakness)**: $G_t$ accumulates *all* history and never shrinks, so $\frac{\alpha}{\sqrt{G_t}}$ eventually drives the effective step toward **zero**, halting learning prematurely. This is the failure mode [[RMSProp]] and [[Adam]] were designed to fix by replacing the cumulative sum with an **exponentially decaying average** of squared gradients.
- **Less common in deep RL/DL today**: superseded in practice by [[RMSProp]] and [[Adam]], though it remains a strong baseline for convex / sparse problems.

```pseudo
Algorithm: Adagrad
──────────────────────────────────────────────
Require: base learning rate α, small ε
Initialize parameters w
Initialize accumulator G ← 0   (same shape as w)

Loop for each step t:
  Sample / receive data, compute gradient g ← ∂L/∂w
  G ← G + g ⊙ g                 # elementwise accumulate squared grads
  w ← w - α * g / (sqrt(G) + ε) # elementwise per-parameter update
until converged
```

## Connections

- Type of: [[Gradient Descent]] (adaptive learning-rate variant), alternative to plain [[SGD]]
- Improved by: [[RMSProp]] (decaying average instead of full sum), [[Adam]] (decaying average + [[Momentum]])
- Contrast: [[Momentum]] adapts the *direction*; Adagrad adapts the *per-parameter magnitude*
- Used for: training [[Neural Networks]] in [[Deep RL]]

## Appears In

- [[Adam]]
- [[RMSProp]]
