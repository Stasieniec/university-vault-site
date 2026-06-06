---
type: concept
aliases: [Momentum (optimization)]
course: [RL]
tags: [optimization, deep-rl, exam-topic]
status: complete
---

# Momentum

## Definition

> [!definition] Momentum
> **Momentum** is a modification of [[Gradient Descent]] that accelerates optimization by accumulating an **exponentially decaying moving average of past gradients** (the *velocity*) and stepping in that direction, instead of stepping in the raw gradient direction. This smooths out noisy [[SGD]] updates and builds up speed along directions of persistent descent.

## Intuition

> [!intuition] A Ball Rolling Down the Loss Surface
> Plain gradient descent is like a memoryless walker: at every step it looks only at the local slope. **Momentum** is like a heavy ball rolling downhill — it carries inertia from previous steps.
>
> - In a **steep, narrow ravine** (ill-conditioned curvature), plain SGD oscillates back and forth across the walls and crawls slowly along the valley floor. Momentum **cancels the oscillating components** (they alternate sign and average out) while **reinforcing the consistent down-valley component** (it keeps the same sign and accumulates).
> - Near small local bumps or noisy gradients, the accumulated velocity lets the optimizer **coast through**, smoothing stochastic noise.
>
> The decay rate $\beta$ controls how much "memory" the ball has: larger $\beta$ means heavier inertia and more smoothing.

## Mathematical Formulation

Maintain a velocity vector $v_t$ (the moving average of gradients) and update the parameters with it. Let $g_t = \nabla_w L(w_{t-1})$ be the gradient at step $t$.

> [!formula] Momentum Update (velocity form)
> $$v_t = \beta\, v_{t-1} + (1 - \beta)\, g_t$$
> $$w_t = w_{t-1} - \alpha\, v_t$$
>
> where:
> - $w_t$ — parameters (weights) at step $t$
> - $g_t = \nabla_w L(w_{t-1})$ — gradient of the loss w.r.t. the weights
> - $v_t$ — velocity: exponential moving average of gradients (the **first moment**)
> - $\beta \in [0,1)$ — momentum / decay coefficient (typically $0.9$); larger = more inertia
> - $\alpha$ — learning rate (step size)

An equivalent and very common formulation uses an **accumulated update** rather than a normalized average:

> [!formula] Momentum Update (accumulation form)
> $$v_t = \beta\, v_{t-1} + g_t \qquad\qquad w_t = w_{t-1} - \alpha\, v_t$$
>
> The two forms differ only by a constant factor $(1-\beta)$ absorbed into the effective learning rate. In the steady state where the gradient is constant $g$, the velocity converges to $v_\infty = \tfrac{g}{1-\beta}$, so momentum effectively scales the step by $\tfrac{1}{1-\beta}$ along persistent directions (e.g. $10\times$ for $\beta = 0.9$).

## Key Properties / Variants

- **First moment estimate**: $v_t$ is exactly the first-moment (mean-of-gradients) term reused by [[Adam]]; Adam pairs it with the second-moment ([[RMSProp]]) term for per-parameter adaptive scaling.
- **Bias at start-up**: with the moving-average form, $v_0 = 0$ biases early velocities toward zero. Adam corrects this with bias correction $\hat v_t = v_t / (1 - \beta^t)$; plain momentum usually ignores it.
- **Damps oscillation, accelerates valleys**: cancels alternating-sign gradient components, accumulates consistent ones — the main reason it speeds up ill-conditioned problems.
- **Nesterov Accelerated Gradient (NAG)**: a variant that evaluates the gradient at the *look-ahead* point $w_{t-1} - \alpha\beta v_{t-1}$ rather than at $w_{t-1}$, giving a correction term and often faster, more stable convergence.
- **Hyperparameter coupling**: effective step size grows with $\beta$, so $\alpha$ often needs reducing when momentum is added.

```pseudo
Algorithm: SGD with Momentum
─────────────────────────────────────────────
Input: learning rate α, momentum coefficient β
Initialize weights w, velocity v ← 0

Loop for each step t:
  Sample mini-batch; compute gradient  g ← ∇_w L(w)
  v ← β·v + (1 - β)·g          # accumulate velocity (moving avg)
  w ← w - α·v                  # step along velocity
until converged
```

## Connections

- Special form of / accelerates: [[Gradient Descent]], [[SGD]]
- First-moment component of: [[Adam]]
- Complements: [[RMSProp]] (second moment) — Adam combines both
- Alternative adaptive method: [[Adagrad]]
- Used for training: [[Neural Networks]] in [[Deep RL]]

## Appears In

- [[Adam]]
