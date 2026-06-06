---
type: concept
aliases: [CFG, Classifier-Free Guidance]
course: [RL]
tags: [deep-rl, offline-rl, generative-models, exam-topic]
status: complete
---

# Classifier-Free Guidance

## Definition

> [!definition] Classifier-Free Guidance (CFG)
> **Classifier-free guidance** is a technique for **conditional sampling from diffusion models** that steers the reverse (denoising) process toward samples with a desired property $y$ **without training a separate classifier** $p(y \mid x)$. A single noise-prediction network $\epsilon_\theta$ is trained **jointly** on conditioned inputs $\epsilon_\theta(x_k, y, k)$ and unconditioned inputs $\epsilon_\theta(x_k, \varnothing, k)$ (by randomly dropping the condition $y$ during training). At sampling time, the two predictions are linearly extrapolated with a **guidance weight** $\omega$ to amplify the influence of $y$. In the [[Decision Diffuser]], $y$ is a return level, a constraint, or a skill, and $x$ is a state trajectory.

## Intuition

> [!intuition] One Network, Two Modes
> A diffusion model generates data by starting from pure Gaussian noise and iteratively **denoising** it. To make that generation *conditional* (e.g. "produce a high-return trajectory"), the older approach (**classifier guidance**) trained a separate classifier on noisy data and pushed samples uphill along $\nabla_{x} \log p(y \mid x)$ — but training a classifier on noisy inputs is awkward and adds a second model.
>
> CFG avoids this. During training, the same network sometimes sees the condition $y$ and sometimes sees a **null token** $\varnothing$ (the condition is "dropped"). So the network learns to denoise *both* conditionally and unconditionally. At inference, the **difference** between the conditional and unconditional noise predictions, $\big(\epsilon_\theta(x_k, y, k) - \epsilon_\theta(x_k, \varnothing, k)\big)$, implicitly points in the direction $\nabla_x \log p(y\mid x)$ — the very direction a classifier would have given. We then **over-emphasize** that direction by the weight $\omega$, sharpening how strongly the sample obeys $y$.

## Mathematical Formulation

A diffusion model defines a **forward (noising) process** that gradually corrupts data $x_0$ into noise over $K$ steps, and learns to reverse it.

> [!formula] Forward Diffusion (DDPM)
> $$q(x_k \mid x_{k-1}) = \mathcal{N}\!\left(x_k;\ \sqrt{1-\beta_k}\,x_{k-1},\ \beta_k \mathbf{I}\right), \qquad x_k = \sqrt{\bar\alpha_k}\,x_0 + \sqrt{1-\bar\alpha_k}\,\epsilon$$
>
> where:
> - $x_0$ — clean sample (in [[Decision Diffuser]], a state trajectory $(s_t,\dots,s_{t+H})$)
> - $k = 1,\dots,K$ — diffusion timestep (not the RL time index)
> - $\beta_k \in (0,1)$ — noise variance schedule
> - $\bar\alpha_k = \prod_{i=1}^{k}(1-\beta_i)$ — cumulative signal-retention factor
> - $\epsilon \sim \mathcal{N}(0,\mathbf{I})$ — the noise actually injected

The network $\epsilon_\theta$ is trained to **predict that injected noise**, with the condition $y$ randomly replaced by $\varnothing$ with probability $1-p$ (dropout).

> [!formula] Denoising Training Loss (with condition dropout)
> $$\mathcal{L}(\theta) = \mathbb{E}_{k,\,x_0,\,\epsilon,\,\beta}\Big[\ \big\lVert \epsilon - \epsilon_\theta(x_k,\, (1-\beta)\,y + \beta\,\varnothing,\, k)\big\rVert^2\ \Big]$$
>
> where:
> - $\epsilon_\theta$ — noise-prediction network (a **temporal U-Net** in Decision Diffuser)
> - $y$ — the conditioning property (return, constraint, skill), projected to a latent $z$ via an MLP
> - $\varnothing$ — null / "unconditioned" token
> - $\beta \sim \text{Bernoulli}(1-p)$ — drops the condition so the model also learns $\epsilon_\theta(x_k,\varnothing,k)$

At **sampling time**, the conditional and unconditional predictions are combined into a single **guided** noise estimate:

> [!formula] Classifier-Free Guided Noise Prediction
> $$\hat{\epsilon}_\theta(x_k, y, k) = \epsilon_\theta(x_k, \varnothing, k) + \omega\,\Big(\epsilon_\theta(x_k, y, k) - \epsilon_\theta(x_k, \varnothing, k)\Big)$$
>
> where:
> - $\hat{\epsilon}_\theta$ — guided noise used in the reverse step to compute $x_{k-1}$
> - $\epsilon_\theta(x_k,\varnothing,k)$ — unconditional prediction
> - $\epsilon_\theta(x_k,y,k)$ — conditional prediction
> - $\omega \ge 0$ — **guidance weight**: $\omega = 0$ gives unconditional sampling, $\omega = 1$ gives ordinary conditional sampling, $\omega > 1$ **amplifies** adherence to $y$ (sharper conditioning, less diversity)

> [!intuition] Why the Extrapolation Works
> Score-matching theory says $\epsilon_\theta(x_k,k) \propto -\nabla_x \log p(x_k)$. By Bayes' rule $\log p(x_k \mid y) = \log p(x_k) + \log p(y \mid x_k) - \log p(y)$, so the **gap** between conditional and unconditional scores equals the classifier gradient $\nabla_x \log p(y\mid x_k)$. CFG reconstructs that gradient *without a classifier* and scales it by $\omega$, recovering classifier guidance with strength $\omega$ as a special case.

## Key Properties / Variants

- **No separate classifier**: avoids training a noise-robust classifier $p(y\mid x_k)$; one network handles both conditional and unconditional generation.
- **Guidance weight $\omega$ trades fidelity vs diversity**: larger $\omega$ produces samples that more strongly satisfy $y$ but reduces sample diversity (and can introduce artifacts).
- **Condition dropout probability** $1-p$: a hyperparameter; the model must see enough unconditioned examples to learn a usable $\epsilon_\theta(x_k,\varnothing,k)$.
- **Composable conditions**: because conditioning is just an input to $\epsilon_\theta$, multiple guidance signals (e.g. several constraints) can be combined — the property the [[Decision Diffuser]] exploits to satisfy *combinations* of constraints, which classifier guidance struggles with.
- **vs classifier guidance**: the original *Diffuser* (Janner et al., 2022) uses **classifier guidance** — it perturbs the reverse process by the gradient of a learned return predictor — and generates **state-action pairs**; Decision Diffuser switches to CFG and generates **states only** (with an [[Inverse Dynamics Model]] recovering actions).
- **Low-temperature sampling** is typically combined with CFG at inference: reducing the variance of the predicted noise yields more deterministic, higher-quality plans.

Sampling procedure with CFG inside the reverse diffusion loop:

```pseudo
Algorithm: Conditional Sampling with Classifier-Free Guidance
─────────────────────────────────────────────────────────────
Input: trained ε_θ, condition y, guidance weight ω, schedule {β_k}
Sample x_K ~ N(0, I)                       # start from pure noise

Loop for k = K, K-1, ..., 1:
    # two forward passes through the SAME network
    ε_cond   ← ε_θ(x_k, y, k)              # conditional prediction
    ε_uncond ← ε_θ(x_k, ∅, k)              # unconditioned prediction

    # classifier-free guided noise (extrapolate)
    ε̂ ← ε_uncond + ω * (ε_cond - ε_uncond)

    # one reverse (denoising) step using ε̂
    x_{k-1} ← reverse_step(x_k, ε̂, k)      # optionally low-temperature
end Loop

return x_0                                 # e.g. a generated state trajectory
# Decision Diffuser then applies inverse dynamics: a_t = f_φ(s_t, s_{t+1})
```

## Connections

- Conditioning mechanism used by: [[Decision Diffuser]]
- Action recovery after generation: [[Inverse Dynamics Model]]
- Sits inside: [[Offline Reinforcement Learning]] (generate high-return plans from a fixed dataset)
- Conceptual sibling: classifier guidance (original *Diffuser*, Janner et al.) — uses an explicit return-predictor gradient
- Related conditioning idea in RL: [[Decision Transformer]] conditions on return-to-go via the input sequence rather than diffusion guidance
- Contrast with value-based offline RL: [[Conservative Q-Learning (CQL)]]
- Builds on denoising diffusion probabilistic models (DDPM) from generative vision

## Appears In

- [[RL-L11 - SAC, Decision Transformer & Diffuser]]
