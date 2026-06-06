---
type: concept
aliases: [Diffusion Model]
course: [RecSys]
tags: [generative-rec, sequential-rec, exam-topic]
status: complete
---

# Diffusion Models

## Definition

> [!definition] Diffusion Model
> A **diffusion model** is a generative model that learns a data distribution $p(\mathbf{x}_0)$ by reversing a fixed **noising process**. A *forward* (diffusion) process gradually corrupts a data point $\mathbf{x}_0$ into pure Gaussian noise $\mathbf{x}_T$ over $T$ steps; a learned *reverse* (denoising) process $p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)$ removes noise step-by-step to sample new data from noise. In recommendation it is the **non-LLM generative backbone** of slide RS-L03b's "Generative Model (e.g., LLM, Diffusion)" box: it can either denoise recommender embeddings back onto the existing item pool, or generate new item content.

## Intuition

> [!intuition] Learn to denoise, then run the camera backwards
> Imagine slowly adding static to a photo until nothing is left but noise — that is the *fixed* forward process, no learning required. The model's only job is to learn the **reverse** step: given a noisy image at level $t$, predict the noise that was added, and subtract a bit of it. Stack $T$ such tiny denoising steps and you can start from pure noise and walk back to a clean sample. Because each step solves an easy regression problem (predict the noise), training is stable — unlike a GAN's adversarial game. For recommendation, "the photo" can be a user/item embedding (denoise it toward a real catalogue item, conditioned on collaborative signal) or actual content like a fashion image.

## Mathematical Formulation

> [!formula] Forward noising, reverse denoising, and the training loss (DDPM)
> **Forward process** (fixed, no parameters) — add Gaussian noise on a variance schedule $\beta_1,\dots,\beta_T$:
> $$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\big(\mathbf{x}_t;\ \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t \mathbf{I}\big)$$
> A convenient closed form lets us jump to any step $t$ in one shot (with $\alpha_t = 1-\beta_t$, $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$):
> $$\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\qquad \boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0},\mathbf{I})$$
> **Reverse process** (learned) — a network parameterizes each denoising step:
> $$p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t) = \mathcal{N}\!\big(\mathbf{x}_{t-1};\ \boldsymbol\mu_\theta(\mathbf{x}_t, t),\ \Sigma_\theta(\mathbf{x}_t, t)\big)$$
> **Training objective** — instead of the full variational bound, DDPM trains a noise-predictor $\boldsymbol\epsilon_\theta$ with a simple MSE:
> $$\mathcal{L}_{\text{simple}} = \mathbb{E}_{\mathbf{x}_0,\ \boldsymbol\epsilon,\ t}\Big[\ \big\lVert \boldsymbol\epsilon - \boldsymbol\epsilon_\theta\big(\sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\ t\big)\big\rVert^2\ \Big]$$
>
> where:
> - $\mathbf{x}_0$ — clean data sample (an image, or a recommender embedding)
> - $\mathbf{x}_t$ — the sample after $t$ noising steps; $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0},\mathbf{I})$
> - $\beta_t \in (0,1)$ — variance schedule (how much noise is added at step $t$)
> - $\alpha_t = 1-\beta_t$, $\bar\alpha_t = \prod_{s\le t}\alpha_s$ — cumulative signal retained up to step $t$
> - $\boldsymbol\epsilon$ — the Gaussian noise actually added; $\boldsymbol\epsilon_\theta$ — the network's prediction of it
> - $t$ — diffusion step, sampled uniformly from $\{1,\dots,T\}$ during training
> - $\boldsymbol\mu_\theta, \Sigma_\theta$ — mean/covariance of the learned reverse step (recoverable from $\boldsymbol\epsilon_\theta$)
>
> **Conditioning.** To make sampling controllable (the recommendation use), the denoiser takes a condition $\mathbf{c}$ — e.g., user history or a CF embedding — giving $\boldsymbol\epsilon_\theta(\mathbf{x}_t, t, \mathbf{c})$, optionally combined with [[Classifier-Free Guidance]].

## Key Properties / Variants

- **Sampling (ancestral) algorithm.** Generation is the reverse loop, one denoising step at a time:

```pseudo
Algorithm: DDPM Sampling (conditioned on c)
──────────────────────────────────────────────
x_T ~ N(0, I)                       # start from pure noise
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1 else z = 0
    eps = eps_θ(x_t, t, c)          # predict the noise (c = user/CF condition)
    # one reverse step: subtract a scaled portion of predicted noise
    x_{t-1} = (1/√α_t) * ( x_t - (β_t / √(1-ᾱ_t)) * eps ) + √β_t * z
return x_0                          # clean sample (embedding or content)
```

- **Why it is stable to train.** The loss is a plain per-step MSE on noise — no adversarial min-max (unlike GANs), no autoregressive token ordering. This is the key contrast with the [[Autoregressive Generation|autoregressive]] semantic-ID decoders (TIGER-style) that dominate the rest of the GenRec lecture.
- **Three roles "generative" plays in RecSys** (RS-L04 slide 3 explicitly disambiguates the term):
  1. *Generate item identifiers* — autoregressive over semantic IDs (TIGER); **not** a diffusion model. This is the main lecture focus.
  2. *Diffusion for embedding denoising* (DDRM, SIGIR 2024) — diffusion denoises user/item embeddings; **collaborative signal conditions the reverse process**; output is grounded in the existing item pool, so **no new item content is created**.
  3. *Diffusion for content generation* (DiFashion, SIGIR 2024) — generates **new** item content (fashion images) conditioned on user history + constraints.
- **Trainable or frozen.** On RS-L03b's generative-recommender diagram the backbone carries both flame and snowflake icons — a diffusion denoiser can be trained on platform data or used as a frozen pretrained generator.
- **Continuous vs. discrete output.** Diffusion operates naturally on continuous vectors (embeddings, pixels). To recommend a real item it must be **grounded**: either denoise toward and look up the nearest catalogue embedding (DDRM), or pair with a retrieval/ranking step — analogous to the validity/grounding problem the autoregressive route solves with a trie.
- **Latent diffusion.** Running the process in a compressed latent space (rather than raw pixels/full embeddings) cuts cost — the standard trick for image generators and applicable to large recommender embedding spaces.
- **Cost.** Sampling needs many sequential reverse steps ($T$ can be hundreds), so inference latency is a real concern under a recommendation serving budget, mirroring the decoding-cost limitation of generative recommenders generally.

## Connections

- Sibling generative backbone to the [[Autoregressive Generation|autoregressive]] semantic-ID route in [[Generative Recommendation]] (the "LLM, Diffusion" alternatives)
- Controllable sampling via [[Classifier-Free Guidance]]
- Contrast with adversarial / token-by-token generators; aligns with [[Next-Item Prediction]] when conditioned on a user history
- Used as a planner in RL via the [[Decision Diffuser]] / [[Diffusion Models|decision diffusion]] line (sequence generation over trajectories)
- Embedding-denoising variant grounds output using ideas from [[Dense Retrieval]] (nearest-item lookup)
- Quantization-based alternative for discrete item codes: [[RQ-VAE]] / [[Semantic IDs]]

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
