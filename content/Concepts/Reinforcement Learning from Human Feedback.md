---
type: concept
aliases: [RLHF]
course: [IR, RL, RecSys]
tags: [policy-gradient, deep-rl, exam-topic]
status: complete
---

# Reinforcement Learning from Human Feedback

## Definition

> [!definition] RLHF
> **Reinforcement Learning from Human Feedback (RLHF)** is a three-stage pipeline for aligning a pretrained generative model (typically an LLM) with human preferences. Instead of optimizing a hand-specified reward, RLHF (1) collects **pairwise human preference comparisons** over model outputs, (2) trains a **reward model** $r_\phi$ to predict those preferences, and (3) uses [[Reinforcement Learning]] (usually [[PPO|Proximal Policy Optimization]]) to fine-tune the generative **policy** $\pi_\theta$ to maximize the learned reward while staying close to the original supervised model via a **KL penalty**.

## Intuition

For tasks like helpful dialogue or summarization there is no programmable reward — "quality" lives in human judgement, and absolute scalar ratings are noisy and uncalibrated. RLHF exploits the fact that humans are far more reliable at **relative** judgements: "response A is better than response B." This is exactly the [[Pairwise LTR|pairwise]] setting from [[Learning to Rank]] — the reward model is trained with a [[RankNet]]-style logistic loss on score differences. Once a differentiable reward model captures the preference signal, we can score the model's *own* generations and push the policy toward outputs the reward model prefers.

The KL penalty is the crucial safety valve: maximizing a *learned* reward unconstrained leads to **reward hacking** (the policy finds adversarial outputs that fool $r_\phi$ but are gibberish to humans). Anchoring the policy to the supervised reference $\pi_{\text{ref}}$ keeps generations fluent and on-distribution.

## Mathematical Formulation

**Stage 1 — Supervised fine-tuning (SFT).** Start from a pretrained model and fine-tune on demonstration data to get the reference policy $\pi_{\text{ref}} = \pi^{\text{SFT}}$.

**Stage 2 — Reward model.** Given a prompt $x$ and two completions where humans labelled $y_w$ (winner) preferred over $y_l$ (loser), fit a scalar reward model $r_\phi(x,y)$ under the **Bradley–Terry** preference model:

$$P(y_w \succ y_l \mid x) = \sigma\!\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big), \qquad \mathcal{L}_{\text{RM}}(\phi) = -\,\mathbb{E}_{(x, y_w, y_l)\sim \mathcal{D}}\Big[\log \sigma\!\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]$$

where:
- $r_\phi(x,y)$ — scalar reward (a regression head on top of the LM), read off the final token
- $\sigma(z) = 1/(1+e^{-z})$ — logistic sigmoid mapping a score *difference* to a preference probability
- $y_w \succ y_l$ — human judged $y_w$ better than $y_l$ for prompt $x$
- $\mathcal{D}$ — dataset of human pairwise comparisons

Note this is **identical in form** to the [[RankNet]] logistic loss $\log(1 + e^{-(s_i - s_j)})$ on the score difference $s_i - s_j$.

**Stage 3 — RL fine-tuning.** Optimize the policy $\pi_\theta$ to maximize the learned reward minus a per-token KL penalty to the reference:

$$\max_{\theta}\; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\Big[\, r_\phi(x, y)\,\Big] \;-\; \beta\, \mathbb{E}_{x}\big[\, D_{\text{KL}}\!\big(\pi_\theta(\cdot \mid x)\,\|\,\pi_{\text{ref}}(\cdot \mid x)\big)\big]$$

equivalently optimized with [[PPO]] using a per-token shaped reward:

$$R(x, y) = r_\phi(x, y) - \beta \sum_{t} \log\frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}$$

where:
- $\pi_\theta$ — the trainable policy (LLM), initialized from $\pi_{\text{ref}}$
- $r_\phi(x,y)$ — frozen reward model from Stage 2, giving a sparse terminal reward
- $\beta$ — KL coefficient controlling how far the policy may drift from $\pi_{\text{ref}}$
- $D_{\text{KL}}$ — penalizes the policy for moving away from the SFT model (prevents reward hacking / mode collapse)

The MDP framing matches [[IR-L13 - RL for Reasoning and Search]]: state $s_t$ = prompt + tokens so far, action $a_t$ = next token, $\pi_\theta(a_t \mid s_t)$ = the LM's token distribution; the reward is sparse (delivered at the end of the completion).

## Key Properties / Variants

- **Why pairwise, not absolute:** relative preferences are cheaper and more consistent to elicit from annotators than calibrated scalar scores — directly the [[Pairwise Learning to Rank]] argument.
- **PPO is the standard RL optimizer** (InstructGPT, ChatGPT, Claude). Requires four models in memory: policy, reference, reward model, and critic (value network).
- **KL penalty is load-bearing:** without it the policy reward-hacks $r_\phi$; with too-large $\beta$ it never moves from the SFT model.
- **GRPO variant:** [[GRPO|Group Relative Policy Optimization]] drops the critic and replaces the advantage with a group-normalized z-score over $G$ sampled completions, used in [[DeepSeek-R1]] and [[SEARCH-R1]]. Cheaper but with verifiable rewards it can skip the learned reward model entirely (RL from verifiable reward).
- **DPO (Direct Preference Optimization):** reparameterizes the Stage-2/Stage-3 objective into a single supervised loss, optimizing the preference objective *directly* on $\pi_\theta$ with no explicit reward model or RL rollouts.
- **Failure modes:** reward over-optimization (Goodhart), sycophancy, distribution shift between RM training data and policy generations, annotator disagreement.

```pseudo
Algorithm: RLHF (PPO variant)
──────────────────────────────────────────────
Stage 1 — SFT:
  π_ref ← fine-tune pretrained LM on demonstration data
  π_θ   ← copy of π_ref   (the trainable policy)

Stage 2 — Reward Model:
  Collect comparisons: for prompt x, humans label y_w ≻ y_l
  Fit r_φ by minimizing:
    L_RM = - E[ log σ( r_φ(x, y_w) - r_φ(x, y_l) ) ]

Stage 3 — RL fine-tuning (PPO):
  Loop:
    Sample prompts x ~ D
    Generate completions y ~ π_θ(·|x)             # rollouts
    Compute reward:
      R(x,y) = r_φ(x,y) - β · Σ_t log[ π_θ(y_t|·) / π_ref(y_t|·) ]
    Estimate advantages Â_t (GAE, via critic V_ψ)
    For K epochs:
      r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
      maximize L^CLIP = E[ min( r_t Â_t,
                                clip(r_t, 1-ε, 1+ε) Â_t ) ]
    θ_old ← θ
```

## Connections

- Reward model loss is the [[RankNet]] / [[Pairwise LTR]] logistic loss on score differences (see [[IR-L10 - Learning to Rank]])
- RL stage uses [[PPO]], built on [[REINFORCE]] and the [[Policy Gradient Theorem]] within an [[Actor-Critic]] framework
- Advantage estimation via [[Generalized Advantage Estimation]]
- Critic-free alternative: [[GRPO]] (group-relative advantages)
- Applied to reasoning and retrieval: [[DeepSeek-R1]], [[SEARCH-R1]]
- LLM generation cast as a [[Markov Decision Process]] over tokens; the policy is the [[Policy]] $\pi_\theta$

## Appears In

- [[IR-L10 - Learning to Rank]]
- [[IR-L13 - RL for Reasoning and Search]]
- [[RS-L04 - Generative Recommendation]]
