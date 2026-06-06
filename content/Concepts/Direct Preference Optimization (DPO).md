---
type: concept
aliases: [DPO]
course: [RecSys]
tags: [generative-rec, llm, exam-topic]
status: complete
---

# Direct Preference Optimization (DPO)

> [!info] Lecture context
> Optimize directly on preferred-vs-rejected pairs without a separate reward model.

## Definition

> [!definition] Direct Preference Optimization (DPO)
> **DPO** is a preference-tuning objective that aligns a generative model on **(preferred, rejected) response pairs** *without* ever training an explicit reward model and *without* an [[RL]] loop. It is the standard "no-RL" alternative to [[Reinforcement Learning from Human Feedback|RLHF]]: instead of learning a reward $r_\phi$ and then running [[Proximal Policy Optimization|PPO]] against it, DPO shows that the optimal RLHF policy has a **closed form**, and uses that fact to fold reward learning and policy optimization into a **single classification loss** on the preference pairs.
>
> In generative recommendation it is one of four ways to shape the training objective once items are tokens — alongside [[Supervised Fine-Tuning (SFT)|SFT]], self-supervised/contrastive learning, and reward-based [[RL]] (e.g. [[GRPO]]). DPO directly teaches the model to rank a preferred next-item identifier (a positive Semantic ID) above a rejected one.

## Intuition

> [!intuition] The reward model is hiding inside the policy
> RLHF is two stages: (1) fit a reward model to preference data, (2) optimize the policy against that reward with a KL leash to a frozen reference model $\pi_{\text{ref}}$. DPO's key observation is that for the standard KL-regularized RLHF objective, the optimal policy and the reward are related in closed form — so the **reward can be written as a function of the policy itself** (specifically the log-ratio $\log \tfrac{\pi_\theta}{\pi_{\text{ref}}}$).
>
> Substituting that into the Bradley–Terry preference model collapses the whole pipeline into one logistic-regression-style loss: push up the log-probability of the **preferred** response $y_w$ relative to the reference, push down the **rejected** response $y_l$. No reward network, no sampling, no on-policy rollouts — just a supervised loss over pairs. This is why the slides list DPO as "no reward model needed; training is stable," in direct contrast to RL which is "reward-driven... needs feedback and is unstable to train."

## Mathematical Formulation

The KL-regularized RLHF objective DPO starts from is
$$
\max_{\pi_\theta}\ \mathbb{E}_{x,\,y\sim\pi_\theta}\big[r(x,y)\big] - \beta\,\mathbb{D}_{\mathrm{KL}}\!\big(\pi_\theta(y\mid x)\,\|\,\pi_{\text{ref}}(y\mid x)\big).
$$
Its optimal policy is $\pi^*(y\mid x)=\tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\exp\!\big(\tfrac{1}{\beta}r(x,y)\big)$, which can be inverted to express the reward as $r(x,y)=\beta\log\tfrac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$. Plugging this into the Bradley–Terry model $P(y_w\succ y_l)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$ makes $Z(x)$ cancel and yields the **DPO loss**:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}}) = -\,\mathbb{E}_{(x,\,y_w,\,y_l)\sim\mathcal{D}}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]
$$

where:
- $x$ — the prompt / context (in RecSys: the user interaction history, or its tokenized Semantic ID sequence)
- $y_w$ — the **preferred** ("winning") response; in GenRec, the positive next-item identifier the user actually engaged with
- $y_l$ — the **rejected** ("losing") response; a negative / dispreferred item identifier
- $\pi_\theta$ — the policy being trained (the generative model)
- $\pi_{\text{ref}}$ — the frozen reference policy, usually the [[Supervised Fine-Tuning (SFT)|SFT]] checkpoint; the KL anchor that keeps $\pi_\theta$ from drifting
- $\beta$ — temperature controlling how hard the KL constraint pulls toward $\pi_{\text{ref}}$ (larger $\beta$ = stay closer to reference)
- $\sigma$ — the logistic sigmoid; $\mathbb{D}_{\mathrm{KL}}$ — Kullback–Leibler divergence; $Z(x)$ — the (cancelled) partition function
- $\hat r_\theta(x,y)=\beta\log\tfrac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ — the **implicit reward** DPO optimizes; the loss is a binary classifier on $\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l)$

The gradient is informative:
$$
\nabla_\theta\mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}\Big[\,\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)\,\big(\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\big)\Big]
$$
It raises $\log\pi_\theta(y_w)$ and lowers $\log\pi_\theta(y_l)$, **weighted** by how badly the current implicit reward ranks the pair (the $\sigma(\cdot)$ term is large exactly when the model is wrong) — an automatic hard-example weighting that a naive log-likelihood objective lacks.

## Key Properties / Variants

- **No reward model, no RL loop.** Reward learning and policy optimization are merged into one supervised loss; there is no separate $r_\phi$ network and no PPO-style sampling. This is the main reason the lecture flags DPO as more *stable* and cheaper to train than [[RL]].
- **Reference model is required.** The frozen $\pi_{\text{ref}}$ (typically the [[Supervised Fine-Tuning (SFT)|SFT]] model) appears in every term; it both defines the implicit reward and regularizes the update. DPO is normally run *after* an SFT stage.
- **Off-policy / offline.** It learns from a fixed dataset of pre-collected preference pairs $\mathcal{D}$ — no fresh on-policy rollouts are needed, unlike [[GRPO]] or PPO.
- **$\beta$ trades fit vs. drift.** Small $\beta$ lets the policy move far from the reference (sharper preferences, more overfitting/degeneracy risk); large $\beta$ keeps it conservative.
- **Position in the GenRec objective menu (RS-L03b §4.1.3):** the four training-objective choices are SFT (positives only, weak margin), SSL/contrastive (template-robust), RL (encodes explicit negatives & non-differentiable metrics, but unstable), and **DPO** (direct preferred-vs-rejected pairs, stable). RecSys variants named in the lectures: LettinGo, RosePO, SPRec, and S-DPO (softmax/multi-negative DPO for sequential recommendation); listed alongside GRPO and Rec-R1 as preference/RL fine-tuning for generative recommenders.
- **What a "pair" is in RecSys.** $x$ = user history; $y_w$ = a positive item (its Semantic ID / identifier sequence); $y_l$ = a negative — a non-interacted, low-reward, or invalid item ID. This lets DPO inject the explicit-negative signal that plain next-item [[Supervised Fine-Tuning (SFT)|SFT]] (positives-only cross-entropy) cannot represent.

```pseudo
Algorithm: DPO (offline preference tuning)
──────────────────────────────────────────────
Inputs: SFT model π_ref (frozen), preference data D = {(x, y_w, y_l)}, β
Initialize π_θ ← π_ref

Loop over minibatches {(x, y_w, y_l)} ~ D:
    # log-probs under both models (teacher-forced over the token sequence)
    lp_w_θ   = log π_θ(y_w | x);   lp_l_θ   = log π_θ(y_l | x)
    lp_w_ref = log π_ref(y_w | x); lp_l_ref = log π_ref(y_l | x)   # no grad

    # implicit reward log-ratios
    Δ_w = lp_w_θ - lp_w_ref
    Δ_l = lp_l_θ - lp_l_ref

    loss = -log σ( β * (Δ_w - Δ_l) )      # Bradley–Terry classification
    θ ← θ - η ∇_θ loss
return π_θ
```

## Connections

- Replaces the two-stage pipeline of: [[Reinforcement Learning from Human Feedback]] (reward model + [[Proximal Policy Optimization|PPO]])
- Alternative to: [[GRPO]] (on-policy, group-relative, sampling-based reward fine-tuning) for the same "go beyond cross-entropy" goal
- Usually preceded by: [[Supervised Fine-Tuning (SFT)]] (provides the reference policy $\pi_{\text{ref}}$)
- Sits in the objective menu beside: [[Contrastive Learning]] / self-supervised pretraining, [[Negative Sampling]]
- Foundations: an instance of [[Off-Policy Learning|off-policy]] preference optimization; uses the [[Entropy|KL]]-regularized objective and a logistic (Bradley–Terry) preference model
- Applied over: [[Semantic IDs]] generated by a [[Generative Recommender]] (e.g. [[TIGER]]-style token sequences)
- Contrast in stability with: [[RL]] (reward-driven, unstable to train per the lecture)

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
- [[RS-L04 - Generative Recommendation]]
