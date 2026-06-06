---
type: concept
aliases: [Upside-Down Reinforcement Learning, UDRL]
course: [RL]
tags: [deep-rl, offline-rl, policy-gradient, exam-topic]
status: complete
---

# Upside-Down RL

## Definition

> [!definition] Upside-Down RL (UDRL)
> **Upside-Down Reinforcement Learning** turns RL "on its head": instead of using rewards to *optimize* a value function or policy, it uses rewards (and desired horizons) as **inputs** to a policy that is trained by **plain supervised learning**. The agent learns a **command-conditioned policy** $\pi(a \mid s, c)$ where the command $c$ specifies a **desired return** $d^r$ to obtain over a **desired horizon** $d^h$. At test time you *command* the behavior you want by feeding in a target return, and the policy maps it to actions. There are no value functions, no Bellman backups, and no policy gradients.

## Intuition

> [!intuition] Rewards as Inputs, Not Targets
> Standard RL asks: "what action maximizes expected future reward?" UDRL flips this: "given that I *want* to collect return $d^r$ within $d^h$ steps, and I am in state $s$, which action achieves that?"
>
> Because every logged trajectory **did** achieve *some* return over *some* horizon, every trajectory is a valid supervised training example — you just relabel it with the return it actually obtained (**hindsight relabeling**). A mediocre rollout teaches the policy what low-return behavior looks like; a good rollout teaches high-return behavior. Then, at deployment, you simply command a return at least as high as the best you have seen, and the policy extrapolates to expert-level behavior. This is exactly the conditioning idea later scaled up by the [[Decision Transformer]] (which adds a sequence model over history) and [[Decision Diffuser]].

## Mathematical Formulation

The agent learns a **behavior function** $B$ (the command-conditioned policy), mapping a state and a command to a distribution over actions:

$$\pi_\theta(a_t \mid s_t, c_t), \qquad c_t = (d^r_t, d^h_t)$$

where:
- $s_t$ — current state (observation)
- $c_t = (d^r_t, d^h_t)$ — the **command**: a desired return $d^r_t$ to be achieved within a desired horizon (time budget) $d^h_t$
- $\theta$ — parameters of the behavior function $B$ (typically a neural network / [[MLP]])

**Hindsight relabeling.** Given a logged trajectory segment from time $t_1$ to $t_2$ drawn from the replay buffer, the realized command that *was actually satisfied* is computed and used as the supervised input:

$$d^r = \sum_{k=t_1}^{t_2-1} r_k, \qquad d^h = t_2 - t_1$$

where:
- $\sum_{k=t_1}^{t_2-1} r_k$ — the observed return over the segment (the return-to-go that this segment actually delivered)
- $t_2 - t_1$ — the number of steps the segment actually took

**Training objective (supervised behavioral cloning under relabeled commands).** The policy is trained to reproduce the action that was taken in each relabeled $(s_t, c_t)$ pair, by maximum likelihood / minimizing cross-entropy (discrete actions) or MSE (continuous actions):

$$\mathcal{L}(\theta) = -\,\mathbb{E}_{(s_t, a_t, c_t)\sim \mathcal{D}}\big[\log \pi_\theta(a_t \mid s_t, c_t)\big]$$

where:
- $\mathcal{D}$ — the replay buffer of past episodes, with each transition relabeled by the command it satisfied in hindsight
- $-\log \pi_\theta(a_t \mid s_t, c_t)$ — negative log-likelihood of the action actually taken, given the state and the achieved command

> [!intuition] Why This Is "Supervised", Not RL
> The loss above is an ordinary classification/regression loss — the reward never appears as something to *maximize*. The reward only enters through the command $c_t$ on the **input** side. The optimization is therefore stable, gradient-friendly supervised learning, sidestepping the [[Deadly Triad]] and the brittleness of bootstrapped value estimation.

## Key Properties / Variants

- **No value function, no policy gradient**: avoids [[Bootstrapping]], TD targets, and [[Policy Gradient Theorem|policy gradients]] entirely — purely supervised.
- **Learns from suboptimal data**: every trajectory is usable via hindsight relabeling, unlike naive behavioral cloning which needs expert demonstrations.
- **Command extrapolation at test time**: command a return higher than any seen in training to elicit better-than-demonstration behavior (within the limits of generalization).
- **Single-state input (the distinguishing feature vs. Decision Transformer)**: classic UDRL conditions on the *current* state $s_t$ only — there is no sequence model over trajectory history. The [[Decision Transformer]] generalizes UDRL by feeding a windowed sequence of past $(\hat{G}, s, a)$ triples into a GPT-style transformer.
- **Online or offline**: the original formulation alternates between collecting fresh episodes (commanding ambitious returns) and supervised fits, but the same machinery applies directly to [[Offline Reinforcement Learning|offline RL]] on a fixed dataset.
- **Family relatives**: closely related to Reward-Conditioned Policies (Kumar et al.) and a deep-learning cousin of [[Reward-Weighted Regression]] / RWR-style "imitate the good bits" methods.

```pseudo
Algorithm: Upside-Down RL (Command-Conditioned Behavior Learning)
─────────────────────────────────────────────────────────────────
Initialize behavior function B_θ (policy π_θ(a | s, c))
Initialize replay buffer D with a few (possibly random) episodes

Loop:
  # ---- Supervised training phase ----
  for each training step:
    Sample an episode (or segment t1..t2) from D
    Pick a time t in the segment
    Compute achieved command in hindsight:
       d^r ← Σ_{k=t}^{t2-1} r_k       # observed return-to-go
       d^h ← t2 - t                   # remaining horizon
    Set input c_t ← (d^r, d^h), target ← a_t
    Update θ by SGD on  -log π_θ(a_t | s_t, c_t)

  # ---- Exploration / data-collection phase ----
  Construct an ambitious command c0 = (d^r, d^h):
     e.g. d^r ← (mean + std) of returns of the best recent episodes
          d^h ← typical length of those episodes
  Reset env, observe s0
  for each step t until done or d^h_t = 0:
    a_t ~ π_θ(· | s_t, c_t)           # act by command
    Take a_t, observe r_t, s_{t+1}
    d^r_{t+1} ← d^r_t - r_t           # decrement desired return
    d^h_{t+1} ← d^h_t - 1             # decrement desired horizon
  Add the new episode to D (replacing worst, fixed-size buffer)
```

> [!warning] Shares Monte-Carlo Weaknesses
> Because UDRL conditions on **observed full-trajectory returns** rather than bootstrapped estimates, it inherits the limitations of [[Monte Carlo Methods]]: high variance in returns, difficulty in long-horizon credit assignment, and sensitivity to the return distribution in the data. It is also only as good as the relationship between commanded and achieved returns generalizes — commanding an unrealistically high return need not produce it.

## Connections

- Precursor / single-state special case of: [[Decision Transformer]]
- Conceptual sibling: [[Decision Diffuser]] (generative conditioning instead of direct policy regression)
- "Imitate the good bits" family: [[Reward-Weighted Regression]], Reward-Conditioned Policies
- Applied to: [[Offline Reinforcement Learning]]
- Contrast with value-based offline RL: [[Conservative Q-Learning (CQL)]]
- Avoids: [[Policy Gradient Theorem|policy gradients]], [[Bootstrapping]], the [[Deadly Triad]]
- Related supervised baseline it improves on: behavioral cloning

## Appears In

- [[RL-L11 - SAC, Decision Transformer & Diffuser]]
