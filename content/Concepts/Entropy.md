---
type: concept
aliases: [Shannon Entropy]
course: [RL]
tags: [policy-gradient, exploration, deep-rl, exam-topic]
status: complete
---

# Entropy

## Definition

> [!definition] (Shannon) Entropy
> The **entropy** of a discrete probability distribution $p$ measures its **uncertainty** — the expected amount of "surprise" (in nats if using $\ln$, or bits if using $\log_2$) when sampling from it. For a policy $\pi(\cdot\mid s)$ over actions, the entropy is
> $$H(\pi(\cdot\mid s)) = -\sum_{a} \pi(a\mid s)\,\log \pi(a\mid s) = \mathbb{E}_{a\sim\pi}\!\left[-\log \pi(a\mid s)\right].$$
> It is **maximized by the uniform distribution** (maximum uncertainty / exploration) and **minimized (=0) by a deterministic distribution** (a point mass — full certainty / greedy).

## Intuition

Entropy answers: "how spread out is this distribution?"

- A **uniform** policy over $|A|$ actions has the largest entropy, $\log |A|$ — every action is equally likely, so you are maximally uncertain and maximally exploratory.
- A **deterministic / peaked** policy (one action has probability $\approx 1$) has entropy $\approx 0$ — no surprise, but also no exploration.

In RL we exploit this directly: adding an **entropy term** to the objective discourages the policy from collapsing too early onto a single action. This keeps the policy stochastic, preserving exploration and preventing premature convergence to a suboptimal deterministic policy. The information-theoretic reading is that $-\log p(x)$ is the **self-information** ("surprisal") of outcome $x$; entropy is its expectation.

## Mathematical Formulation

**Entropy of a policy.** For state $s$,
$$H(\pi(\cdot\mid s)) = -\sum_{a\in A} \pi_\theta(a\mid s)\,\log \pi_\theta(a\mid s).$$

where:
- $\pi_\theta(a\mid s)$ — probability the policy assigns to action $a$ in state $s$
- the sum runs over all actions; for continuous actions it becomes an integral (differential entropy)
- $H \ge 0$ for discrete distributions, with $0 \le H \le \log|A|$

**Entropy regularization (entropy bonus).** Policy-gradient methods add an entropy term to encourage exploration. For [[REINFORCE]] / [[Actor-Critic]] the per-step objective gradient becomes
$$\nabla_\theta J(\theta) \;\propto\; \mathbb{E}\!\left[\nabla_\theta \log \pi_\theta(a\mid s)\,\big(G_t - b(s)\big) \;+\; \beta\,\nabla_\theta H(\pi_\theta(\cdot\mid s))\right].$$

where:
- $G_t - b(s)$ — return minus [[Baseline]] (the [[Advantage]] signal driving the policy update)
- $\beta$ — entropy coefficient (regularization strength); larger $\beta \Rightarrow$ more exploration
- $\nabla_\theta H$ — pushes $\pi_\theta$ toward higher entropy (more uniform)

**Maximum-entropy objective.** [[Soft Actor-Critic (SAC)]] augments the reward with an entropy term at every step, yielding the [[Maximum Entropy RL]] objective
$$J(\pi) = \sum_{t} \mathbb{E}_{(s_t,a_t)\sim\pi}\!\left[\, r(s_t,a_t) \;+\; \alpha\, H\big(\pi(\cdot\mid s_t)\big)\,\right].$$

where:
- $r(s_t,a_t)$ — environment reward
- $\alpha$ — **temperature**, trading off reward vs. entropy ($\alpha \to 0$ recovers standard RL)
- $H(\pi(\cdot\mid s_t))$ — policy entropy, here treated as an intrinsic reward for acting stochastically

## Key Properties / Variants

- **Bounds:** $0 \le H(\pi) \le \log|A|$ (discrete). Maximum at uniform $\pi$, minimum at a deterministic $\pi$.
- **Concavity:** $H$ is a concave function of the distribution, so an entropy bonus is a concave regularizer (well-behaved for gradient ascent).
- **Self-information:** $H = \mathbb{E}[-\log p(x)]$; the integrand $-\log p(x)$ is the surprisal of a single outcome.
- **Relation to cross-entropy / KL:** $D_{\mathrm{KL}}(p\,\|\,q) = \underbrace{\mathbb{E}_p[-\log q]}_{\text{cross-entropy}} - \underbrace{\big(-\mathbb{E}_p[-\log p]\big)}_{H(p)}$, i.e. cross-entropy $=$ entropy $+$ KL divergence. Minimizing KL with fixed $p$ is the same as minimizing cross-entropy.
- **Temperature link:** in a [[Softmax Policy]] $\pi(a\mid s)\propto \exp(f_\theta(s,a)/\tau)$, raising $\tau$ raises entropy (toward uniform); lowering $\tau\to 0$ drives entropy to $0$ (toward argmax).
- **Differential entropy:** for a continuous policy (e.g. a [[Gaussian Policy]]) entropy depends on the variance; a Gaussian's entropy is $\tfrac{1}{2}\log(2\pi e\,\sigma^2)$ per dimension. Unlike the discrete case it can be negative.

Computing an entropy bonus for a softmax policy:

```pseudo
Function: entropy_bonus(logits, beta)
─────────────────────────────────────
  p   ← softmax(logits)                 # action probabilities π(a|s)
  logp ← log_softmax(logits)            # numerically stable log π(a|s)
  H   ← -Σ_a  p[a] * logp[a]            # Shannon entropy of the policy
  return beta * H                       # add to objective (gradient ASCENT on H)
```

> [!warning] Sign and Coefficient
> Entropy is **added** to the objective for gradient *ascent* (or its negative is *subtracted* from a loss for gradient descent). Get the sign wrong and you penalize exploration, collapsing the policy. The coefficient ($\beta$ or temperature $\alpha$) must be tuned/annealed: too high keeps the policy near-uniform and it never exploits; too low gives no exploration benefit. In SAC, $\alpha$ is often learned automatically to hit a target entropy.

## Connections

- Regularizes / explores in: [[Softmax Policy]], [[REINFORCE]], [[Actor-Critic]], [[PPO]], [[A3C]]
- Core of: [[Maximum Entropy RL]], [[Soft Actor-Critic (SAC)]]
- Continuous-action entropy: [[Gaussian Policy]]
- Alternative to exploration via: [[Epsilon-Greedy]], [[Optimistic Initial Values]]
- Information-theoretic relatives: cross-entropy, KL divergence

## Appears In

- [[Softmax Policy]] — uses policy entropy as its built-in exploration mechanism
- [[RL-L11 - SAC, Decision Transformer & Diffuser]]
- [[RL-L09 - Policy Gradient Methods]]
- [[RL-L10 - Advanced Policy Search]]
