---
type: concept
aliases: [HRL]
course: [RL]
tags: [temporal-abstraction, exploration, exam-topic]
status: complete
---

# Hierarchical Reinforcement Learning

## Definition

> [!definition] Hierarchical Reinforcement Learning (HRL)
> **HRL** decomposes a control problem into a hierarchy of policies operating at **different levels of temporal abstraction**. A **high-level policy** selects *subgoals* or *temporally-extended actions* (skills), and a **low-level policy** executes primitive actions to fulfil them. The central object is the **option** $\omega = \langle \mathcal{I}_\omega, \pi_\omega, \beta_\omega \rangle$: an action that, once invoked, runs for many primitive time steps before returning control. This turns a flat [[Markov Decision Process|MDP]] over primitive actions into a **Semi-Markov Decision Process (SMDP)** over options.

## Intuition

> [!intuition] Why decompose into a hierarchy
> A flat agent must learn long, brittle sequences of primitive actions, and exploration via [[Epsilon-Greedy|ε-greedy]] jitter rarely strings together the hundreds of correct micro-decisions needed to reach a distant reward. HRL attacks this with **temporal abstraction**: the high level reasons in coarse, reusable chunks ("walk to the door", "navigate to city X"), so a single high-level decision commits the agent to a *consistent multi-step behaviour*. This shortens the effective horizon the top level sees, makes exploration directed (the agent jumps between subgoals rather than wiggling), and lets learned skills **transfer** across tasks that share sub-behaviours.
>
> In the [[RL-ES01 - Exercise Set Week 1]] driving example: a *low-level* controller learns "how to drive" (accelerator/brake), while a *high-level* controller learns "where to go" — the hybrid is exactly HRL.

## Mathematical Formulation

An **option** $\omega$ over an MDP is the triple

$$\omega = \langle\, \mathcal{I}_\omega,\; \pi_\omega,\; \beta_\omega \,\rangle$$

where:
- $\mathcal{I}_\omega \subseteq \mathcal{S}$ — **initiation set**, the states in which $\omega$ may be started
- $\pi_\omega(a \mid s)$ — the option's **internal (low-level) policy** over primitive actions
- $\beta_\omega(s) \in [0,1]$ — **termination condition**, the probability the option ends in state $s$

The high-level policy $\mu(\omega \mid s)$ chooses options. Because an option runs for a random number of steps $k$, the system is an **SMDP**. The Bellman equation for the option-value function $Q_\mu(s,\omega)$ uses **multi-step, discounted** option models:

$$Q_\mu(s,\omega) = \sum_{s',\,k} P(s',k \mid s,\omega)\,\Big[\, r(s,\omega) + \gamma^{k}\, \textstyle\sum_{\omega'} \mu(\omega' \mid s')\, Q_\mu(s',\omega') \,\Big]$$

where:
- $r(s,\omega) = \mathbb{E}\!\left[ R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{k-1} R_{t+k} \mid s, \omega \right]$ — expected **accumulated** reward while $\omega$ executes
- $k$ — (random) number of primitive steps until $\beta_\omega$ triggers termination
- $\gamma^{k}$ — discount applied across the *whole* option duration, not a single step
- $P(s',k \mid s,\omega)$ — joint probability of terminating in $s'$ after exactly $k$ steps

**Intra-option / SMDP Q-learning update** (learning the high level while options run):

$$Q(s_t, \omega) \leftarrow Q(s_t, \omega) + \alpha \Big[\, \underbrace{r + \gamma^{k} \max_{\omega'} Q(s_{t+k}, \omega')}_{\text{SMDP TD target}} - Q(s_t, \omega) \,\Big]$$

where $r$ is the accumulated discounted reward over the $k$ steps the option ran and $s_{t+k}$ is the state at termination. The low-level $\pi_\omega$ is trained separately, typically on an **intrinsic/subgoal reward** $r^{\text{int}}$ rather than the environment reward.

## Key Properties / Variants

- **Options framework** (Sutton, Precup & Singh): the canonical formalism above; a primitive action is just an option with $\beta \equiv 1$ that lasts one step, so HRL strictly generalizes flat RL.
- **Feudal / goal-conditioned HRL** (FeUdal Networks, HIRO): the high-level policy emits a **goal vector** $g_t$ every $c$ steps; the low-level policy is goal-conditioned $\pi(a \mid s, g)$ and rewarded for reaching $g_t$. HIRO uses **off-policy goal relabelling** to make manager transitions valid as the worker changes.
- **Option-Critic**: learns option policies $\pi_\omega$ *and* terminations $\beta_\omega$ end-to-end with [[Policy Gradient]]s — no hand-designed subgoals.
- **Benefits**: directed **exploration** over a shorter effective horizon; **transfer** and **reuse** of skills across tasks; mitigates the curse of dimensionality / sparse rewards.
- **Difficulties**: *non-stationarity* (the low level shifts under the high level during joint training); discovering useful subgoals/options automatically is hard; defining good intrinsic rewards and termination is delicate.

```pseudo
Algorithm: SMDP Q-Learning over Options (high-level control)
────────────────────────────────────────────────────────────
Initialize Q(s, ω) for all states s and options ω

Loop for each episode:
  Initialize S
  Loop until S terminal:
    Choose option ω from S using policy from Q   (e.g. ε-greedy over ω ∈ available(S))
    r ← 0;  τ ← 0                                  # accumulated reward, elapsed steps
    Loop (execute the option):
      Choose primitive A ~ π_ω(·|S)
      Take A, observe R, S'
      r ← r + γ^τ · R
      τ ← τ + 1
      S ← S'
    until terminate with prob. β_ω(S)  or  S terminal
    # high-level (SMDP) update spanning the whole option
    Q(S_start, ω) ← Q(S_start, ω) + α [ r + γ^τ · max_ω' Q(S, ω') − Q(S_start, ω) ]
    S_start ← S
```

## Connections

- Generalizes: [[Markov Decision Process]] — flat MDP becomes an SMDP over options (primitive action = 1-step option)
- Builds on: [[Q-Learning]] / [[Temporal Difference Learning]] (the SMDP Q-update), [[Discount Factor]] (applied over option durations)
- Low-level skills trained via: [[Policy Gradient]] / [[Actor-Critic]] (e.g. Option-Critic)
- Addresses: [[Exploration vs Exploitation]] under sparse, long-horizon rewards; the curse of dimensionality
- Contrast: a flat [[Optimal Policy]] over primitive actions vs. a hierarchy of [[Policy|policies]]

## Appears In

- [[RL-ES01 - Exercise Set Week 1]]
