---
type: concept
aliases: [TD learning, temporal difference, TD]
course: [RL]
tags: [tabular-methods, exam-topic, key-formula]
status: complete
---

# Temporal Difference Learning

## Definition

> [!definition] Temporal Difference (TD) Learning
> **TD learning** combines ideas from [[Monte Carlo Methods]] and [[Dynamic Programming]]. Like MC, it learns from experience without a model. Like DP, it updates estimates based on other estimates (**bootstrapping**) without waiting for the end of an episode.

## TD(0) — The Core Update

> [!formula] TD(0) Update Rule
> $$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$
> 
> where:
> - $\alpha$ — learning rate (step size)
> - $R_{t+1} + \gamma V(S_{t+1})$ — **TD target** (one-step bootstrap estimate of $G_t$)
> - $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ — **[[TD Error]]** $\delta_t$

> [!intuition] The Key Idea
> Instead of waiting for the actual return $G_t$ (like MC does), TD uses the estimate $R_{t+1} + \gamma V(S_{t+1})$ as a target. It updates $V(S_t)$ **immediately** after one transition — no need to wait for the episode to end.

## Comparison: TD vs MC vs DP

| Property | [[Dynamic Programming|DP]] | [[Monte Carlo Methods|MC]] | **TD** |
|----------|:---:|:---:|:---:|
| Requires model? | ✅ | ❌ | ❌ |
| Bootstraps? | ✅ | ❌ | ✅ |
| Learns from experience? | ❌ | ✅ | ✅ |
| Requires complete episodes? | N/A | ✅ | ❌ |
| Online (step-by-step)? | N/A | ❌ | ✅ |

> [!intuition] The Best of Both Worlds
> TD = sample-based (like MC) + bootstrapping (like DP). It doesn't need a model AND doesn't need to wait for episode termination.

## TD for Control

### [[SARSA]] — On-Policy TD Control

> [!formula] SARSA Update
> $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$
> 
> Name comes from the quintuple: $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$

### [[Q-Learning]] — Off-Policy TD Control

> [!formula] Q-Learning Update
> $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$
> 
> Uses $\max_a$ instead of following the actual next action — learns about the **greedy** (optimal) policy regardless of what action the behavior policy takes.

### [[Expected SARSA]]

> [!formula] Expected SARSA Update
> $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$
> 
> Takes the expectation over next actions under the policy instead of sampling a single $A_{t+1}$. Lower variance than SARSA.

## Backup Diagrams

**TD(0):**
```
  (S_t)
    |
   [A_t]    ← single sampled action
    |
  (S_{t+1}) ← single sampled next state, uses V(S_{t+1})
```
Samples one step, bootstraps from the estimate. Contrast with MC (samples to end) and DP (considers all branches).

## Key Properties

- **Biased but consistent**: TD targets are biased (use estimates), but converge to correct values
- **Lower variance than MC**: Because it doesn't use the full noisy return
- **Can learn online**: Updates after every step, no need to wait for episode end
- **Works for continuing tasks**: Unlike MC which needs episode termination
- **TD(0) converges** to $v_\pi$ under appropriate step-size conditions (tabular case)

## Connections

- Combines: [[Monte Carlo Methods]] (sampling) + [[Dynamic Programming]] (bootstrapping)
- Uses: [[TD Error]], [[Bootstrapping]]
- Control algorithms: [[SARSA]], [[Q-Learning]], [[Expected SARSA]]
- Extended by: [[Semi-Gradient Methods]], [[Function Approximation]]
- Deep version: [[Deep Q-Network (DQN)]]

## Appears In

- [[RL-L04 - Temporal Difference Learning]]
- [[RL-Book Ch6 - Temporal-Difference Learning]]
- [[RL-ES02 - Exercise Set Week 2]]
- [[RL-CA03 - Temporal Difference]]
