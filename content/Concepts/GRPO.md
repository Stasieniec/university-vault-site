---
type: concept
aliases: [Group Relative Policy Optimization, GRPO]
course: [RL, IR]
tags: [policy-gradient, deep-rl, llm-training]
status: complete
---

# Group Relative Policy Optimization (GRPO)

> [!definition] GRPO
> A [[Policy Gradient Methods|policy gradient]] algorithm that estimates advantages through **group-relative comparisons** rather than a learned value function. For each prompt, multiple responses are sampled and their rewards are normalized within the group to compute advantages.

## Motivation

Standard [[PPO]] requires a **critic network** to estimate the value function $V(s)$ for advantage computation. For large language models:
- Adding a critic roughly doubles parameters
- Training the critic is challenging (moving target)
- Critic quality directly impacts gradient quality

GRPO eliminates the critic by using **within-group statistics**.

## Algorithm

For each prompt $x$, sample $G$ responses $\{y_1, \ldots, y_G\}$ from the current policy $\pi_\theta$.

> [!formula] **GRPO Advantage**
> $$\hat{A}_i = \frac{R(x, y_i) - \mu_G}{\sigma_G}$$
>
> where:
> - $\mu_G = \frac{1}{G} \sum_{j=1}^G R(x, y_j)$ — group mean reward
> - $\sigma_G = \text{std}(\{R(x, y_j)\}_{j=1}^G)$ — group standard deviation
> - The advantage is the **z-score** of the response within its group

**Gradient update**:
$$\nabla_\theta J \approx \frac{1}{|B|} \sum_{x \in B} \frac{1}{G} \sum_{i=1}^G \hat{A}_i \nabla_\theta \log \pi_\theta(y_i | x)$$

## Comparison with PPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| **Advantage source** | Learned $V_\phi(s)$ | Group statistics |
| **Additional networks** | Critic | None |
| **Samples per prompt** | 1 | $G$ (typically 4-16) |
| **Memory** | High | Low |
| **Implementation** | Complex | Simple |

## Intuition

> [!intuition] **Why Group Normalization Works**
> - Responses are compared **relative to each other**, not absolute value
> - Good responses get positive advantage, bad ones get negative
> - Automatically adapts to reward scale
> - Handles sparse rewards (0/1) naturally

## Properties

- **No critic training**: Simpler optimization landscape
- **Automatic baseline**: Group mean serves as baseline
- **Variance normalization**: Group std normalizes gradient scale
- **Sample efficient**: Reuses multiple samples per prompt

## Connections

- Simplifies [[PPO]] by removing critic
- Used in [[DeepSeek-R1]] and [[SEARCH-R1]]
- Related to [[REINFORCE]] with baseline
- Alternative to [[Actor-Critic]] methods

## Appears In

- [[IR-L13 - RL for Reasoning and Search]]
- DeepSeek-R1 paper (2025)
- SEARCH-R1 paper (2025)
