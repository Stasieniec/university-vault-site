---
type: concept
aliases: [UCB, upper confidence bound, UCB1]
course: [RL]
tags: [foundations]
status: complete
---

# Upper Confidence Bound (UCB)

> [!formula] UCB Action Selection
> $$A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$
> 
> where:
> - $Q_t(a)$ — estimated value (exploitation term)
> - $c\sqrt{\frac{\ln t}{N_t(a)}}$ — exploration bonus (decreases as action $a$ is tried more)
> - $N_t(a)$ — number of times action $a$ has been selected
> - $c > 0$ — controls degree of exploration

> [!intuition] Optimism in the Face of Uncertainty
> UCB adds a bonus to actions that haven't been tried much. The less you know about an action, the higher its bonus. As you try it more, the bonus shrinks. This systematically explores uncertain options before settling on the best.

More principled than [[Epsilon-Greedy Policy]] — preferentially explores uncertain actions rather than exploring uniformly at random.

## Appears In

- [[RL-L01 - Intro, MDPs & Bandits]], [[RL-Book Ch2 - Multi-Armed Bandits]]
