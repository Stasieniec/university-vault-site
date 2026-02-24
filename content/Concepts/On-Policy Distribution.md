---
type: concept
aliases: [Stationary Distribution under Policy]
course: [RL]
tags: [approximation]
status: complete
---

# On-Policy Distribution

> [!definition] On-Policy Distribution ($\mu$)
> The **On-Policy Distribution**, denoted as $\mu(s)$, is the stationary distribution of states encountered while following a policy $\pi$. It represents the fraction of time the agent spends in each state $s$ in the limit as time goes to infinity.

> [!formula] The $\mu(s)$ Weighting
> In function approximation, the on-policy distribution is used to weight the Mean Squared Value Error (MSVE):
> $$\overline{\text{VE}}(w) = \sum_{s \in \mathcal{S}} \mu(s) [v_\pi(s) - \hat{v}(s, w)]^2$$
> 
> where:
> - $\mu(s)$ — the probability of being in state $s$ under policy $\pi$.
> - $v_\pi(s)$ — true value function.
> - $\hat{v}(s, w)$ — approximate value function with parameters $w$.

> [!intuition] Importance Weighting
> We cannot usually approximate the value function perfectly for all states. The on-policy distribution tells us which states are the most important to get right—namely, those we visit most often. If $\mu(s)$ is high, an error in state $s$ contributes more to the total loss than an error in a state the agent rarely visits.

## Properties in RL

- **Self-Weighting**: In on-policy learning, the updates naturally follow $\mu(s)$ because states are sampled by interacting with the environment using $\pi$.
- **Objective Function**: It defines the "average" error we are trying to minimize during [[Gradient Descent|gradient descent]] updates to the value function parameters.

## Connections

- Used in: [[Value Function Approximation]], [[Mean Squared Value Error (MSVE)]]
- Contrast with: [[Off-Policy Learning]], which often uses an *off-policy* distribution (from a behavior policy).

## Appears In

- [[RL-L06 - Value Function Approximation]]
