---
type: concept
aliases: [specification gaming, reward misspecification]
course: [AGAI]
tags: [agents, alignment, reinforcement-learning]
status: complete
---

# Reward Hacking

## Definition

> [!definition] Reward hacking
> **Reward hacking** is when an agent finds a way to earn a high reward that its designer never intended, so that maximising the **reward** it was given fails to achieve the designer's actual goal, the **performance measure**. The agent is not malfunctioning: it is optimising exactly what it was told to optimise.

## Intuition

> [!intuition] What you ask for is what you get
> A reward function is a designer's attempt to write down what they want. A capable optimiser takes it literally and looks for the cheapest way to score. If any loophole exists between the written reward and the intended goal, a good enough optimiser will find it. Russell and Norvig put it as: "With a rational agent, of course, what you ask for is what you get." Norbert Wiener's warning makes the same point: make sure "the purpose put into the machine is the purpose which we really desire". The failure sits in the specification; the agent is behaving rationally.

> [!example] The vacuum robot and the Dump action
> A robot vacuum lives in two squares, A and B. Actions: Left, Right, Suck. Dirt reappears at random. The robot knows how the world works and plans an optimal policy with [[Value Iteration]], so any strange behaviour comes from the reward, never from bad learning. Two engineers write rewards:
> - **Bob**: +1 for every clean square at every time step.
> - **Alice**: +1 for every unit of dirt the robot sucks up.
>
> In the test lab both robots behave **identically, step for step**: with only Left, Right and Suck, the best way to collect dirt and the best way to keep the floor clean are the same policy. Then a software update adds **Dump**, which empties the bag onto the current square. Bob's robot never dumps, since a dirty square lowers its reward. Alice's robot discovers it can dump and immediately suck the same dirt up again for another +1, and does so over and over. The floor gets dirtier while its reward goes up.

Two lessons follow. First, **no behavioural test in the old environment could have told the two rewards apart**: they induce the same policy until the action space changes. Second, **giving an agent more actions or tools can expose a flaw in its reward** that was invisible before.

## Mathematical Formulation

The agent maximises expected discounted reward under the reward function $r$ it was given:

$$\pi^{*}_{r} = \arg\max_{\pi} \; \mathbb{E}_{\pi}\Big[\sum_{t=0}^{\infty} \gamma^{t}\, r(s_t, a_t, s_{t+1})\Big]$$

Reward hacking is the case where $\pi^{*}_{r}$ scores badly on the performance measure $M$ the designer actually cares about:

$$M(\pi^{*}_{r}) \ll \max_{\pi} M(\pi)$$

where:
- $r(s, a, s')$: the designer-written reward for moving from state $s$ to $s'$ with action $a$
- $\gamma$: the [[Discount Factor]]
- $\pi$: a [[Policy]]; $\pi^{*}_{r}$ the policy optimal for $r$
- $M$: the performance measure (in the vacuum world, the fraction of floor that is clean)

## Key Properties / Variants

- **Reward versus performance measure.** The reward is what the agent maximises; the performance measure is how we judge it. Reward hacking lives in the gap between the two.
- **Design rule** (Russell and Norvig): design performance measures according to **what you want achieved in the environment** (a clean floor) rather than **how you think the agent should behave** (sucking up dirt). Bob's reward follows the rule; Alice's rewards an activity, and an activity can be manufactured.
- **Not every odd behaviour is hacking.** With Bob's reward and no new dirt, once the floor is clean every action earns the same reward, so the robot may shuttle back and forth or stay put at random. That looks pointless but is perfectly rational for its reward; adding a small cost per move fixes it.
- **Learned rewards are hackable too** (background beyond the course material). When the reward comes from a model trained on human feedback, the policy can find outputs that fool the reward model; the KL penalty in [[Reinforcement Learning from Human Feedback]] exists to limit this.
- **Why it matters for LLM agents.** LLM agents are increasingly trained with reinforcement learning, and they are given ever more tools, which is exactly the situation where the Dump example bites.
- **Related ideas** (background beyond the course material): Goodhart's law (in Marilyn Strathern's phrasing, "when a measure becomes a target, it ceases to be a good measure"); "specification gaming" is the broader term covering any exploit of a misspecified objective.

## Connections

- The agent is still a [[Rational Agent]]: the failure is in the performance measure or [[Reward Signal]]
- Setting: [[Reinforcement Learning]], [[Markov Decision Process]], planned with [[Value Iteration]]
- Learned-reward version: [[Reinforcement Learning from Human Feedback]]
- Modern concern: [[LLM Agent]]
- Multi-agent version: payoffs that make deception pay in a [[Multi-Agent System]]

## Appears In

- [[AGAI-Lab01 - Simple Agents, Complex Dynamics]]
- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
- [[AGAI-Book Ch2 - Intelligent Agents]]
