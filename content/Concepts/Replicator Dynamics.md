---
type: concept
aliases: [replicator equation]
course: [AGAI]
tags: [game-theory, multi-agent, evolutionary-game-theory, exam-topic]
status: complete
---

# Replicator Dynamics

## Definition

> [!definition] Replicator equation
> In a large population where agents are repeatedly paired at random to play a game, the **replicator equation** says that a strategy becomes more common when it earns more than the population average, at a rate proportional to **how much more it earns** and to **how common it already is**.

It is the central tool of evolutionary game theory: instead of asking what rational agents would compute, it asks what adaptation produces in the long run.

## Intuition

> [!intuition] What works spreads
> Many agents do not compute equilibria; they **adapt**. A trading bot keeps the rule that made money, a reinforcement learner repeats actions that paid off, a platform deploys more copies of the agent that sells best. Model the population as a mix of strategies. Strategies doing better than average grow their share, worse-than-average ones shrink. The factor "how common it already is" means a strategy nobody plays cannot grow, and growth is fastest when there is both a lot of it and a large advantage.

The same equation describes three things: biological evolution (fitter genes replicate faster, where the method came from), **social learning** (agents occasionally meet a random peer and copy its strategy with probability proportional to how much better the peer is doing), and a simple form of **reinforcement learning** (Cross learning with a small learning rate, as shown by Börgers and Sarin, 1997).

## Mathematical Formulation

**Two strategies.** Let $x$ be the fraction of cooperators (C), $1 - x$ the fraction of defectors (D), with payoffs $R, S, T, P$ as in the [[Prisoner's Dilemma]]. A random partner cooperates with probability $x$, so

$$f_C(x) = xR + (1-x)S, \qquad f_D(x) = xT + (1-x)P, \qquad \bar f(x) = x f_C(x) + (1-x) f_D(x)$$

$$\frac{dx}{dt} = x\big(f_C(x) - \bar f(x)\big) = x(1-x)\big(f_C(x) - f_D(x)\big)$$

where:
- $f_C, f_D$: expected payoff (fitness) of a cooperator and a defector
- $\bar f$: average payoff in the population
- $x(1-x)$: zero when either strategy is absent, largest at $x = 1/2$
- $f_C - f_D$: the payoff advantage of cooperating; its sign alone decides the direction of change

The second form follows from substituting $\bar f$: $f_C - \bar f = f_C - x f_C - (1-x) f_D = (1-x)(f_C - f_D)$.

**$k$ strategies.** With frequencies $\mathbf{x} = (x_1, \dots, x_k)$ and payoff matrix $A$ (entry $A_{ij}$ is the payoff of strategy $i$ against $j$):

$$\frac{dx_i}{dt} = x_i\Big((A\mathbf{x})_i - \mathbf{x}^{\top} A\, \mathbf{x}\Big)$$

where $(A\mathbf{x})_i$ is the expected payoff of strategy $i$ against the current population and $\mathbf{x}^{\top} A \mathbf{x}$ is the population average.

## Key Properties / Variants

- **Rest points.** A **rest point** (fixed point) is a state where $dx/dt = 0$. For two strategies, $x = 0$ and $x = 1$ always are; an interior one exists where $f_C = f_D$, at $x^{*} = \frac{P - S}{(R - T) + (P - S)}$ if that lies strictly between 0 and 1.
- **Stability.** A rest point is **stable** if the population returns after a small push in either direction, unstable otherwise. Its **basin of attraction** is the set of starting states that end there. Read it off the **phase line** ($dx/dt$ against $x$): above zero, cooperators increase.
- **Prisoner's Dilemma.** $f_C - f_D = x(R-T) + (1-x)(S-P) < 0$ everywhere (greed and fear), so cooperation always declines. $x = 0$ is stable and matches the Nash equilibrium (D,D); $x = 1$ is unstable because one defector out-earns everyone and spreads. A population starting at 90% cooperators still ends at all-defect.
- **Stag Hunt.** $f_C - f_D$ is negative below $x^{*}$ and positive above. Rest points: $x = 0$ stable, $x^{*} = 2/3$ unstable (for $R=4, S=0, T=3, P=2$), $x = 1$ stable. Starting at 60% cooperators the population goes to all-defect. The interior point is the game's mixed Nash equilibrium, but an adaptive population never settles there: any push sends it to one of the pure equilibria.
- **Belief and frequency coincide.** The tipping point $x^{*}$ equals the belief threshold $p^{*}$ of a rational Stag Hunt player, because in a randomly mixed population the chance that a partner cooperates is just $x$.
- **Several populations.** Separate groups can each follow their own two-strategy equation coupled through payoffs, as in the signalling game of AGAI Lab 1, where high-quality senders, low-quality senders (who may deceive) and receivers (trusting or sceptical) co-evolve.
- **Link to equilibria** (background beyond the course material): every stable rest point is a Nash equilibrium, and every strict Nash equilibrium is stable.

## Connections

- Games analysed: [[Prisoner's Dilemma]], [[Stag Hunt]], strategies of a [[Repeated Game]]
- Rest points versus: [[Nash Equilibrium]]
- Learning interpretation: [[Reinforcement Learning]] (Cross learning)
- Setting: populations in a [[Multi-Agent System]]

## Appears In

- [[AGAI-Lab01 - Simple Agents, Complex Dynamics]]
- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
