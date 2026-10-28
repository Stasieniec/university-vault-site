---
type: concept
aliases: [agent programs, simple reflex agent, model-based reflex agent, goal-based agent, utility-based agent, learning agent]
course: [AGAI]
tags: [agents]
status: complete
---

# Agent Program

## Definition

> [!definition] Agent program
> The **agent program** is the concrete implementation of an agent function: it takes the **current percept** from the sensors and returns an action to the actuators. It runs on an architecture, so $\text{agent} = \text{architecture} + \text{program}$. The agent function may depend on the whole percept history; the program sees only the current percept, so if history matters it must remember it.

Russell and Norvig describe four basic designs of increasing capability (simple reflex, model-based reflex, goal-based, utility-based), and any of them can be turned into a learning agent.

## Intuition

> [!intuition] From a giant table to a small program
> The obvious program is a lookup table indexed by percept sequences. It is doomed: with percept set $\mathcal{P}$ and lifetime $T$ the table has $\sum_{t=1}^{T} |\mathcal{P}|^{t}$ entries, which for chess alone is at least $10^{150}$. Nobody could store it, write it, or learn it. The challenge of AI is to produce rational behaviour from a small program, the way a five-line implementation of Newton's method replaced tables of square roots.

## Key Properties / Variants

**Simple reflex agent.** Acts on the current percept only, through **condition-action rules** ("if car-in-front-is-braking then initiate-braking"). Small and fast, but it works only if the right decision can be made from the current percept, which means the environment must be fully observable. In a partially observable world it can loop forever (a vacuum with no location sensor that always moves Left after seeing Clean). Randomising the action can break such loops.

```pseudo
function REFLEX-VACUUM-AGENT([location, status]) returns an action
  if status = Dirty then return Suck
  else if location = A then return Right
  else if location = B then return Left
```

**Model-based reflex agent.** Keeps an **internal state** that tracks the parts of the world it cannot currently see. Updating that state needs two kinds of knowledge: a **transition model** (how the world changes, including the effects of the agent's own actions) and a **sensor model** (how the world state shows up in percepts). The state is a best guess; then the agent picks an action by rule matching, like the reflex agent.

```pseudo
function MODEL-BASED-REFLEX-AGENT(percept) returns an action
  persistent: state, transition model, sensor model, rules, action
  state  ← UPDATE-STATE(state, action, percept, transition model, sensor model)
  rule   ← RULE-MATCH(state, rules)
  action ← rule.ACTION
  return action
```

**Goal-based agent.** Adds **goal** information (desirable situations, such as a destination) and uses the model to ask "what will happen if I do this, and does it reach the goal?". Choosing a sequence of actions is the job of search and planning. Less efficient than a reflex agent but far more flexible: change the goal and the behaviour changes, while a reflex agent's rules would all need rewriting.

**Utility-based agent.** Goals only split states into happy and unhappy. A **utility function** scores states by how desirable they are; it is an internalisation of the performance measure. If the two agree, maximising utility is rational. Utility handles two cases where goals fail: conflicting goals (speed versus safety) and several goals none of which is certain to be reached (weighing the likelihood of success against importance). Under uncertainty the agent maximises **expected utility**:

$$a^{*} = \arg\max_{a} \sum_{s'} P(s' \mid s, a)\, U(s')$$

where:
- $s$: the agent's current (estimated) state
- $P(s' \mid s, a)$: the transition model, the probability of reaching $s'$ by doing $a$ in $s$
- $U(s')$: the utility of outcome state $s'$

**Learning agent.** Any of the designs above can learn. Four components:
- **Performance element**: what the earlier designs called the whole agent; it selects actions.
- **Learning element**: improves the performance element.
- **Critic**: tells the learning element how well the agent is doing against a fixed **performance standard** that the agent must not modify. Needed because a percept ("checkmate") does not say by itself whether it is good.
- **Problem generator**: suggests exploratory actions that may be suboptimal now but lead to informative experience (see [[Exploration vs Exploitation]]).

Learning, in Russell and Norvig's summary, is "a process of modification of each component of the agent to bring the components into closer agreement with the available feedback information, thereby improving the overall performance of the agent." The performance standard singles out part of the percept as a **reward**, which links this design to [[Reinforcement Learning]]. In multi-agent settings learning is harder because the other learners make the environment non-stationary.

> [!example] Agents in AGAI Lab 1
> The Prisoner's Dilemma strategies are deliberately simple: memory-zero strategies (AllC, AllD) ignore history, and memory-one strategies (Tit-for-Tat, Win-Stay Lose-Shift) map the last round's outcome straight to an action by a fixed rule (see [[Repeated Game]]). The vacuum robot in the reward-hacking part is a model-based, utility-based agent: it knows the transition model and plans with [[Value Iteration]] to maximise discounted reward.

## Connections

- Implements the agent function of an [[Intelligent Agent]]
- Choice of design driven by the [[Task Environment]] (partial observability needs internal state, see [[Partial Observability]] and [[Belief State]])
- Utility-based design is the operational form of a [[Rational Agent]]
- Transition model: [[Model of the Environment]]; model-free learners skip it ([[Model-Based Reinforcement Learning]] for the contrast)
- Learning agent: [[Reinforcement Learning]], [[Reward Signal]]
- Modern variant: [[LLM Agent]], where the LLM plays the decision role

## Appears In

- [[AGAI-Book Ch2 - Intelligent Agents]]
- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
- [[AGAI-Lab01 - Simple Agents, Complex Dynamics]]
